"""Patch MAC-bearing layers to use the L-Mul product approximation.

This module is intentionally eval-oriented. It keeps the trained fake-quant
parametrizations in place, but replaces Conv/ConvTranspose/Linear products with
the same L-Mul multiply primitive used by the attention feasibility tests.
"""

import types
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from DCNN.utils.complexPyTorch import complexLayers as cl


def _pair(v) -> Tuple[int, int]:
    if isinstance(v, tuple):
        return int(v[0]), int(v[1])
    return int(v), int(v)


def _multiply(x, y, mantissa_bits: int, bit_exact: bool):
    return cl._lmul_multiply_ste(x, y, mantissa_bits, bit_exact)


def _lmul_linear(input, weight, bias=None, mantissa_bits: int = 4,
                 bit_exact: bool = False, chunk_size: int = 0,
                 checkpoint_chunks: bool = False):
    return cl._lmul_linear(
        input, weight, bias,
        mantissa_bits=mantissa_bits,
        bit_exact=bit_exact,
        chunk_size=chunk_size,
        checkpoint_chunks=checkpoint_chunks,
    )


def _lmul_pointwise_conv2d(input, weight, bias, groups, mantissa_bits: int,
                           bit_exact: bool, linear_chunk_size: int,
                           checkpoint_chunks: bool):
    n, in_channels, h, w = input.shape
    out_channels = weight.shape[0]
    in_per_group = in_channels // groups
    out_per_group = out_channels // groups
    x_flat = input.permute(0, 2, 3, 1).reshape(n * h * w, in_channels)
    pieces = []
    for group_idx in range(groups):
        in_start = group_idx * in_per_group
        in_end = in_start + in_per_group
        out_start = group_idx * out_per_group
        out_end = out_start + out_per_group
        w_g = weight[out_start:out_end, :, 0, 0]
        y_g = _lmul_linear(
            x_flat[:, in_start:in_end],
            w_g,
            None,
            mantissa_bits=mantissa_bits,
            bit_exact=bit_exact,
            chunk_size=linear_chunk_size,
            checkpoint_chunks=checkpoint_chunks,
        )
        pieces.append(y_g)
    out = torch.cat(pieces, dim=-1).reshape(n, h, w, out_channels).permute(0, 3, 1, 2)
    if bias is not None:
        out = out + bias.view(1, -1, 1, 1)
    return out


def _lmul_conv2d(input, weight, bias, stride, padding, dilation, groups,
                 mantissa_bits: int, bit_exact: bool, linear_chunk_size: int,
                 checkpoint_chunks: bool):
    stride = _pair(stride)
    padding = _pair(padding)
    dilation = _pair(dilation)
    n, in_channels, h, w = input.shape
    out_channels, channels_per_group, kh, kw = weight.shape

    if (
        kh == 1 and kw == 1
        and stride == (1, 1)
        and padding == (0, 0)
        and dilation == (1, 1)
    ):
        return _lmul_pointwise_conv2d(
            input, weight, bias, groups, mantissa_bits, bit_exact,
            linear_chunk_size, checkpoint_chunks,
        )

    out_h = (h + 2 * padding[0] - dilation[0] * (kh - 1) - 1) // stride[0] + 1
    out_w = (w + 2 * padding[1] - dilation[1] * (kw - 1) - 1) // stride[1] + 1

    cols = F.unfold(input, (kh, kw), dilation=dilation, padding=padding, stride=stride)
    spatial = cols.shape[-1]
    out_dtype = torch.promote_types(input.dtype, weight.dtype)

    if groups == in_channels and channels_per_group == 1 and out_channels == in_channels:
        weight_dw = weight.reshape(out_channels, kh * kw).view(1, out_channels, kh * kw, 1)
        pieces = []
        chunk = int(linear_chunk_size or 0)
        if chunk <= 0:
            chunk = spatial
        cols_dw = cols.reshape(n, in_channels, kh * kw, spatial)
        for start in range(0, spatial, chunk):
            cols_chunk = cols_dw[..., start:start + chunk]
            products = _multiply(cols_chunk, weight_dw, mantissa_bits, bit_exact)
            pieces.append(products.sum(dim=2).to(out_dtype))
        out = torch.cat(pieces, dim=-1)
    else:
        in_per_group = in_channels // groups
        out_per_group = out_channels // groups
        pieces = []
        for group_idx in range(groups):
            col_start = group_idx * in_per_group * kh * kw
            col_end = col_start + in_per_group * kh * kw
            out_start = group_idx * out_per_group
            out_end = out_start + out_per_group
            x_g = cols[:, col_start:col_end, :].transpose(1, 2)
            w_g = weight[out_start:out_end].reshape(out_per_group, -1).transpose(0, 1)
            y_g = _lmul_linear(
                x_g, w_g.transpose(0, 1), None,
                mantissa_bits=mantissa_bits,
                bit_exact=bit_exact,
                chunk_size=linear_chunk_size,
                checkpoint_chunks=checkpoint_chunks,
            )
            pieces.append(y_g.transpose(1, 2))
        out = torch.cat(pieces, dim=1).to(out_dtype)

    out = out.reshape(n, out_channels, out_h, out_w)
    if bias is not None:
        out = out + bias.view(1, -1, 1, 1)
    return out


def _lmul_conv_transpose2d(input, weight, bias, stride, padding, output_padding,
                           groups, dilation, mantissa_bits: int, bit_exact: bool,
                           linear_chunk_size: int, checkpoint_chunks: bool):
    stride = _pair(stride)
    padding = _pair(padding)
    output_padding = _pair(output_padding)
    dilation = _pair(dilation)
    n, in_channels, h, w = input.shape
    _, out_per_group, kh, kw = weight.shape
    in_per_group = in_channels // groups
    out_channels = out_per_group * groups
    out_h = (h - 1) * stride[0] - 2 * padding[0] + dilation[0] * (kh - 1) + output_padding[0] + 1
    out_w = (w - 1) * stride[1] - 2 * padding[1] + dilation[1] * (kw - 1) + output_padding[1] + 1

    out_dtype = torch.promote_types(input.dtype, weight.dtype)
    out = input.new_zeros((n, out_channels, out_h, out_w), dtype=out_dtype)
    out_flat = out.view(n, out_channels, -1)

    h_in = torch.arange(h, device=input.device)
    w_in = torch.arange(w, device=input.device)

    for kh_idx in range(kh):
        oh_all = h_in * stride[0] - padding[0] + kh_idx * dilation[0]
        valid_h = (oh_all >= 0) & (oh_all < out_h)
        if not bool(valid_h.any()):
            continue
        h_sel = h_in[valid_h]
        oh = oh_all[valid_h]

        for kw_idx in range(kw):
            ow_all = w_in * stride[1] - padding[1] + kw_idx * dilation[1]
            valid_w = (ow_all >= 0) & (ow_all < out_w)
            if not bool(valid_w.any()):
                continue
            w_sel = w_in[valid_w]
            ow = ow_all[valid_w]
            oh_grid, ow_grid = torch.meshgrid(oh, ow, indexing="ij")
            spatial_idx = (oh_grid * out_w + ow_grid).reshape(-1).long()

            x_hw = input.index_select(2, h_sel).index_select(3, w_sel)
            x_flat = x_hw.permute(0, 2, 3, 1).reshape(n, -1, in_channels)

            if groups == in_channels and out_per_group == 1:
                w_dw = weight[:, 0, kh_idx, kw_idx].view(1, 1, in_channels)
                products = _multiply(x_flat, w_dw, mantissa_bits, bit_exact)
                out_flat.index_add_(2, spatial_idx, products.transpose(1, 2).to(out_dtype))
            else:
                for group_idx in range(groups):
                    in_start = group_idx * in_per_group
                    in_end = in_start + in_per_group
                    out_start = group_idx * out_per_group
                    out_end = out_start + out_per_group
                    w_g = weight[in_start:in_end, :, kh_idx, kw_idx]
                    y_g = _lmul_linear(
                        x_flat[:, :, in_start:in_end],
                        w_g.transpose(0, 1),
                        None,
                        mantissa_bits=mantissa_bits,
                        bit_exact=bit_exact,
                        chunk_size=linear_chunk_size,
                        checkpoint_chunks=checkpoint_chunks,
                    )
                    out_flat[:, out_start:out_end, :].index_add_(
                        2, spatial_idx, y_g.transpose(1, 2).to(out_dtype)
                    )

    if bias is not None:
        out = out + bias.view(1, -1, 1, 1)
    return out


def enable_all_lmul_multiply(model: nn.Module, mantissa_bits: int = 4,
                             bit_exact: bool = True,
                             linear_chunk_size: int = 0,
                             checkpoint_chunks: bool = False,
                             attention_mantissa_bits: int = None) -> nn.Module:
    """Patch Conv2d, ConvTranspose2d, and Linear modules in-place.

    `mantissa_bits` controls conv/linear L-Mul precision (legacy global).
    `attention_mantissa_bits` overrides MHA's L-Mul precision; when None,
    falls back to `mantissa_bits` (legacy behaviour).

    Per-type overrides via environment variables (for ablation/probes):
      BCCTN_CONV_M           overrides Conv2d mantissa_bits
      BCCTN_CONV_TRANSPOSE_M overrides ConvTranspose2d mantissa_bits
      BCCTN_LINEAR_M         overrides Linear mantissa_bits
    """
    import os as _os
    counts = {"conv2d": 0, "conv_transpose2d": 0, "linear": 0, "attention": 0}
    mha_mbits = int(attention_mantissa_bits) if attention_mantissa_bits is not None else int(mantissa_bits)
    conv_mbits = int(_os.environ.get("BCCTN_CONV_M") or mantissa_bits)
    convt_mbits = int(_os.environ.get("BCCTN_CONV_TRANSPOSE_M") or mantissa_bits)
    lin_mbits = int(_os.environ.get("BCCTN_LINEAR_M") or mantissa_bits)

    for module in model.modules():
        if isinstance(module, nn.Conv2d):
            def forward(self, input):
                if self.padding_mode != "zeros":
                    input = F.pad(input, self._reversed_padding_repeated_twice, mode=self.padding_mode)
                    padding = (0, 0)
                else:
                    padding = self.padding
                return _lmul_conv2d(
                    input, self.weight, self.bias, self.stride, padding,
                    self.dilation, self.groups, conv_mbits, bit_exact,
                    linear_chunk_size, checkpoint_chunks,
                )
            module.forward = types.MethodType(forward, module)
            counts["conv2d"] += 1

        elif isinstance(module, nn.ConvTranspose2d):
            def forward(self, input, output_size=None):
                if output_size is not None:
                    raise ValueError("all-L-Mul ConvTranspose2d does not support explicit output_size")
                return _lmul_conv_transpose2d(
                    input, self.weight, self.bias, self.stride, self.padding,
                    self.output_padding, self.groups, self.dilation,
                    convt_mbits, bit_exact, linear_chunk_size, checkpoint_chunks,
                )
            module.forward = types.MethodType(forward, module)
            counts["conv_transpose2d"] += 1

        elif isinstance(module, nn.Linear):
            def forward(self, input):
                return _lmul_linear(
                    input, self.weight, self.bias, lin_mbits, bit_exact,
                    linear_chunk_size, checkpoint_chunks,
                )
            module.forward = types.MethodType(forward, module)
            counts["linear"] += 1

        if module.__class__.__name__ == "ComplexMultiheadAttention":
            datapath = dict(getattr(module, "attention_datapath", {}) or {})
            datapath["enabled"] = True
            datapath["lmul_enabled"] = True
            datapath["lmul_all_projection_ops"] = True
            datapath["lmul_mantissa_bits"] = mha_mbits
            datapath["lmul_bit_exact"] = bool(bit_exact)
            datapath["lmul_linear_chunk_size"] = int(linear_chunk_size or 0)
            datapath["lmul_checkpoint_chunks"] = bool(checkpoint_chunks)
            targets = set(datapath.get("lmul_targets", []) or [])
            targets.update(["qk", "score_v"])
            datapath["lmul_targets"] = sorted(targets)
            module.attention_datapath = datapath
            module.use_custom_forward = True
            counts["attention"] += 1

    print(
        "L-Mul all-MAC patch enabled: "
        f"{counts['conv2d']} Conv2d (m={conv_mbits}), "
        f"{counts['conv_transpose2d']} ConvTranspose2d (m={convt_mbits}), "
        f"{counts['linear']} Linear (m={lin_mbits}), "
        f"{counts['attention']} attention modules (m={mha_mbits}); "
        f"bit_exact={bit_exact}, linear_chunk_size={linear_chunk_size}, "
        f"checkpoint_chunks={checkpoint_chunks}"
    )
    return model
