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
                             attention_mantissa_bits: int = None,
                             learnable_bias_correction: bool = False,
                             learnable_bc_init: float = 0.0,
                             ste_mode: str = None) -> nn.Module:
    """Patch Conv2d, ConvTranspose2d, and Linear modules in-place.

    `mantissa_bits` controls conv/linear L-Mul precision (legacy global).
    `attention_mantissa_bits` overrides MHA's L-Mul precision; when None,
    falls back to `mantissa_bits` (legacy behaviour).

    Per-type overrides via environment variables (for ablation/probes):
      BCCTN_CONV_M           overrides Conv2d mantissa_bits
      BCCTN_CONV_TRANSPOSE_M overrides ConvTranspose2d mantissa_bits
      BCCTN_LINEAR_M         overrides Linear mantissa_bits

    Single-layer swap (for heatmap hot-layer validation):
      BCCTN_FORCE_REAL_LAYERS  comma-separated layer NAMES to skip L-Mul on
                               (those layers keep stock PyTorch forward → real mult)
                               Layer name format matches model.named_modules().

    Per-channel bias-correction — two modes (mutually exclusive):

    Mode A: fixed offline calibration (post-hoc patch).
      BCCTN_BIAS_CORRECTION_PATH  path to a .pt file containing a dict
                                  {layer_name: tensor[C_out]} of additive
                                  correction values. When present, each
                                  patched Conv2d / ConvTranspose2d / Linear
                                  adds δ broadcast to its L-Mul output. The δ
                                  is meant to compensate the *local* L-Mul
                                  arithmetic offset (NOT teacher-vs-student
                                  distillation residual). Calibrate with
                                  scripts/calibrate_bias_correction.py.
      BCCTN_BIAS_CORRECTION_ALPHA scalar multiplier applied to every δ.
                                  Default 1.0. Use 0.0 / 0.1 / 0.25 / 0.5
                                  for post-QAT over-correction sweeps.

    Mode B: learnable bias-correction during QAT.
      `learnable_bias_correction=True` registers an nn.Parameter[C_out]
      named `learnable_bc` on every patched Conv/ConvT/Linear, initialised
      to `learnable_bc_init` (default 0.0). The parameter participates in
      `model.parameters()` so the trainer's optimizer updates it. Forward
      adds it broadcast to the L-Mul output. Saved/loaded via state_dict
      so curriculum stages can persist learned δ across resumes.
      Mutually exclusive with Mode A (BCCTN_BIAS_CORRECTION_PATH unset).
      Hardware deployment note: at inference time these δ values can be
      folded into each layer's existing `bias` parameter (or kept as a
      cheap C_out-sized add), preserving the "all-L-Mul" multiplier story.

    STE backward variant for L-Mul (training-only; inference unchanged):
      `ste_mode` (or env var BCCTN_STE_MODE) selects the backward path:
        "real"             legacy: backward = real-mult identity STE
        "autograd_lmul"    backward = autograd through L-Mul forward expression
                           (gradient naturally damped by ~0.5–0.67×; OOM @ b128)
        "lmul_exact"       backward = analytic L-Mul gradient (pow2floor): ≈
                           autograd_lmul (differs only at clamp-saturated bins),
                           real-STE memory cost
        "grad_scale_<f>"   backward = real-mult × scalar f (e.g. "grad_scale_0.67")
      Inference / no-grad path is bit-exact identical across modes. Set via
      config to `model.low_power.all_lmul_multiply.ste_mode`. When `ste_mode`
      is provided, this function sets BCCTN_STE_MODE so that all subsequent
      L-Mul multiply / matmul calls (including MHA datapath) inherit it.
    """
    import os as _os
    counts = {"conv2d": 0, "conv_transpose2d": 0, "linear": 0, "attention": 0,
              "skipped_real": 0, "bias_corr_applied": 0}
    mha_mbits = int(attention_mantissa_bits) if attention_mantissa_bits is not None else int(mantissa_bits)
    conv_mbits = int(_os.environ.get("BCCTN_CONV_M") or mantissa_bits)
    convt_mbits = int(_os.environ.get("BCCTN_CONV_TRANSPOSE_M") or mantissa_bits)
    lin_mbits = int(_os.environ.get("BCCTN_LINEAR_M") or mantissa_bits)

    # Publish STE mode via env var so every downstream _lmul_multiply_ste /
    # _lmul_matmul call inherits it — including those buried inside the MHA
    # custom forward path. Explicit kwarg wins over pre-existing env var so
    # config can override at runtime; if neither is set, leaves env alone
    # and downstream falls back to "real" by default.
    _active_ste_mode = None
    if ste_mode is not None and str(ste_mode).strip() != "":
        _active_ste_mode = str(ste_mode).strip()
        _os.environ["BCCTN_STE_MODE"] = _active_ste_mode
    else:
        _active_ste_mode = _os.environ.get("BCCTN_STE_MODE", "real")
    # Validate now so we fail loudly before the first forward.
    from DCNN.utils.complexPyTorch.complexLayers import _resolve_ste_mode
    _resolve_ste_mode(_active_ste_mode)

    _bias_corr_path = _os.environ.get("BCCTN_BIAS_CORRECTION_PATH", "")
    _bias_corr_alpha = float(_os.environ.get("BCCTN_BIAS_CORRECTION_ALPHA", "1.0"))
    if learnable_bias_correction and _bias_corr_path:
        raise ValueError(
            "Mutually exclusive: learnable_bias_correction=True cannot be "
            "used together with BCCTN_BIAS_CORRECTION_PATH (fixed offline δ). "
            "Pick one. Learnable δ should start from zeros (or a calibration "
            "warm-start that you bake separately via your own scripts), not "
            "be double-applied with a fixed offset."
        )
    bias_correction_dict = {}
    if _bias_corr_path:
        if not _os.path.isfile(_bias_corr_path):
            raise FileNotFoundError(
                f"BCCTN_BIAS_CORRECTION_PATH does not exist: {_bias_corr_path}"
            )
        try:
            bias_correction_dict = torch.load(_bias_corr_path,
                                              map_location="cpu",
                                              weights_only=True)
        except TypeError:
            bias_correction_dict = torch.load(_bias_corr_path, map_location="cpu")
        if not isinstance(bias_correction_dict, dict):
            raise ValueError(
                f"bias-correction .pt must be a dict[layer_name -> tensor], "
                f"got {type(bias_correction_dict).__name__}"
            )
        bias_correction_dict = {
            k: v for k, v in bias_correction_dict.items()
            if k != "_meta"
        }
        valid_bias_names = {
            n for n, m in model.named_modules()
            if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d, nn.Linear))
        }
        invalid = set(bias_correction_dict) - valid_bias_names
        if invalid:
            raise ValueError(
                "BCCTN_BIAS_CORRECTION_PATH contains layer names that do not "
                "match any L-Mul-targetable layer. "
                f"Invalid: {sorted(invalid)[:10]}"
            )
        for layer_name, bc in bias_correction_dict.items():
            if not torch.is_tensor(bc):
                raise TypeError(
                    f"bias-correction for {layer_name} must be a tensor, "
                    f"got {type(bc).__name__}"
                )
            module = dict(model.named_modules())[layer_name]
            expected = module.out_features if isinstance(module, nn.Linear) else module.out_channels
            if bc.numel() != expected:
                raise ValueError(
                    f"bias-correction for {layer_name} has {bc.numel()} values, "
                    f"expected {expected}"
                )
        if _bias_corr_alpha != 1.0:
            bias_correction_dict = {
                k: v * _bias_corr_alpha for k, v in bias_correction_dict.items()
            }
        print(f"[L-Mul patch] loaded bias-correction for "
              f"{len(bias_correction_dict)} layers from {_bias_corr_path} "
              f"(alpha={_bias_corr_alpha})")

    _force_real_raw = _os.environ.get("BCCTN_FORCE_REAL_LAYERS", "")
    force_real_names = set(s.strip() for s in _force_real_raw.split(",") if s.strip())
    if force_real_names:
        # Validate every requested name matches an L-Mul-targetable module
        # (Conv2d / ConvTranspose2d / Linear). A typo or a stale CSV name would
        # otherwise silently produce a "swap" run identical to the control.
        valid_target_names = {
            n for n, m in model.named_modules()
            if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d, nn.Linear))
        }
        invalid = force_real_names - valid_target_names
        if invalid:
            raise ValueError(
                "BCCTN_FORCE_REAL_LAYERS contains names that don't match any "
                "L-Mul-targetable layer (Conv2d/ConvTranspose2d/Linear). "
                f"Invalid: {sorted(invalid)}. "
                "Check the heatmap CSV's name column or model.named_modules()."
            )
        print(f"[L-Mul patch] forcing real-mult on {len(force_real_names)} layer(s): "
              f"{sorted(force_real_names)}")

    counts["learnable_bc_registered"] = 0
    for name, module in model.named_modules():
        if name in force_real_names:
            # Skip L-Mul patching for this specific layer → keeps stock forward
            counts["skipped_real"] += 1
            continue

        # Attach bias-correction tensor (if any) BEFORE patching forward so the
        # closure sees it via self.bias_correction. None when no entry exists
        # for this layer, the patched forward skips the add.
        _bc = bias_correction_dict.get(name, None)
        if _bc is not None:
            module.bias_correction = _bc  # tensor [C_out]; moved to device on first forward
        else:
            module.bias_correction = None

        # Register learnable per-channel δ as nn.Parameter BEFORE we patch
        # forward. Initialised to `learnable_bc_init` (typically 0.0); the
        # trainer's optimizer picks it up via self.parameters() automatically.
        if learnable_bias_correction and isinstance(
            module, (nn.Conv2d, nn.ConvTranspose2d, nn.Linear)
        ):
            c_out = (
                module.out_features
                if isinstance(module, nn.Linear)
                else module.out_channels
            )
            # Inherit the module weight's device/dtype so the new parameter
            # lands on the same device regardless of whether the model was
            # already moved to GPU before patching (e.g. standalone smokes
            # call .to(cuda) before enable_all_lmul_multiply, whereas the
            # Lightning flow moves the model after patching).
            ref = module.weight
            module.register_parameter(
                "learnable_bc",
                nn.Parameter(torch.full(
                    (c_out,), float(learnable_bc_init),
                    device=ref.device, dtype=ref.dtype,
                )),
            )
            counts["learnable_bc_registered"] += 1

        if isinstance(module, nn.Conv2d):
            def forward(self, input):
                if self.padding_mode != "zeros":
                    input = F.pad(input, self._reversed_padding_repeated_twice, mode=self.padding_mode)
                    padding = (0, 0)
                else:
                    padding = self.padding
                out = _lmul_conv2d(
                    input, self.weight, self.bias, self.stride, padding,
                    self.dilation, self.groups, conv_mbits, bit_exact,
                    linear_chunk_size, checkpoint_chunks,
                )
                bc = self.bias_correction
                if bc is not None:
                    if bc.device != out.device or bc.dtype != out.dtype:
                        bc = bc.to(device=out.device, dtype=out.dtype)
                        self.bias_correction = bc  # cache moved tensor
                    out = out + bc.view(1, -1, 1, 1)
                lbc = getattr(self, "learnable_bc", None)
                if lbc is not None:
                    out = out + lbc.view(1, -1, 1, 1)
                return out
            module.forward = types.MethodType(forward, module)
            counts["conv2d"] += 1
            if _bc is not None:
                counts["bias_corr_applied"] += 1

        elif isinstance(module, nn.ConvTranspose2d):
            def forward(self, input, output_size=None):
                if output_size is not None:
                    raise ValueError("all-L-Mul ConvTranspose2d does not support explicit output_size")
                out = _lmul_conv_transpose2d(
                    input, self.weight, self.bias, self.stride, self.padding,
                    self.output_padding, self.groups, self.dilation,
                    convt_mbits, bit_exact, linear_chunk_size, checkpoint_chunks,
                )
                bc = self.bias_correction
                if bc is not None:
                    if bc.device != out.device or bc.dtype != out.dtype:
                        bc = bc.to(device=out.device, dtype=out.dtype)
                        self.bias_correction = bc
                    out = out + bc.view(1, -1, 1, 1)
                lbc = getattr(self, "learnable_bc", None)
                if lbc is not None:
                    out = out + lbc.view(1, -1, 1, 1)
                return out
            module.forward = types.MethodType(forward, module)
            counts["conv_transpose2d"] += 1
            if _bc is not None:
                counts["bias_corr_applied"] += 1

        elif isinstance(module, nn.Linear):
            def forward(self, input):
                out = _lmul_linear(
                    input, self.weight, self.bias, lin_mbits, bit_exact,
                    linear_chunk_size, checkpoint_chunks,
                )
                bc = self.bias_correction
                if bc is not None:
                    if bc.device != out.device or bc.dtype != out.dtype:
                        bc = bc.to(device=out.device, dtype=out.dtype)
                        self.bias_correction = bc
                    # Linear output [..., C_out] — last-dim broadcast
                    out = out + bc
                lbc = getattr(self, "learnable_bc", None)
                if lbc is not None:
                    out = out + lbc
                return out
            module.forward = types.MethodType(forward, module)
            counts["linear"] += 1
            if _bc is not None:
                counts["bias_corr_applied"] += 1

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
        f"skipped_real={counts['skipped_real']}; "
        f"bit_exact={bit_exact}, linear_chunk_size={linear_chunk_size}, "
        f"checkpoint_chunks={checkpoint_chunks}"
    )
    if learnable_bias_correction:
        print(
            f"[L-Mul patch] learnable bias-correction enabled: "
            f"{counts['learnable_bc_registered']} nn.Parameter(C_out) registered, "
            f"init={learnable_bc_init}"
        )
    print(f"[L-Mul patch] STE backward mode: {_active_ste_mode}")
    return model


def enable_outlier_fp8_lmul(model: nn.Module, mantissa_bits: int = 4,
                            bit_exact: bool = True) -> nn.Module:
    """L-Mul ONLY the FP8-preserved decoder_pw_outlier conv layers (FP8×FP8 L-Mul).

    Full-multiplier-free target: W4A8 conv/proj stay real-mult (= INT4 shift-add
    exact proxy) + pow2 scale, qk/pv are L-Mul via the attention datapath, and
    these ~12 FP8 outlier pointwise convs become FP8×FP8 L-Mul (mantissa-add, no
    multiplier). FP8 operands are 3-bit, so m4 captures the mantissa losslessly.

    Patches ONLY the modules `enable_mixed_precision_qat` tagged in
    `model._fp8_preserved_names`. Does NOT touch MHA, INT4 conv, or Linear —
    unlike `enable_all_lmul_multiply`, which would also force MHA-projection
    L-Mul. Must be called AFTER enable_mixed_precision_qat.
    """
    names = set(getattr(model, "_fp8_preserved_names", []) or [])
    if not names:
        print("[outlier-lmul] WARN: model._fp8_preserved_names empty — nothing "
              "patched. (enable_mixed_precision_qat must run first.)")
        return model
    m = int(mantissa_bits)
    be = bool(bit_exact)
    patched = []
    for name, module in model.named_modules():
        if name not in names:
            continue
        if isinstance(module, nn.Conv2d):
            def forward(self, input, _m=m, _be=be):
                if self.padding_mode != "zeros":
                    input = F.pad(input, self._reversed_padding_repeated_twice,
                                  mode=self.padding_mode)
                    padding = (0, 0)
                else:
                    padding = self.padding
                return _lmul_conv2d(
                    input, self.weight, self.bias, self.stride, padding,
                    self.dilation, self.groups, _m, _be, 0, False,
                )
            module.forward = types.MethodType(forward, module)
            patched.append(name)
        elif isinstance(module, nn.ConvTranspose2d):
            def forward(self, input, output_size=None, _m=m, _be=be):
                if output_size is not None:
                    raise ValueError("outlier L-Mul ConvTranspose2d: explicit output_size unsupported")
                return _lmul_conv_transpose2d(
                    input, self.weight, self.bias, self.stride, self.padding,
                    self.output_padding, self.groups, self.dilation, _m, _be, 0, False,
                )
            module.forward = types.MethodType(forward, module)
            patched.append(name)
    missing = sorted(names - set(patched))
    print(f"[outlier-lmul] FP8×FP8 L-Mul (m={m}) patched on {len(patched)} "
          f"decoder_pw_outlier layers: {sorted(patched)}")
    if missing:
        print(f"[outlier-lmul] WARN: {len(missing)} tagged names not Conv2d/ConvT, "
              f"left unpatched: {missing}")
    return model
