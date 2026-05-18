#!/usr/bin/env python3
"""Analytical MAC counter for baseline (mode=off) and each P1 grouped variant.

Reports two views per spec §4.4:
  1. Decoder pointwise eligible-layer MAC (the thing this PR actually changes).
  2. Full-model MAC (so reduction is not over-claimed).

Analytical because a forward pass through DCNN requires a proper audio batch
through STFT (which only accepts <=2D in older torch). We instead read
Conv2d/ConvTranspose2d dims and compute output spatial size from input shape
+ stride/padding/kernel.

Run:
  conda run -n bcctn python scripts/count_p1_macs.py
"""
from __future__ import annotations
import math
import sys
from pathlib import Path
from collections import OrderedDict

import torch
from omegaconf import OmegaConf
from torch import nn

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from DCNN.models.model import DCNN  # noqa: E402


VARIANTS = [
    ("baseline_off", "off", 1, 1),
    ("g11", "grouped", 1, 1),
    ("g21", "grouped", 2, 1),
    ("g12", "grouped", 1, 2),
    ("g22", "grouped", 2, 2),
    ("g42", "grouped", 4, 2),
    ("g44", "grouped", 4, 4),
]


def build_model(mode: str, gx: int, gs: int) -> nn.Module:
    cfg = OmegaConf.load(REPO / "config" / "config.yaml")
    cfg.model.decoder_grouped_pointwise.mode = mode
    cfg.model.decoder_grouped_pointwise.groups_x = gx
    cfg.model.decoder_grouped_pointwise.groups_skip = gs
    cfg.model.attention_softmax.mode = "shift_topk_pow2_clamp8_nomantissa"
    cfg.training.top_k_attention = 4
    model = DCNN(**cfg.model)
    model.eval()
    return model


def conv2d_out(in_size, kernel, stride, padding, dilation=1):
    """floor((in + 2*pad - dilation*(k-1) - 1) / stride) + 1"""
    return (in_size + 2 * padding - dilation * (kernel - 1) - 1) // stride + 1


def conv_transpose2d_out(in_size, kernel, stride, padding, output_padding, dilation=1):
    """(in - 1)*stride - 2*pad + dilation*(k-1) + output_padding + 1"""
    return (in_size - 1) * stride - 2 * padding + dilation * (kernel - 1) + output_padding + 1


def expected_spatial_sizes(fft_len=128, win_inc=80, n_samples=16000, n_encoder=6):
    """Return list of (F, T) per stage:
       stage 0 = input to encoder layer 0
       stage k = output of encoder layer k-1 (and input to encoder layer k)
       Decoder mirrors with upsampling.

    STFT: onesided fft_len/2+1 bins; the model drops DC bin (-1).
    T = floor((n - win_len)/hop) + 1 with center=False, but actual T is
    irrelevant to RATIOS since it's identical across variants.
    """
    F0 = fft_len // 2  # after dropping DC
    # Encoder downsamples freq by stride=2, kernel=5, pad=2 each layer
    F_enc = [F0]
    for _ in range(n_encoder):
        F_enc.append(conv2d_out(F_enc[-1], kernel=5, stride=2, padding=2))
    # Decoder mirrors (n_encoder layers), input F = F_enc[-1], output F at each step
    F_dec_in = list(reversed(F_enc[:-1]))  # not used, but illustrative
    F_dec_out = []
    f = F_enc[-1]
    for _ in range(n_encoder):
        f_new = conv_transpose2d_out(f, kernel=5, stride=2, padding=2, output_padding=1)
        F_dec_out.append((f, f_new))
        f = f_new

    # Time axis: assume center=False, win_len=128, hop=80, n_samples=16000
    T0 = (n_samples - 128) // win_inc + 1
    return F_enc, F_dec_out, T0


def count_macs_analytical(model: nn.Module, fft_len=128, win_inc=80, n_samples=16000):
    """Walk modules, compute MAC for each Conv2d / ConvTranspose2d.

    For each layer we need (in_H, in_W) or (out_H, out_W). We derive these by
    matching the module name pattern to known stage positions. Returns dict:
      name -> macs
    """
    # Count encoder layers directly from the model
    n_encoder = sum(1 for n, _ in model.named_children() if False)
    n_encoder = 0
    for n, _ in model.named_modules():
        if n.startswith("encoder.model.") and n.count(".") == 2:
            try:
                idx = int(n.split(".")[2])
                n_encoder = max(n_encoder, idx + 1)
            except ValueError:
                pass

    F_enc, F_dec_out, T0 = expected_spatial_sizes(
        fft_len=fft_len, win_inc=win_inc, n_samples=n_samples,
        n_encoder=n_encoder,
    )
    captured = OrderedDict()

    # Encoder layers: encoder.model.<k>.0.<depthwise|pointwise>_<r|i>
    # Decoder layers: decoder.model.<k>.0... (either ComplexDepthwiseSeparable*
    #                  or _DecoderBlockSplit with .split_conv.<pointwise|depthwise>...)
    for name, mod in model.named_modules():
        if not isinstance(mod, (nn.Conv2d, nn.ConvTranspose2d)):
            continue

        kh, kw = mod.kernel_size if isinstance(mod.kernel_size, tuple) else (mod.kernel_size, mod.kernel_size)
        ic = mod.in_channels
        oc = mod.out_channels
        groups = mod.groups

        # Figure out spatial output size from name (encoder.model.X.* / decoder.model.X.*)
        if name.startswith("encoder.model."):
            k = int(name.split("encoder.model.")[1].split(".")[0])
            # Encoder layer k consumes F_enc[k], outputs F_enc[k+1]
            out_F = F_enc[k + 1]
        elif name.startswith("decoder.model."):
            k = int(name.split("decoder.model.")[1].split(".")[0])
            # Decoder layer k input/output F
            in_F, out_F = F_dec_out[k]
            # Pointwise (1x1) output keeps in_F; depthwise transpose changes it
            if "pointwise" in name:
                out_F = in_F
            # depthwise transpose uses out_F as defined
        else:
            # Skip non-encoder/decoder convs (shouldn't exist in this model)
            continue

        out_T = T0
        macs = oc * (ic // groups) * kh * kw * out_F * out_T
        captured[name] = macs

    return captured


def get_decoder_layers(model: nn.Module):
    """Return list of decoder layer names (encoder mirror to know n_encoder)."""
    return [n for n, _ in model.named_modules() if n.endswith(".decoder.model") or n == "decoder.model"]


def is_decoder_pointwise(name: str) -> bool:
    if not name.startswith("decoder.model."):
        return False
    if "pointwise" not in name:
        return False
    return True


def main():
    rows = []
    base_total = base_dec_pw = None
    base_breakdown = None

    for name, mode, gx, gs in VARIANTS:
        model = build_model(mode, gx, gs)
        try:
            captured = count_macs_analytical(model)
        except Exception as e:
            print(f"[{name}] FAILED: {e}", file=sys.stderr)
            del model
            continue
        total = sum(captured.values())
        dec_pw = sum(m for n, m in captured.items() if is_decoder_pointwise(n))
        if name == "baseline_off":
            base_total, base_dec_pw = total, dec_pw
            base_breakdown = captured
        rows.append((name, mode, gx, gs, total, dec_pw))
        del model

    print()
    print("=== P1 MAC ANALYSIS (analytical, per-conv via in/out/groups/spatial) ===")
    print()
    print(f"{'variant':<14} {'mode':<8} {'gx':<3} {'gs':<3} {'dec_pw_MAC':>15} {'Δ dec_pw':>11} {'full_MAC':>15} {'Δ full':>9}")
    print("-" * 92)
    for name, mode, gx, gs, total, dec_pw in rows:
        dec_pw_pct = 100.0 * (dec_pw - base_dec_pw) / base_dec_pw if base_dec_pw else 0.0
        full_pct = 100.0 * (total - base_total) / base_total if base_total else 0.0
        print(f"{name:<14} {mode:<8} {gx:<3} {gs:<3} {dec_pw:>15,} {dec_pw_pct:>+10.2f}% {total:>15,} {full_pct:>+8.2f}%")
    print()
    print(f"baseline decoder-pointwise MAC = {base_dec_pw:,}")
    print(f"baseline full-conv MAC          = {base_total:,}")
    if base_total:
        print(f"decoder-pointwise share of conv = {100.0*base_dec_pw/base_total:.2f}%")

    print()
    print("=== baseline decoder pointwise per-layer ===")
    for n, m in base_breakdown.items():
        if is_decoder_pointwise(n):
            print(f"  {n:<70} {m:>15,}")


if __name__ == "__main__":
    main()
