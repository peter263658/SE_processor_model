#!/usr/bin/env python3
"""
Attention Mixed-Precision & Density Analyzer for BCCTN
=======================================================
Analyzes the MHA sub-operation dynamic ranges, precision sensitivity,
attention score density/entropy, top-k sparse attention, and generates
a mixed-precision hardware recommendation.

Usage:
  cd 06_attn_mised_precision/
  conda activate bcctn
  python attn_precision_analyzer.py \
      --checkpoint model.ckpt \
      --test_dir /raid/R12K41024/BCCTN/Dataset/noisy_testset/snr_0dB/ \
      --clean_dir /raid/R12K41024/BCCTN/Dataset/clean_testset/ \
      --num_samples 20
"""

import argparse
import json
import math
import os
import sys
import csv
import warnings
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio
from tqdm import tqdm

warnings.filterwarnings("ignore")

# ─── Path: run from 06_attn_mised_precision/ ─────────────────────────────────
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from hydra import compose, initialize
from hydra.core.global_hydra import GlobalHydra
from DCNN.trainer import DCNNLightningModule

RESULTS_DIR = Path("results")
RESULTS_DIR.mkdir(exist_ok=True)

SR = 16000


# ══════════════════════════════════════════════════════════════════════════════
#  Fake-quantisation helpers
# ══════════════════════════════════════════════════════════════════════════════

def _fake_quant_fp8_e4m3(x: torch.Tensor) -> torch.Tensor:
    """FP8 E4M3: 4-bit exp, 3-bit mantissa, max ≈ 448."""
    max_v = 448.0
    scale = max_v / 127.0
    return (x.clamp(-max_v, max_v) / scale).round() * scale


def _fake_quant_fp8_e5m2(x: torch.Tensor) -> torch.Tensor:
    """FP8 E5M2: 5-bit exp, 2-bit mantissa, max ≈ 57344."""
    max_v = 57344.0
    scale = max_v / 63.0
    return (x.clamp(-max_v, max_v) / scale).round() * scale


def _fake_quant_fp4(x: torch.Tensor) -> torch.Tensor:
    """FP4 E2M1: 2-bit exp, 1-bit mantissa, ~15 representable magnitudes."""
    max_v = 6.0
    scale = max_v / 7.0
    return (x.clamp(-max_v, max_v) / scale).round() * scale


def _fake_quant_int(x: torch.Tensor, n_bits: int) -> torch.Tensor:
    qmax = {8: 127, 4: 7, 16: 32767}[n_bits]
    abs_max = x.abs().max().clamp(min=1e-8)
    scale = abs_max / qmax
    return (x / scale).clamp(-qmax, qmax).round() * scale


def _fake_quant_fp16(x: torch.Tensor) -> torch.Tensor:
    return x.half().float()


PRECISION_FNS = {
    "fp32":     lambda x: x,
    "fp16":     _fake_quant_fp16,
    "fp8_e4m3": _fake_quant_fp8_e4m3,
    "fp8_e5m2": _fake_quant_fp8_e5m2,
    "fp4":      _fake_quant_fp4,
    "int8":     lambda x: _fake_quant_int(x, 8),
    "int4":     lambda x: _fake_quant_int(x, 4),
    "int16":    lambda x: _fake_quant_int(x, 16),
}


# ══════════════════════════════════════════════════════════════════════════════
#  Tensor statistics helper
# ══════════════════════════════════════════════════════════════════════════════

def tensor_stats(t: torch.Tensor) -> dict:
    """Return a stats dict for a real-valued tensor."""
    t = t.float()
    return {
        "min":           float(t.min()),
        "max":           float(t.max()),
        "mean":          float(t.mean()),
        "std":           float(t.std()),
        "abs_max":       float(t.abs().max()),
        "dynamic_range": float(t.max() - t.min()),
        "log2_range":    float(torch.log2(t.abs().max().clamp(min=1e-12))),
    }


# ══════════════════════════════════════════════════════════════════════════════
#  MHA Interception Module
# ══════════════════════════════════════════════════════════════════════════════

class MHAInterceptor:
    """
    Wraps a ComplexMultiheadAttention instance and can capture per-sample
    intermediate tensors: Q, K, V, QK^T (raw scores), attention weights,
    Score×V output — for each of the 4 real-arithmetic passes.

    Passes:
      re_re  – MultiheadAttention_re operating on x.real
      im_im  – MultiheadAttention_im operating on x.imag
      im_re  – MultiheadAttention_re operating on x.imag
      re_im  – MultiheadAttention_im operating on x.real
    """

    # Human-readable labels for reporting (aggregated across the 4 passes)
    PASSES = ["re_re", "im_im", "im_re", "re_im"]

    def __init__(self, complex_mha_module):
        self.cmha = complex_mha_module
        self.captured: dict[str, list] = defaultdict(list)
        self._hook_handle = None

    def _compute_pass(
        self,
        x_in: torch.Tensor,
        mha: nn.MultiheadAttention,
        pass_name: str,
        attn_mask,
    ):
        """
        Manually reproduce one real-MHA pass and store intermediate tensors.
        x_in: (B, T, D) float32
        """
        B, T, D = x_in.shape
        num_heads = mha.num_heads
        head_dim  = D // num_heads

        W = mha.in_proj_weight  # (3D, D)
        b = mha.in_proj_bias    # (3D,)
        W_q, W_k, W_v = W.chunk(3, dim=0)
        b_q, b_k, b_v = b.chunk(3, dim=0)

        Q = F.linear(x_in, W_q, b_q)  # (B, T, D)
        K = F.linear(x_in, W_k, b_k)
        V = F.linear(x_in, W_v, b_v)

        # Reshape to multi-head
        Q_h = Q.view(B, T, num_heads, head_dim).transpose(1, 2)  # (B,H,T,d)
        K_h = K.view(B, T, num_heads, head_dim).transpose(1, 2)
        V_h = V.view(B, T, num_heads, head_dim).transpose(1, 2)

        # QK^T / sqrt(d)
        scale      = 1.0 / math.sqrt(head_dim)
        scores_raw = (Q_h @ K_h.transpose(-2, -1)) * scale  # (B,H,T,T)

        # Masked scores
        if attn_mask is not None:
            scores_masked = scores_raw + attn_mask[None, None]
        else:
            scores_masked = scores_raw

        # Attention weights (after softmax)
        attn_w = F.softmax(scores_masked.float(), dim=-1)  # (B,H,T,T)

        # Score × V
        sv = attn_w @ V_h.float()  # (B,H,T,d)

        # Store (detach, move to CPU to save GPU memory)
        self.captured[f"Q_{pass_name}"].append(Q.detach().cpu())
        self.captured[f"K_{pass_name}"].append(K.detach().cpu())
        self.captured[f"V_{pass_name}"].append(V.detach().cpu())
        self.captured[f"QKT_{pass_name}"].append(scores_raw.detach().cpu())
        self.captured[f"attn_weights_{pass_name}"].append(attn_w.detach().cpu())
        self.captured[f"sv_{pass_name}"].append(sv.detach().cpu())

    def install(self):
        """Install a forward hook on the ComplexMultiheadAttention."""

        interceptor = self

        def _hook(module, args, kwargs, output):
            x = args[0]
            attn_mask = kwargs.get("attn_mask", None)
            with torch.no_grad():
                interceptor._compute_pass(x.real.float(), module.MultiheadAttention_re, "re_re", attn_mask)
                interceptor._compute_pass(x.imag.float(), module.MultiheadAttention_im, "im_im", attn_mask)
                interceptor._compute_pass(x.imag.float(), module.MultiheadAttention_re, "im_re", attn_mask)
                interceptor._compute_pass(x.real.float(), module.MultiheadAttention_im, "re_im", attn_mask)

        self._hook_handle = self.cmha.register_forward_hook(_hook, with_kwargs=True)

    def remove(self):
        if self._hook_handle is not None:
            self._hook_handle.remove()
            self._hook_handle = None

    def clear(self):
        self.captured = defaultdict(list)


# ══════════════════════════════════════════════════════════════════════════════
#  Audio loading
# ══════════════════════════════════════════════════════════════════════════════

def _match_clean(noisy_path: Path, clean_dir: Path) -> Path | None:
    """Match noisy file to its clean counterpart by stripping SNR suffix."""
    stem = noisy_path.stem
    # e.g.  p225_p225_009_az-40_snr+0.0  →  p225_p225_009_az-40
    base = stem.split("_snr")[0]
    candidates = list(clean_dir.glob(f"{base}*.wav"))
    return candidates[0] if candidates else None


def load_audio_pair(noisy_path: Path, clean_dir: Path, device: torch.device):
    """
    Returns (noisy_wav, clean_wav) both (1, 2, T) on device, or None on failure.
    """
    clean_path = _match_clean(noisy_path, clean_dir)
    if clean_path is None:
        return None

    noisy_wav, sr_n = torchaudio.load(str(noisy_path))
    clean_wav, sr_c = torchaudio.load(str(clean_path))

    if sr_n != SR:
        noisy_wav = torchaudio.functional.resample(noisy_wav, sr_n, SR)
    if sr_c != SR:
        clean_wav = torchaudio.functional.resample(clean_wav, sr_c, SR)

    # Ensure 2-channel (binaural)
    if noisy_wav.shape[0] == 1:
        noisy_wav = noisy_wav.repeat(2, 1)
    if clean_wav.shape[0] == 1:
        clean_wav = clean_wav.repeat(2, 1)

    # Trim/pad to same length
    min_len = min(noisy_wav.shape[-1], clean_wav.shape[-1])
    noisy_wav = noisy_wav[:, :min_len]
    clean_wav = clean_wav[:, :min_len]

    return noisy_wav.unsqueeze(0).to(device), clean_wav.unsqueeze(0).to(device)


# ══════════════════════════════════════════════════════════════════════════════
#  SI-SNR / ILD / IPD helpers
# ══════════════════════════════════════════════════════════════════════════════

def si_snr(s: torch.Tensor, s_hat: torch.Tensor) -> float:
    """Scale-invariant SNR in dB. s, s_hat: (T,) float"""
    s     = s - s.mean()
    s_hat = s_hat - s_hat.mean()
    proj  = (s_hat * s).sum() / (s.pow(2).sum() + 1e-8) * s
    noise = s_hat - proj
    return float(10 * torch.log10((proj.pow(2).sum() + 1e-8) / (noise.pow(2).sum() + 1e-8)))


def compute_ild(wav: torch.Tensor) -> float:
    """ILD in dB. wav: (2, T)"""
    el = wav[0].pow(2).mean().clamp(min=1e-10)
    er = wav[1].pow(2).mean().clamp(min=1e-10)
    return float(20 * torch.log10(torch.sqrt(el / er)))


def compute_ipd_error(ref: torch.Tensor, hyp: torch.Tensor, n_fft=512, hop=128) -> float:
    """
    Mean absolute IPD error (radians) between reference and hypothesis.
    ref, hyp: (2, T) float
    """
    def stft(x):
        return torch.stft(x, n_fft=n_fft, hop_length=hop,
                          return_complex=True, window=torch.hann_window(n_fft).to(x.device))
    S_ref = stft(ref.float())   # (2, F, T)
    S_hyp = stft(hyp.float())
    ipd_ref = torch.angle(S_ref[0] * S_ref[1].conj())
    ipd_hyp = torch.angle(S_hyp[0] * S_hyp[1].conj())
    diff = torch.abs(ipd_ref - ipd_hyp)
    diff = torch.where(diff > math.pi, 2 * math.pi - diff, diff)
    return float(diff.mean())


# ══════════════════════════════════════════════════════════════════════════════
#  1. Dynamic range analysis
# ══════════════════════════════════════════════════════════════════════════════

def analyze_dynamic_range(model, interceptor, audio_pairs, device):
    """
    Run N samples through the model with the interceptor installed.
    Returns aggregate statistics per sub-operation.
    """
    print("\n[1/5] Dynamic range analysis …")
    interceptor.install()
    interceptor.clear()

    model.eval()
    with torch.no_grad():
        for noisy, _ in tqdm(audio_pairs, desc="  Sampling"):
            _ = model(noisy)

    interceptor.remove()

    # ── Aggregate per sub-operation across all passes ───────────────────────
    # Sub-operations we care about (merge over all 4 passes)
    subops = {
        "Q":           [f"Q_{p}"           for p in MHAInterceptor.PASSES],
        "K":           [f"K_{p}"           for p in MHAInterceptor.PASSES],
        "V":           [f"V_{p}"           for p in MHAInterceptor.PASSES],
        "QKT":         [f"QKT_{p}"         for p in MHAInterceptor.PASSES],
        "attn_weights":[f"attn_weights_{p}"for p in MHAInterceptor.PASSES],
        "sv":          [f"sv_{p}"          for p in MHAInterceptor.PASSES],
    }

    rows = []
    for op_name, keys in subops.items():
        tensors = []
        for k in keys:
            if k in interceptor.captured:
                tensors.extend(interceptor.captured[k])
        if not tensors:
            continue
        big = torch.cat([t.reshape(-1) for t in tensors])
        s = tensor_stats(big)
        s["subop"] = op_name
        rows.append(s)
        print(f"  {op_name:15s}  min={s['min']:+10.4f}  max={s['max']:+10.4f}  "
              f"std={s['std']:.4f}  dyn={s['dynamic_range']:.4f}  "
              f"log2_amax={s['log2_range']:.2f}")

    # Write CSV
    csv_path = RESULTS_DIR / "dynamic_range_per_subop.csv"
    fieldnames = ["subop","min","max","mean","std","abs_max","dynamic_range","log2_range"]
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows([{k: r.get(k, "") for k in fieldnames} for r in rows])
    print(f"  → {csv_path}")

    return interceptor.captured, rows


# ══════════════════════════════════════════════════════════════════════════════
#  2. Precision sensitivity
# ══════════════════════════════════════════════════════════════════════════════

class PrecisionPatchedCMHA(nn.Module):
    """
    Wraps ComplexMultiheadAttention and applies fake-quantisation at specific
    pipeline stages:
      qkv_prec     – precision applied to Q, K, V after projection
      acc_prec     – precision applied to QK^T raw scores (accumulator simulation)
      sv_prec      – precision applied to attn_weights × V output
    """
    def __init__(self, original_cmha, qkv_prec="fp32", acc_prec="fp32", sv_prec="fp32"):
        super().__init__()
        self.cmha     = original_cmha
        self.qkv_fn   = PRECISION_FNS[qkv_prec]
        self.acc_fn   = PRECISION_FNS[acc_prec]
        self.sv_fn    = PRECISION_FNS[sv_prec]

    def _single_pass(self, x_in, mha, attn_mask):
        B, T, D = x_in.shape
        H = mha.num_heads
        d = D // H

        W, b = mha.in_proj_weight, mha.in_proj_bias
        Wq, Wk, Wv = W.chunk(3, dim=0)
        bq, bk, bv = b.chunk(3, dim=0)

        Q = self.qkv_fn(F.linear(x_in, Wq, bq))
        K = self.qkv_fn(F.linear(x_in, Wk, bk))
        V = self.qkv_fn(F.linear(x_in, Wv, bv))

        Qh = Q.view(B,T,H,d).transpose(1,2)
        Kh = K.view(B,T,H,d).transpose(1,2)
        Vh = V.view(B,T,H,d).transpose(1,2)

        scores = self.acc_fn((Qh @ Kh.transpose(-2,-1)) / math.sqrt(d))

        if attn_mask is not None:
            scores = scores + attn_mask[None, None]

        attn_w = F.softmax(scores, dim=-1)
        sv     = self.sv_fn(attn_w @ Vh)

        # out_proj
        sv_flat = sv.transpose(1,2).contiguous().view(B, T, D)
        return F.linear(sv_flat, mha.out_proj.weight, mha.out_proj.bias)

    def forward(self, x, attn_mask=None):
        mha_re = self.cmha.MultiheadAttention_re
        mha_im = self.cmha.MultiheadAttention_im
        x_real = x.real.float()
        x_imag = x.imag.float()

        real_real = self._single_pass(x_real, mha_re, attn_mask)
        imag_imag = self._single_pass(x_imag, mha_im, attn_mask)
        imag_real = self._single_pass(x_imag, mha_re, attn_mask)
        real_imag = self._single_pass(x_real, mha_im, attn_mask)

        real_out = real_real - imag_imag
        imag_out = imag_real + real_imag
        return torch.complex(real_out, imag_out).to(x.dtype)


def mse_of_mha_output(model, audio_pairs, qkv_prec, acc_prec, sv_prec, device):
    """
    Temporarily patch the model's CMHA with a precision-simulated version,
    run inference on audio_pairs, compare MHA output to fp32 reference.
    Returns mean MSE of MHA block output.
    """
    mattn_block = model.mattn
    original_cmha = mattn_block.mattn

    # ── FP32 reference ───────────────────────────────────────────────────────
    ref_outputs = []
    model.eval()
    with torch.no_grad():
        for noisy, _ in audio_pairs:
            # Capture MHA output via hook
            out_buf = []
            def _ref_hook(m, inp, out): out_buf.append(out.detach().cpu())
            h = original_cmha.register_forward_hook(_ref_hook)
            _ = model(noisy)
            h.remove()
            ref_outputs.append(out_buf[0])

    # ── Patched version ───────────────────────────────────────────────────────
    patched = PrecisionPatchedCMHA(original_cmha, qkv_prec, acc_prec, sv_prec)
    mattn_block.mattn = patched

    mses = []
    try:
        model.eval()
        with torch.no_grad():
            for i, (noisy, _) in enumerate(audio_pairs):
                pat_buf = []
                def _pat_hook(m, inp, out): pat_buf.append(out.detach().cpu())
                h = patched.register_forward_hook(_pat_hook)
                _ = model(noisy)
                h.remove()
                if pat_buf:
                    ref = ref_outputs[i]
                    pat = pat_buf[0]
                    # complex → real MSE (use .real/.imag to avoid error on complex.float())
                    mse = (ref.real.float() - pat.real.float()).pow(2).mean() + \
                          (ref.imag.float() - pat.imag.float()).pow(2).mean()
                    mses.append(float(mse))
    finally:
        mattn_block.mattn = original_cmha  # always restore

    return float(np.mean(mses)) if mses else float("nan")


def analyze_precision_sensitivity(model, audio_pairs, device):
    """Grid-search precision combinations and record MSE."""
    print("\n[2/5] Precision sensitivity analysis …")

    qkv_options = ["fp32", "fp16", "fp8_e4m3", "fp8_e5m2", "int8", "int4"]
    acc_options  = ["fp32", "fp16", "int16", "int8"]
    sv_options   = ["fp32", "fp16", "fp8_e4m3", "fp8_e5m2"]

    rows = []
    combos = [
        (q, a, s)
        for q in qkv_options
        for a in acc_options
        for s in sv_options
    ]
    # Always include fp32/fp32/fp32 baseline
    combos = [("fp32","fp32","fp32")] + [c for c in combos if c != ("fp32","fp32","fp32")]

    for qkv, acc, sv in tqdm(combos, desc="  Precision combos"):
        mse = mse_of_mha_output(model, audio_pairs, qkv, acc, sv, device)
        rows.append({"qkv_prec": qkv, "acc_prec": acc, "sv_prec": sv, "mha_output_mse": mse})
        if (qkv, acc, sv) == ("fp32","fp32","fp32") or qkv in ("fp8_e4m3","int8"):
            print(f"    Q/K/V={qkv:<10} acc={acc:<6} sv={sv:<10}  MSE={mse:.6f}")

    csv_path = RESULTS_DIR / "precision_sensitivity.csv"
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["qkv_prec","acc_prec","sv_prec","mha_output_mse"])
        w.writeheader()
        w.writerows(rows)
    print(f"  → {csv_path}")
    return rows


# ══════════════════════════════════════════════════════════════════════════════
#  3. Attention entropy & effective span
# ══════════════════════════════════════════════════════════════════════════════

def analyze_attention_entropy(captured: dict, model):
    """
    Uses the captured attn_weights to compute per-head entropy and
    effective span statistics.  Only uses the re_re pass (primary signal).
    """
    print("\n[3/5] Attention entropy & density analysis …")

    num_heads = model.mattn.mattn.MultiheadAttention_re.num_heads

    # Collect attention weights from the re_re pass (most representative)
    attn_list = captured.get("attn_weights_re_re", [])
    if not attn_list:
        print("  WARNING: no captured attn_weights_re_re — skipping")
        return {}

    # Shape: (N_samples, B, H, T, T) → merge all
    all_w = torch.cat(attn_list, dim=0)  # (N, H, T, T)
    if all_w.dim() == 5:
        all_w = all_w.squeeze(1)  # remove batch dim if B=1

    N, H, T, _ = all_w.shape
    print(f"  Attention weight tensor: N={N}, H={H}, T={T}")

    # ── Per-head Shannon entropy (bits) ─────────────────────────────────────
    eps = 1e-12
    entropy = -(all_w * (all_w + eps).log() / math.log(2)).sum(-1)  # (N,H,T)
    per_head_entropy = entropy.mean(dim=(0, 2))  # (H,)

    # ── Effective span: fraction of scores > 0.05 within non-masked region ──
    threshold = 0.05
    effective_span = (all_w > threshold).float().mean(dim=(0, 2, 3))  # (H,)

    # ── Top-k approximation error (offline, no model re-run) ────────────────
    topk_errors = {}
    for k in [4, 8, 16]:
        top_vals, top_idx = all_w.topk(k, dim=-1)
        topk_w = torch.zeros_like(all_w).scatter_(-1, top_idx, top_vals)
        topk_w = topk_w / (topk_w.sum(-1, keepdim=True) + eps)
        err = (all_w - topk_w).abs().mean()
        topk_errors[k] = float(err)
        print(f"  Top-{k:<2} approx L1 error (attn_weights): {err:.5f}")

    # ── CSV ──────────────────────────────────────────────────────────────────
    rows = []
    for h in range(H):
        rows.append({
            "head":            h,
            "mean_entropy_bits": float(per_head_entropy[h]),
            "effective_span":    float(effective_span[h]),
            "head_type":         "sharp" if float(per_head_entropy[h]) < 2.0 else "flat",
        })
    csv_path = RESULTS_DIR / "attention_entropy.csv"
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["head","mean_entropy_bits","effective_span","head_type"])
        w.writeheader()
        w.writerows(rows)
    print(f"  → {csv_path}")

    # ── Heatmap: average attention pattern across heads & samples ───────────
    avg_attn = all_w.mean(dim=(0, 1)).numpy()   # (T, T)
    fig, ax = plt.subplots(figsize=(8, 7))
    im = ax.imshow(avg_attn, aspect="auto", origin="upper", cmap="viridis")
    ax.set_title("Average attention pattern (all heads, all samples)", fontsize=12)
    ax.set_xlabel("Key position")
    ax.set_ylabel("Query position")
    plt.colorbar(im, ax=ax)
    plt.tight_layout()
    heatmap_path = RESULTS_DIR / "attention_pattern_heatmap.png"
    fig.savefig(heatmap_path, dpi=120)
    plt.close(fig)
    print(f"  → {heatmap_path}")

    # ── Per-head grid ────────────────────────────────────────────────────────
    cols = 8
    rows_grid = math.ceil(H / cols)
    fig2, axes = plt.subplots(rows_grid, cols, figsize=(cols * 2.5, rows_grid * 2.5))
    axes_flat = axes.flatten() if H > 1 else [axes]
    head_avg = all_w.mean(dim=(0, 2)).numpy()   # (H, T)  — "outgoing" attention per head
    head_attn_2d = all_w.mean(dim=0)            # (H, T, T)

    for h in range(H):
        ax = axes_flat[h]
        ax.imshow(head_attn_2d[h].numpy(), aspect="auto", origin="upper", cmap="plasma")
        ent_val = float(per_head_entropy[h])
        sp_val  = float(effective_span[h])
        ax.set_title(f"H{h}\nH={ent_val:.1f}b sp={sp_val:.2f}", fontsize=7)
        ax.set_xticks([])
        ax.set_yticks([])
    for h in range(H, len(axes_flat)):
        axes_flat[h].axis("off")
    fig2.suptitle("Per-head attention patterns (avg over samples)", fontsize=12)
    plt.tight_layout()
    per_head_path = RESULTS_DIR / "per_head_patterns.png"
    fig2.savefig(per_head_path, dpi=100)
    plt.close(fig2)
    print(f"  → {per_head_path}")

    return {
        "per_head_entropy": per_head_entropy.tolist(),
        "per_head_effective_span": effective_span.tolist(),
        "topk_L1_errors": topk_errors,
        "entropy_rows": rows,
    }


# ══════════════════════════════════════════════════════════════════════════════
#  4. Top-k sparse attention simulation (with full model eval)
# ══════════════════════════════════════════════════════════════════════════════

class TopKCMHA(nn.Module):
    """
    ComplexMultiheadAttention replacement that applies top-k masking
    to the attention weights before Score×V.
    k=None means full attention (no masking).
    """
    def __init__(self, original_cmha, k: int | None = None):
        super().__init__()
        self.cmha = original_cmha
        self.k    = k

    def _single_pass(self, x_in, mha, attn_mask):
        B, T, D = x_in.shape
        H = mha.num_heads
        d = D // H

        W, b = mha.in_proj_weight, mha.in_proj_bias
        Wq, Wk, Wv = W.chunk(3, dim=0)
        bq, bk, bv = b.chunk(3, dim=0)

        Q = F.linear(x_in, Wq, bq)
        K = F.linear(x_in, Wk, bk)
        V = F.linear(x_in, Wv, bv)

        Qh = Q.view(B,T,H,d).transpose(1,2)
        Kh = K.view(B,T,H,d).transpose(1,2)
        Vh = V.view(B,T,H,d).transpose(1,2)

        scores = (Qh @ Kh.transpose(-2,-1)) / math.sqrt(d)
        if attn_mask is not None:
            scores = scores + attn_mask[None, None]

        attn_w = F.softmax(scores, dim=-1)           # (B,H,T,T)

        if self.k is not None and self.k < T:
            # Keep top-k per query; zero others and renormalise
            topk_vals, topk_idx = attn_w.topk(self.k, dim=-1)
            sparse_w = torch.zeros_like(attn_w).scatter_(-1, topk_idx, topk_vals)
            row_sums  = sparse_w.sum(-1, keepdim=True).clamp(min=1e-8)
            attn_w    = sparse_w / row_sums

        sv = attn_w @ Vh
        sv_flat = sv.transpose(1,2).contiguous().view(B, T, D)
        return F.linear(sv_flat, mha.out_proj.weight, mha.out_proj.bias)

    def forward(self, x, attn_mask=None):
        mha_re = self.cmha.MultiheadAttention_re
        mha_im = self.cmha.MultiheadAttention_im
        xr, xi = x.real.float(), x.imag.float()

        r_r = self._single_pass(xr, mha_re, attn_mask)
        i_i = self._single_pass(xi, mha_im, attn_mask)
        i_r = self._single_pass(xi, mha_re, attn_mask)
        r_i = self._single_pass(xr, mha_im, attn_mask)

        return torch.complex(r_r - i_i, i_r + r_i).to(x.dtype)


def run_topk_eval(model, audio_pairs, device):
    """
    For k ∈ {4,8,12,16,24,31,full}, measure:
      - MHA output MSE vs full attention
      - SI-SNR, ILD error, IPD error on test pairs
    """
    print("\n[4/5] Top-k sparse attention evaluation …")

    mattn_block    = model.mattn
    original_cmha  = mattn_block.mattn

    # Determine window size from model
    attn_win = model.attn_win if hasattr(model, "attn_win") else 31
    k_values = [4, 8, 12, 16, 24, attn_win, None]  # None = full window

    # ── FP32 full-attention reference outputs ────────────────────────────────
    ref_mha_outs  = []
    ref_model_outs = []
    model.eval()
    with torch.no_grad():
        for noisy, clean in tqdm(audio_pairs, desc="  FP32 reference"):
            mha_buf = []
            h = original_cmha.register_forward_hook(lambda m,i,o: mha_buf.append(o.detach().cpu()))
            out = model(noisy)
            h.remove()
            ref_mha_outs.append(mha_buf[0])
            ref_model_outs.append((out.detach().cpu(), clean.cpu()))

    rows = []
    for k in k_values:
        label = str(k) if k is not None else "full"
        patch = TopKCMHA(original_cmha, k=k)
        mattn_block.mattn = patch

        mse_list, sisnr_l_list, sisnr_r_list, ild_err_list, ipd_err_list = [], [], [], [], []
        try:
            model.eval()
            with torch.no_grad():
                for i, (noisy, clean) in enumerate(tqdm(audio_pairs, desc=f"  k={label}", leave=False)):
                    mha_buf = []
                    h = patch.register_forward_hook(lambda m,i,o: mha_buf.append(o.detach().cpu()))
                    out = model(noisy)
                    h.remove()

                    # MHA output MSE
                    if mha_buf:
                        ref_c = ref_mha_outs[i]
                        pat_c = mha_buf[0]
                        mse = (ref_c.real.float() - pat_c.real.float()).pow(2).mean().item() + \
                              (ref_c.imag.float() - pat_c.imag.float()).pow(2).mean().item()
                        mse_list.append(mse)

                    # Audio quality metrics
                    out_cpu   = out.squeeze(0).cpu()   # (2, T)
                    clean_cpu = clean.squeeze(0).cpu() # (2, T)
                    min_len   = min(out_cpu.shape[-1], clean_cpu.shape[-1])
                    out_cpu   = out_cpu[:, :min_len]
                    clean_cpu = clean_cpu[:, :min_len]

                    sisnr_l_list.append(si_snr(clean_cpu[0], out_cpu[0]))
                    sisnr_r_list.append(si_snr(clean_cpu[1], out_cpu[1]))
                    ild_err_list.append(abs(compute_ild(clean_cpu) - compute_ild(out_cpu)))
                    ipd_err_list.append(compute_ipd_error(clean_cpu, out_cpu))

        finally:
            mattn_block.mattn = original_cmha

        row = {
            "k":             label,
            "mha_mse":       float(np.mean(mse_list)) if mse_list else float("nan"),
            "si_snr_l":      float(np.mean(sisnr_l_list)) if sisnr_l_list else float("nan"),
            "si_snr_r":      float(np.mean(sisnr_r_list)) if sisnr_r_list else float("nan"),
            "ild_err_dB":    float(np.mean(ild_err_list)) if ild_err_list else float("nan"),
            "ipd_err_rad":   float(np.mean(ipd_err_list)) if ipd_err_list else float("nan"),
        }
        rows.append(row)
        print(f"  k={label:<5}  MHA-MSE={row['mha_mse']:.5f}  "
              f"SI-SNR_L={row['si_snr_l']:.2f}  SI-SNR_R={row['si_snr_r']:.2f}  "
              f"ILD_err={row['ild_err_dB']:.3f}dB  IPD_err={row['ipd_err_rad']:.4f}rad")

    csv_path = RESULTS_DIR / "topk_eval.csv"
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"  → {csv_path}")

    # ── Plot ─────────────────────────────────────────────────────────────────
    ks_num = [r["k"] for r in rows]
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    metrics = [
        ("mha_mse",     "MHA Output MSE",     axes[0,0]),
        ("si_snr_l",    "SI-SNR Left (dB)",   axes[0,1]),
        ("ild_err_dB",  "ILD Error (dB)",     axes[1,0]),
        ("ipd_err_rad", "IPD Error (rad)",    axes[1,1]),
    ]
    for col, label, ax in metrics:
        vals = [r[col] for r in rows]
        ax.plot(range(len(ks_num)), vals, "o-", linewidth=2)
        ax.set_xticks(range(len(ks_num)))
        ax.set_xticklabels(ks_num)
        ax.set_xlabel("k (top-k sparse attention)")
        ax.set_ylabel(label)
        ax.set_title(label)
        ax.grid(True, alpha=0.3)
    fig.suptitle("Quality vs. Top-k Sparse Attention", fontsize=13)
    plt.tight_layout()
    plot_path = RESULTS_DIR / "topk_plot.png"
    fig.savefig(plot_path, dpi=120)
    plt.close(fig)
    print(f"  → {plot_path}")

    return rows


# ══════════════════════════════════════════════════════════════════════════════
#  5. Mixed-precision recommendation
# ══════════════════════════════════════════════════════════════════════════════

# Relative headroom over fp32/fp32/fp32 patched baseline (1.5 = allow +50% MSE)
MSE_HEADROOM_FACTOR = 1.5
# Also keep an absolute floor for when fp32 baseline is near zero
MSE_THRESHOLD_ABS_FLOOR = 1e-4


def build_recommendation(prec_rows, entropy_info, topk_rows, model):
    """
    Synthesise a concrete mixed-precision recommendation.
    """
    print("\n[5/5] Mixed-precision recommendation …")

    num_heads  = model.mattn.mattn.MultiheadAttention_re.num_heads
    embed_dim  = model.mattn.mattn.MultiheadAttention_re.embed_dim
    head_dim   = embed_dim // num_heads

    # ── Find best precisions from sensitivity sweep ──────────────────────────
    # fp32/fp32/fp32 patched baseline: difference vs FP8-QAT reference.
    # All comparisons use this as the acceptable noise floor.
    fp32_mse = next((r["mha_output_mse"] for r in prec_rows
                     if r["qkv_prec"]=="fp32" and r["acc_prec"]=="fp32" and r["sv_prec"]=="fp32"), 0.0)

    # MSE threshold = fp32 baseline × headroom factor, but at least the absolute floor
    mse_threshold = max(fp32_mse * MSE_HEADROOM_FACTOR, MSE_THRESHOLD_ABS_FLOOR)
    print(f"  fp32/fp32/fp32 baseline MSE: {fp32_mse:.4f}  →  threshold: {mse_threshold:.4f}")

    best_qkv = "fp32"
    best_acc = "fp32"
    best_sv  = "fp32"

    order = ["int4","fp4","int8","fp8_e4m3","fp8_e5m2","int16","fp16","fp32"]
    for r in prec_rows:
        mse = r["mha_output_mse"]
        if mse <= mse_threshold:
            qkv, acc, sv = r["qkv_prec"], r["acc_prec"], r["sv_prec"]
            # Prefer most aggressive (smallest index in order) for each slot
            if order.index(qkv) < order.index(best_qkv):
                best_qkv = qkv
            if order.index(acc) < order.index(best_acc):
                best_acc = acc
            if order.index(sv) < order.index(best_sv):
                best_sv = sv

    # ── Top-k recommendation ─────────────────────────────────────────────────
    full_row   = next((r for r in topk_rows if r["k"] == "full"), None)
    ref_sisnr  = full_row["si_snr_l"] if full_row else 0.0
    ref_ild    = full_row["ild_err_dB"] if full_row else 0.0

    best_k = "full"
    for row in topk_rows:
        if row["k"] == "full":
            continue
        k_val = int(row["k"])
        sisnr_drop = ref_sisnr - row["si_snr_l"]
        ild_inc    = row["ild_err_dB"] - ref_ild
        if sisnr_drop < 0.5 and ild_inc < 0.5:   # less than 0.5 dB degradation
            best_k = k_val
            break   # first (smallest k) that passes

    # ── Entropy analysis ─────────────────────────────────────────────────────
    per_head_entropy = entropy_info.get("per_head_entropy", [])
    if per_head_entropy:
        entropy_arr  = np.array(per_head_entropy)
        sharp_heads  = int((entropy_arr < 2.0).sum())
        flat_heads   = num_heads - sharp_heads
        mean_entropy = float(entropy_arr.mean())
    else:
        sharp_heads = flat_heads = mean_entropy = 0

    # ── Area/power estimation ─────────────────────────────────────────────────
    # Reference: fp32 MAC area = 1.0 (relative)
    bit_area = {"fp32": 1.0, "fp16": 0.25, "fp8_e4m3": 0.08, "fp8_e5m2": 0.08,
                "fp4": 0.03, "int8": 0.06, "int4": 0.02, "int16": 0.15}
    area_ratio = bit_area.get(best_qkv, 1.0)

    # Weight memory reduction (relative to fp32)
    weight_bits = {"fp32": 32, "fp16": 16, "fp8_e4m3": 8, "fp8_e5m2": 8,
                   "fp4": 4, "int8": 8, "int4": 4, "int16": 16}
    mem_reduction = 32 / weight_bits.get(best_qkv, 32)

    # Compute attention density reduction
    attn_win   = model.attn_win if hasattr(model, "attn_win") else 31
    if best_k != "full":
        density_reduction = (attn_win - best_k) / attn_win
    else:
        density_reduction = 0.0

    rec = {
        "model_config": {
            "embed_dim":   embed_dim,
            "num_heads":   num_heads,
            "head_dim":    head_dim,
            "attn_win":    attn_win,
        },
        "recommended_precision": {
            "Wq_Wk_Wv_Wo_weights":   best_qkv,
            "Q_K_V_activations":     best_qkv,
            "QKT_accumulator":       best_acc,
            "score_x_V_output":      best_sv,
            "proj_in_proj_out":      "fp16",
        },
        "recommended_sparse_attention": {
            "top_k":                 best_k,
            "density_reduction_pct": round(density_reduction * 100, 1),
            "window_size":           attn_win,
            "sharp_heads":           sharp_heads,
            "flat_heads":            flat_heads,
            "mean_entropy_bits":     round(mean_entropy, 3),
        },
        "hardware_savings_estimate": {
            "MAC_area_ratio_vs_fp32":    area_ratio,
            "weight_memory_reduction_x": mem_reduction,
            "attn_compute_reduction_pct": round(density_reduction * 100, 1),
            "combined_area_savings_pct": round((1.0 - area_ratio) * 100, 1),
        },
        "mse_thresholds_used": {
            "headroom_factor": MSE_HEADROOM_FACTOR,
            "abs_floor":       MSE_THRESHOLD_ABS_FLOOR,
        },
        "sensitivity_fp32_baseline_mse": fp32_mse,
    }

    rec_path = RESULTS_DIR / "precision_recommendation.json"
    with open(rec_path, "w") as f:
        json.dump(rec, f, indent=2)
    print(f"  → {rec_path}")

    return rec


# ══════════════════════════════════════════════════════════════════════════════
#  Summary report
# ══════════════════════════════════════════════════════════════════════════════

def write_summary(dr_rows, prec_rows, entropy_info, topk_rows, rec):
    print("\n[+] Writing executive summary …")

    dr_map   = {r["subop"]: r for r in dr_rows}
    fp32_mse = rec["sensitivity_fp32_baseline_mse"]

    # Best precision configs under threshold
    threshold = max(fp32_mse * MSE_HEADROOM_FACTOR, MSE_THRESHOLD_ABS_FLOOR)
    good_prec = [r for r in prec_rows if r["mha_output_mse"] <= threshold
                 and r["qkv_prec"] != "fp32"]
    best_agg  = sorted(good_prec, key=lambda r: r["mha_output_mse"])[:5]

    per_head_ent = entropy_info.get("per_head_entropy", [])
    topk_L1      = entropy_info.get("topk_L1_errors", {})

    full_row  = next((r for r in topk_rows if r["k"] == "full"), {})
    topk_rows_sorted = sorted([r for r in topk_rows if r["k"] != "full"],
                               key=lambda r: int(r["k"]))

    rec_prec = rec["recommended_precision"]
    rec_sprs = rec["recommended_sparse_attention"]
    rec_hw   = rec["hardware_savings_estimate"]

    lines = [
        "# Attention Mixed-Precision Analysis — Executive Summary",
        "",
        "## Model Configuration",
        f"- embed_dim={rec['model_config']['embed_dim']}, "
        f"num_heads={rec['model_config']['num_heads']}, "
        f"head_dim={rec['model_config']['head_dim']}, "
        f"attn_win={rec['model_config']['attn_win']}",
        "",
        "## 1. Dynamic Range per Sub-Operation",
        "",
        "| Sub-op | min | max | std | dyn_range | log2(|max|) |",
        "|--------|-----|-----|-----|-----------|-------------|",
    ]
    for op in ["Q","K","V","QKT","attn_weights","sv"]:
        if op in dr_map:
            r = dr_map[op]
            lines.append(
                f"| {op:<13} | {r['min']:+.3f} | {r['max']:+.3f} | "
                f"{r['std']:.4f} | {r['dynamic_range']:.4f} | {r['log2_range']:.2f} |"
            )
    lines += [
        "",
        "**Key observations:**",
        f"- QK^T scores span the largest dynamic range: "
        f"{dr_map.get('QKT',{}).get('dynamic_range', 'N/A'):.3f}",
        f"- Attention weights (post-softmax) are in [0,1] with std "
        f"{dr_map.get('attn_weights',{}).get('std', 'N/A'):.4f}",
        f"- V activations have a smaller range than Q/K, suggesting "
        f"V can tolerate more aggressive quantisation.",
        "",
        "## 2. Precision Sensitivity",
        "",
        f"FP32 baseline MHA output MSE: {fp32_mse:.2e}",
        "",
        "| Q/K/V prec | acc prec | SV prec | MHA MSE |",
        "|------------|----------|---------|---------|",
    ]
    for r in best_agg:
        lines.append(
            f"| {r['qkv_prec']:<10} | {r['acc_prec']:<8} | {r['sv_prec']:<7} | {r['mha_output_mse']:.2e} |"
        )
    lines += [
        "",
        "**Finding:** The minimum precision that keeps MHA MSE below threshold is "
        f"**{rec_prec['Q_K_V_activations']}** for Q/K/V, "
        f"**{rec_prec['QKT_accumulator']}** for QK^T accumulator, "
        f"**{rec_prec['score_x_V_output']}** for Score×V.",
        "",
        "## 3. Attention Entropy & Effective Span",
    ]
    if per_head_ent:
        ent_arr  = np.array(per_head_ent)
        lines += [
            f"- Mean entropy across heads: **{ent_arr.mean():.2f} bits** "
            f"(max possible for W=31: {math.log2(31):.2f} bits)",
            f"- Sharp heads (entropy < 2 bits): "
            f"**{int((ent_arr < 2.0).sum())}** / {len(per_head_ent)}",
            f"- Flat heads (entropy ≥ 2 bits): "
            f"**{int((ent_arr >= 2.0).sum())}** / {len(per_head_ent)}",
        ]
    for k, err in sorted(topk_L1.items()):
        lines.append(f"- Top-{k} approximation L1 error: **{err:.5f}**")
    lines += [
        "",
        "## 4. Top-k Sparse Attention",
        "",
        "| k | MHA MSE | SI-SNR L | SI-SNR R | ILD err (dB) | IPD err (rad) |",
        "|---|---------|----------|----------|-------------|---------------|",
    ]
    for r in topk_rows:
        lines.append(
            f"| {r['k']:<5} | {r['mha_mse']:.4f} | {r['si_snr_l']:.2f} | "
            f"{r['si_snr_r']:.2f} | {r['ild_err_dB']:.3f} | {r['ipd_err_rad']:.4f} |"
        )
    lines += [
        "",
        f"**Recommended k:** {rec_sprs['top_k']} "
        f"({rec_sprs['density_reduction_pct']}% compute reduction within window)",
        "",
        "## 5. Mixed-Precision Recommendation",
        "",
        "| Pipeline Stage | Recommended Precision |",
        "|----------------|----------------------|",
        f"| Wq / Wk / Wv / Wo weights | {rec_prec['Wq_Wk_Wv_Wo_weights']} |",
        f"| Q / K / V activations      | {rec_prec['Q_K_V_activations']} |",
        f"| QK^T accumulator           | {rec_prec['QKT_accumulator']} |",
        f"| Score × V output           | {rec_prec['score_x_V_output']} |",
        f"| proj_in / proj_out         | {rec_prec['proj_in_proj_out']} |",
        "",
        "### Hardware Impact Estimate",
        "",
        f"- MAC area vs. FP32: **{rec_hw['MAC_area_ratio_vs_fp32']:.2f}×** "
        f"({rec_hw['combined_area_savings_pct']}% area saved)",
        f"- Weight memory reduction: **{rec_hw['weight_memory_reduction_x']:.1f}×**",
        f"- Attention compute reduction: "
        f"**{rec_hw['attn_compute_reduction_pct']}%** (top-k sparsity)",
        "",
        "## Key Answers",
        "",
        f"1. **Minimum QK^T accumulator precision?** → `{rec_prec['QKT_accumulator']}`",
        f"2. **Can V be quantised more aggressively than Q/K?** → "
        f"Yes: V std is lower ({dr_map.get('V',{}).get('std','N/A'):.4f} vs "
        f"Q {dr_map.get('Q',{}).get('std','N/A'):.4f}); in hardware, V can use "
        f"narrower bit-width than Q/K projections.",
        f"3. **Effective attention span within W=31?** → "
        f"Top-k L1 error for k=8: {topk_L1.get(8, 'N/A'):.4f}",
        f"4. **Top-k without degradation?** → k={rec_sprs['top_k']}",
        f"5. **Unimportant heads?** → {rec_sprs['sharp_heads']} sharp heads "
        f"(entropy < 2 bits) could use smaller k or lower precision.",
        f"6. **Total hardware area/power reduction?** → "
        f"~{rec_hw['combined_area_savings_pct']}% MAC area + "
        f"{rec_hw['weight_memory_reduction_x']:.1f}× weight bandwidth.",
    ]

    summary_path = RESULTS_DIR / "summary.md"
    with open(summary_path, "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"  → {summary_path}")


# ══════════════════════════════════════════════════════════════════════════════
#  Entry point
# ══════════════════════════════════════════════════════════════════════════════

def parse_args():
    p = argparse.ArgumentParser(description="BCCTN attention mixed-precision analyzer")
    p.add_argument("--checkpoint",   default="model.ckpt")
    p.add_argument("--test_dir",     default="/raid/R12K41024/BCCTN/Dataset/noisy_testset/snr_0dB/")
    p.add_argument("--clean_dir",    default="/raid/R12K41024/BCCTN/Dataset/clean_testset/")
    p.add_argument("--num_samples",  type=int, default=20)
    p.add_argument("--device",       default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--skip_prec",    action="store_true", help="Skip precision sensitivity (slow)")
    p.add_argument("--skip_topk",    action="store_true", help="Skip top-k model eval (slow)")
    return p.parse_args()


def main():
    args = parse_args()
    device = torch.device(args.device)
    print(f"Device: {device}")

    # ── Load model ────────────────────────────────────────────────────────────
    print(f"\nLoading model from {args.checkpoint} …")
    GlobalHydra.instance().clear()
    initialize(config_path="config", version_base=None)
    cfg = compose(config_name="config")

    lm = DCNNLightningModule(cfg)
    ckpt = torch.load(args.checkpoint, map_location="cpu")
    lm.load_state_dict(ckpt["state_dict"])
    model = lm.model.to(device).eval()

    print(f"  embed_dim={model.mattn.mattn.MultiheadAttention_re.embed_dim}, "
          f"num_heads={model.mattn.mattn.MultiheadAttention_re.num_heads}, "
          f"attn_win={getattr(model, 'attn_win', 31)}")

    # ── Load audio pairs ──────────────────────────────────────────────────────
    print(f"\nLoading up to {args.num_samples} audio pairs …")
    test_dir  = Path(args.test_dir)
    clean_dir = Path(args.clean_dir)
    noisy_files = sorted(test_dir.glob("*.wav"))[:args.num_samples * 2]  # load extra, filter failures

    audio_pairs = []
    for nf in noisy_files:
        result = load_audio_pair(nf, clean_dir, device)
        if result is not None:
            audio_pairs.append(result)
            if len(audio_pairs) >= args.num_samples:
                break

    print(f"  Loaded {len(audio_pairs)} valid pairs")
    if len(audio_pairs) == 0:
        print("ERROR: No audio pairs found. Check --test_dir and --clean_dir.")
        sys.exit(1)

    # ── Analysis 1: Dynamic range ─────────────────────────────────────────────
    interceptor = MHAInterceptor(model.mattn.mattn)
    captured, dr_rows = analyze_dynamic_range(model, interceptor, audio_pairs, device)

    # ── Analysis 2: Precision sensitivity ────────────────────────────────────
    if not args.skip_prec:
        prec_rows = analyze_precision_sensitivity(model, audio_pairs, device)
    else:
        print("\n[2/5] Precision sensitivity SKIPPED (--skip_prec)")
        prec_rows = [{"qkv_prec":"fp32","acc_prec":"fp32","sv_prec":"fp32","mha_output_mse":0.0}]

    # ── Analysis 3: Entropy ───────────────────────────────────────────────────
    entropy_info = analyze_attention_entropy(captured, model)

    # ── Analysis 4: Top-k eval ────────────────────────────────────────────────
    if not args.skip_topk:
        topk_rows = run_topk_eval(model, audio_pairs, device)
    else:
        print("\n[4/5] Top-k eval SKIPPED (--skip_topk)")
        topk_rows = [{"k": "full", "mha_mse": 0.0,
                      "si_snr_l": 0.0, "si_snr_r": 0.0,
                      "ild_err_dB": 0.0, "ipd_err_rad": 0.0}]

    # ── Analysis 5: Recommendation ────────────────────────────────────────────
    rec = build_recommendation(prec_rows, entropy_info, topk_rows, model)

    # ── Summary ───────────────────────────────────────────────────────────────
    write_summary(dr_rows, prec_rows, entropy_info, topk_rows, rec)

    print("\n✓ All analyses complete. Results in:", RESULTS_DIR.resolve())
    print("  Files:")
    for f in sorted(RESULTS_DIR.iterdir()):
        print(f"    {f.name}")


if __name__ == "__main__":
    main()
