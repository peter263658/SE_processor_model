#!/usr/bin/env python3
"""
Layer-wise bit-width sensitivity analysis for BCCTN (PTQ).

For each layer group, quantize ONLY that group to a target bit-width while
keeping all other layers at their trained (FP8 QAT) precision.  Runs inference
on a small eval set and records ΔSI-SNR, ΔILD, ΔIPD relative to the FP8
baseline.

Usage:
    python bitwidth_sweep.py --checkpoint model.ckpt --num_samples 50

Outputs (under results/):
    sensitivity_matrix.csv
    sensitivity_heatmap.png
    mixed_precision_config.json
"""

import argparse
import json
import random
import re
import sys
from contextlib import contextmanager
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import soundfile as sf
import torch
from tqdm import tqdm

# ──────────────────────────────────────────────────────────────
# 1. Quantisation functions
# ──────────────────────────────────────────────────────────────

def _quant_int(t: torch.Tensor, bits: int) -> torch.Tensor:
    """Symmetric INT quantisation (per-tensor min-max)."""
    if t.is_complex():
        return torch.complex(
            _quant_int(t.real.contiguous(), bits),
            _quant_int(t.imag.contiguous(), bits),
        )
    qmax = 2 ** (bits - 1) - 1
    amax = t.abs().max().clamp(min=1e-8)
    scale = amax / qmax
    return torch.round(t / scale).clamp(-qmax, qmax) * scale


# FP4 E2M1 positive representable magnitudes (NVFP4 convention)
_FP4_VALS = torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0])


def _quant_fp4(t: torch.Tensor) -> torch.Tensor:
    """FP4 E2M1 simulation (scalar-per-tensor scale)."""
    if t.is_complex():
        return torch.complex(
            _quant_fp4(t.real.contiguous()),
            _quant_fp4(t.imag.contiguous()),
        )
    fp4 = _FP4_VALS.to(t.device)
    shape = t.shape
    flat = t.flatten()
    amax = flat.abs().max().clamp(min=1e-8)
    scale = amax / fp4.max()
    norm = flat / scale
    sign = norm.sign()
    mag = norm.abs()
    # [N, 8] → nearest FP4 magnitude
    idx = (mag.unsqueeze(-1) - fp4.unsqueeze(0)).abs().argmin(dim=-1)
    return (fp4[idx] * sign * scale).reshape(shape)


QUANT_FNS = {
    'INT8': lambda t: _quant_int(t, 8),
    'INT4': lambda t: _quant_int(t, 4),
    'INT3': lambda t: _quant_int(t, 3),
    'FP4':  _quant_fp4,
}

# ──────────────────────────────────────────────────────────────
# 2. Layer groups
# ──────────────────────────────────────────────────────────────

def build_layer_groups() -> dict[str, list[str]]:
    """
    Return ordered dict: logical_layer_name → [param_path, ...].
    Paths are relative to the BinauralAttentionDCNN module (no 'model.' prefix).
    Real and imaginary parts of the same logical layer are grouped together.
    """
    g: dict[str, list[str]] = {}

    for i in range(6):
        g[f'enc_dw_{i}'] = [
            f'encoder.model.{i}.0.depthwise_r.parametrizations.weight.original',
            f'encoder.model.{i}.0.depthwise_i.parametrizations.weight.original',
        ]
        g[f'enc_pw_{i}'] = [
            f'encoder.model.{i}.0.pointwise_r.parametrizations.weight.original',
            f'encoder.model.{i}.0.pointwise_i.parametrizations.weight.original',
        ]

    for i in range(6):
        g[f'dec_dw_{i}'] = [
            f'decoder.model.{i}.0.depthwise_r.parametrizations.weight.original',
            f'decoder.model.{i}.0.depthwise_i.parametrizations.weight.original',
        ]
        g[f'dec_pw_{i}'] = [
            f'decoder.model.{i}.0.pointwise_r.parametrizations.weight.original',
            f'decoder.model.{i}.0.pointwise_i.parametrizations.weight.original',
        ]

    # MHA block
    g['mattn_input_proj'] = [
        'mattn.input_proj.parametrizations.weight.original',
    ]
    g['mattn_in_proj_re'] = [
        'mattn.mattn.MultiheadAttention_re.parametrizations.in_proj_weight.original',
    ]
    g['mattn_in_proj_im'] = [
        'mattn.mattn.MultiheadAttention_im.parametrizations.in_proj_weight.original',
    ]
    g['mattn_out_proj_re'] = [
        'mattn.mattn.MultiheadAttention_re.out_proj.parametrizations.weight.original',
    ]
    g['mattn_out_proj_im'] = [
        'mattn.mattn.MultiheadAttention_im.out_proj.parametrizations.weight.original',
    ]
    g['mattn_output_proj'] = [
        'mattn.output_proj.parametrizations.weight.original',
    ]

    return g


# ──────────────────────────────────────────────────────────────
# 3. Model loading
# ──────────────────────────────────────────────────────────────

def load_model(checkpoint_path: str, device: torch.device):
    """Load BinauralAttentionDCNN from a Lightning checkpoint via Hydra config."""
    from hydra import compose, initialize
    from hydra.core.global_hydra import GlobalHydra
    from DCNN.trainer import DCNNLightningModule

    GlobalHydra.instance().clear()
    with initialize(config_path='config', version_base=None):
        cfg = compose(config_name='config')

    lm = DCNNLightningModule(cfg)
    ckpt = torch.load(checkpoint_path, map_location=device)
    lm.load_state_dict(ckpt['state_dict'])
    model = lm.model.to(device)
    model.eval()
    return model


# ──────────────────────────────────────────────────────────────
# 4. Weight-patch context manager
# ──────────────────────────────────────────────────────────────

@contextmanager
def patch_params(model: torch.nn.Module, paths: list[str], quant_fn):
    """Temporarily quantise listed parameters in-place, restore on exit."""
    named = dict(model.named_parameters())
    saved: dict[str, torch.Tensor] = {}

    for p in paths:
        if p in named:
            saved[p] = named[p].data.clone()
            named[p].data.copy_(quant_fn(named[p].data))
        else:
            print(f'  [warn] param not found: {p}', flush=True)
    try:
        yield
    finally:
        for p, orig in saved.items():
            named[p].data.copy_(orig)


# ──────────────────────────────────────────────────────────────
# 5. Metrics
# ──────────────────────────────────────────────────────────────

def si_snr(clean: np.ndarray, enhanced: np.ndarray) -> float:
    """Scale-invariant SNR in dB (1-D arrays)."""
    c = clean - clean.mean()
    e = enhanced - enhanced.mean()
    alpha = np.dot(e, c) / (np.dot(c, c) + 1e-8)
    s_t = alpha * c
    e_n = e - s_t
    val = 10.0 * np.log10((np.dot(s_t, s_t) + 1e-8) / (np.dot(e_n, e_n) + 1e-8))
    return float(np.clip(val, -30.0, 40.0))


def _stft_pair(l, r, n_fft, hop, win):
    import librosa
    return (librosa.stft(l, n_fft=n_fft, hop_length=hop, win_length=win),
            librosa.stft(r, n_fft=n_fft, hop_length=hop, win_length=win))


def ild_error(cl, cr, el, er, n_fft=128, hop=80, win=128) -> float:
    """Mean |ILD_clean − ILD_enh| (dB) over speech-active TF bins."""
    eps = 1e-8
    CL, CR = _stft_pair(cl, cr, n_fft, hop, win)
    EL, ER = _stft_pair(el, er, n_fft, hop, win)

    ild_c = 20 * np.log10(np.abs(CL) + eps) - 20 * np.log10(np.abs(CR) + eps)
    ild_e = 20 * np.log10(np.abs(EL) + eps) - 20 * np.log10(np.abs(ER) + eps)

    energy = np.maximum(np.abs(CL) ** 2, np.abs(CR) ** 2)
    mx = energy.max(axis=1, keepdims=True).clip(min=eps)
    mask = 10 * np.log10(energy / mx) > -20

    err = np.abs(ild_c - ild_e)
    return float(err[mask].mean()) if mask.any() else 0.0


def ipd_error(cl, cr, el, er, n_fft=128, hop=80, win=128,
              f_max=1500, sr=16000) -> float:
    """Mean IPD error (degrees, f ≤ f_max) over speech-active TF bins."""
    eps = 1e-8
    CL, CR = _stft_pair(cl, cr, n_fft, hop, win)
    EL, ER = _stft_pair(el, er, n_fft, hop, win)

    max_bin = int(f_max / (sr / n_fft)) + 1
    CL, CR, EL, ER = CL[:max_bin], CR[:max_bin], EL[:max_bin], ER[:max_bin]

    energy = np.maximum(np.abs(CL) ** 2, np.abs(CR) ** 2)
    mx = energy.max(axis=1, keepdims=True).clip(min=eps)
    mask = 10 * np.log10(energy / mx) > -20

    ipd_c = np.angle(CL * np.conj(CR))
    ipd_e = np.angle(EL * np.conj(ER))
    err_deg = np.abs(np.angle(np.exp(1j * (ipd_c - ipd_e)))) * 180.0 / np.pi

    return float(err_deg[mask].mean()) if mask.any() else 0.0


# ──────────────────────────────────────────────────────────────
# 6. Dataset helpers
# ──────────────────────────────────────────────────────────────

def collect_sample_pairs(noisy_dir: Path, clean_dir: Path,
                         n: int = 50, seed: int = 42) -> list[tuple[Path, Path]]:
    """
    Randomly sample `n` (noisy, clean) path pairs from noisy_dir.
    Noisy: {base}_az{az}_snr{snr}.wav  →  Clean: {base}_az{az}.wav
    """
    noisy_files = sorted(noisy_dir.glob('*.wav'))
    if not noisy_files:
        # Try nested structure
        noisy_files = sorted(noisy_dir.glob('**/*.wav'))
    if not noisy_files:
        raise FileNotFoundError(f'No .wav files found in {noisy_dir}')

    pattern = re.compile(r'^(.+_az-?\d+)_snr.*$')
    pairs: list[tuple[Path, Path]] = []
    for nf in noisy_files:
        m = pattern.match(nf.stem)
        if m:
            clean_name = m.group(1) + '.wav'
            cf = clean_dir / clean_name
            if cf.exists():
                pairs.append((nf, cf))

    if not pairs:
        raise RuntimeError('Could not match any noisy→clean pairs. '
                           'Check file naming convention.')

    rng = random.Random(seed)
    rng.shuffle(pairs)
    return pairs[:n]


def load_audio_pair(noisy_path: Path, clean_path: Path):
    """
    Load and return (noisy, clean) as numpy arrays [2, T] at 16 kHz.
    Returns None if the file is unusable.
    """
    import librosa
    noisy, sr_n = sf.read(noisy_path)
    clean, sr_c = sf.read(clean_path)

    if noisy.ndim == 1 or clean.ndim == 1:
        return None  # mono – skip
    if min(len(noisy), len(clean)) / sr_c < 0.3:
        return None  # too short

    # Resample if needed
    if sr_n != 16000:
        noisy = np.stack([
            librosa.resample(noisy[:, ch], orig_sr=sr_n, target_sr=16000)
            for ch in range(2)
        ], axis=1)
    if sr_c != 16000:
        clean = np.stack([
            librosa.resample(clean[:, ch], orig_sr=sr_c, target_sr=16000)
            for ch in range(2)
        ], axis=1)

    return noisy.T.astype(np.float32), clean.T.astype(np.float32)


# ──────────────────────────────────────────────────────────────
# 7. Single-experiment inference
# ──────────────────────────────────────────────────────────────

def run_inference(model, samples, device) -> list[dict]:
    """
    Run model on a list of (noisy[2,T], clean[2,T]) np arrays.
    Returns list of metric dicts with keys: si_snr_l, si_snr_r, ild, ipd.
    """
    results = []
    with torch.no_grad():
        for noisy_np, clean_np in samples:
            inp = torch.from_numpy(noisy_np).unsqueeze(0).to(device)  # [1,2,T]
            enh = model(inp).cpu().numpy()[0]  # [2, T']

            # Align lengths
            T = min(clean_np.shape[1], enh.shape[1])
            c = clean_np[:, :T]
            e = enh[:, :T]

            results.append({
                'si_snr_l': si_snr(c[0], e[0]),
                'si_snr_r': si_snr(c[1], e[1]),
                'ild':      ild_error(c[0], c[1], e[0], e[1]),
                'ipd':      ipd_error(c[0], c[1], e[0], e[1]),
            })
    return results


def mean_metrics(results: list[dict]) -> dict:
    if not results:
        return {}
    keys = results[0].keys()
    return {k: float(np.mean([r[k] for r in results])) for k in keys}


# ──────────────────────────────────────────────────────────────
# 8. Main sweep
# ──────────────────────────────────────────────────────────────

def run_sweep(model, layer_groups, samples, device, quant_configs):
    """
    Returns a dict:  {layer_name: {quant_config: {metric: value}}}
    """
    # ── Baseline (FP8, no extra quantisation) ──────────────────
    print('\n[Baseline] Running FP8 baseline …')
    baseline_results = run_inference(model, samples, device)
    baseline = mean_metrics(baseline_results)
    print(f"  SI-SNR avg: {(baseline['si_snr_l']+baseline['si_snr_r'])/2:.3f} dB  "
          f"ILD: {baseline['ild']:.3f} dB  IPD: {baseline['ipd']:.2f}°")

    sweep: dict = {}

    total = len(layer_groups) * len(quant_configs)
    pbar = tqdm(total=total, desc='Sweep')

    for layer_name, param_paths in layer_groups.items():
        sweep[layer_name] = {}
        for qname, qfn in quant_configs.items():
            pbar.set_description(f'{layer_name} / {qname}')
            with patch_params(model, param_paths, qfn):
                res = run_inference(model, samples, device)
            m = mean_metrics(res)

            sweep[layer_name][qname] = {
                'si_snr_avg': (m['si_snr_l'] + m['si_snr_r']) / 2,
                'delta_si_snr': (m['si_snr_l'] + m['si_snr_r']) / 2
                               - (baseline['si_snr_l'] + baseline['si_snr_r']) / 2,
                'ild':         m['ild'],
                'delta_ild':   m['ild'] - baseline['ild'],
                'ipd':         m['ipd'],
                'delta_ipd':   m['ipd'] - baseline['ipd'],
            }
            pbar.update(1)

    pbar.close()
    return baseline, sweep


# ──────────────────────────────────────────────────────────────
# 9. Post-processing: CSV, heatmap, recommendation
# ──────────────────────────────────────────────────────────────

def save_sensitivity_csv(sweep: dict, out_path: Path):
    rows = []
    for layer, configs in sweep.items():
        for qname, metrics in configs.items():
            rows.append({
                'layer': layer,
                'quant_config': qname,
                **metrics,
            })
    df = pd.DataFrame(rows)
    df.to_csv(out_path, index=False)
    print(f'Saved: {out_path}')
    return df


def save_heatmap(df: pd.DataFrame, metric: str, title: str, out_path: Path):
    """Pivot and plot a seaborn heatmap for one metric."""
    pivot = df.pivot(index='layer', columns='quant_config', values=metric)

    # Reorder columns logically
    col_order = [c for c in ['INT8', 'INT4', 'INT3', 'FP4'] if c in pivot.columns]
    pivot = pivot[col_order]

    # Layer ordering: enc_dw, enc_pw, dec_dw, dec_pw, mattn_*
    row_order = sorted(
        pivot.index,
        key=lambda x: (
            0 if x.startswith('enc_dw') else
            1 if x.startswith('enc_pw') else
            2 if x.startswith('dec_dw') else
            3 if x.startswith('dec_pw') else
            4,
            x
        )
    )
    pivot = pivot.loc[row_order]

    fig, ax = plt.subplots(figsize=(len(col_order) * 2 + 2, len(row_order) * 0.42 + 2))
    sns.heatmap(
        pivot.astype(float),
        ax=ax,
        cmap='RdYlGn_r',   # red = large degradation, green = small
        annot=True,
        fmt='.2f',
        linewidths=0.4,
        cbar_kws={'label': metric},
    )
    ax.set_title(title, fontsize=12)
    ax.set_xlabel('Quantisation config')
    ax.set_ylabel('Layer')
    plt.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'Saved: {out_path}')


def compute_param_bits(model, layer_groups) -> dict[str, int]:
    """Return total number of weight parameter elements per layer group."""
    named = dict(model.named_parameters())
    counts = {}
    for layer_name, paths in layer_groups.items():
        n = sum(named[p].numel() for p in paths if p in named)
        counts[layer_name] = n
    return counts


def make_mixed_precision_config(
        sweep: dict,
        param_bits: dict[str, int],
        thresholds: dict,
) -> dict:
    """
    Assign each layer to a precision tier:
      Tier A – W8  (INT4 degrades quality beyond threshold)
      Tier B – W4  (tolerant to INT4, but INT3 degrades)
      Tier C – W3  (very robust)
    """
    thr_snr  = thresholds['delta_si_snr']   # dB (negative = degradation)
    thr_ild  = thresholds['delta_ild']       # dB
    thr_ipd  = thresholds['delta_ipd']       # degrees

    def is_safe(layer, qname):
        m = sweep[layer].get(qname, {})
        return (m.get('delta_si_snr', -999) >= -thr_snr and
                m.get('delta_ild',    999)  <=  thr_ild and
                m.get('delta_ipd',    999)  <=  thr_ipd)

    config = {}
    for layer in sweep:
        if is_safe(layer, 'INT3'):
            tier, bits = 'C', 3
        elif is_safe(layer, 'INT4'):
            tier, bits = 'B', 4
        else:
            tier, bits = 'A', 8
        config[layer] = {
            'tier': tier,
            'bits': bits,
            'format': 'INT',
            'num_params': param_bits.get(layer, 0),
        }

    # Memory savings summary
    fp8_bytes = sum(v['num_params'] for v in config.values())      # 1 byte / param at FP8
    mp_bytes  = sum(v['num_params'] * v['bits'] // 8 for v in config.values())

    config['_summary'] = {
        'fp8_baseline_bytes': fp8_bytes,
        'mixed_precision_bytes': mp_bytes,
        'savings_bytes': fp8_bytes - mp_bytes,
        'savings_pct': round(100 * (1 - mp_bytes / fp8_bytes), 1) if fp8_bytes else 0,
        'tier_counts': {
            'A_w8': sum(1 for v in config.values() if isinstance(v, dict) and v.get('tier') == 'A'),
            'B_w4': sum(1 for v in config.values() if isinstance(v, dict) and v.get('tier') == 'B'),
            'C_w3': sum(1 for v in config.values() if isinstance(v, dict) and v.get('tier') == 'C'),
        },
    }
    return config


def print_summary(baseline, sweep, config):
    print('\n' + '=' * 60)
    print('SENSITIVITY SWEEP SUMMARY')
    print('=' * 60)
    print(f"FP8 baseline  SI-SNR: {(baseline['si_snr_l']+baseline['si_snr_r'])/2:.3f} dB"
          f"  ILD: {baseline['ild']:.3f} dB  IPD: {baseline['ipd']:.2f}°\n")

    print(f"{'Layer':<25} {'Tier':>4} {'Bits':>4}  "
          f"{'ΔSNR(INT4)':>10}  {'ΔILD(INT4)':>10}  {'ΔIPD(INT4)':>10}")
    print('-' * 70)
    for layer, cfg in config.items():
        if layer.startswith('_'):
            continue
        d = sweep[layer].get('INT4', {})
        print(f"{layer:<25} {cfg['tier']:>4} {cfg['bits']:>4}  "
              f"{d.get('delta_si_snr', float('nan')):>10.3f}  "
              f"{d.get('delta_ild',    float('nan')):>10.3f}  "
              f"{d.get('delta_ipd',    float('nan')):>10.2f}")

    s = config.get('_summary', {})
    print('\nMemory savings vs uniform FP8:')
    print(f"  FP8 baseline : {s.get('fp8_baseline_bytes', 0) / 1024:.1f} KB")
    print(f"  Mixed-prec   : {s.get('mixed_precision_bytes', 0) / 1024:.1f} KB")
    print(f"  Savings      : {s.get('savings_pct', 0):.1f}%")
    print(f"  Tier counts  : {s.get('tier_counts', {})}")


# ──────────────────────────────────────────────────────────────
# 10. Entry point
# ──────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description='BCCTN layer-wise bit-width sweep')
    p.add_argument('--checkpoint', default='model.ckpt',
                   help='Path to Lightning checkpoint')
    p.add_argument('--noisy_dir',
                   default='/raid/R12K41024/BCCTN/Dataset/noisy_testset/snr_0dB',
                   help='Directory with noisy test files')
    p.add_argument('--clean_dir',
                   default='/raid/R12K41024/BCCTN/Dataset/clean_testset',
                   help='Directory with clean reference files')
    p.add_argument('--num_samples', type=int, default=50,
                   help='Number of audio samples for each experiment')
    p.add_argument('--out_dir', default='results',
                   help='Output directory')
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--thr_snr', type=float, default=0.5,
                   help='Max allowed |ΔSI-SNR| for safe quantisation (dB)')
    p.add_argument('--thr_ild', type=float, default=0.5,
                   help='Max allowed ΔILD for safe quantisation (dB)')
    p.add_argument('--thr_ipd', type=float, default=5.0,
                   help='Max allowed ΔIPD for safe quantisation (degrees)')
    return p.parse_args()


def main():
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Device: {device}')

    # ── Load model ────────────────────────────────────────────
    print(f'Loading model from {args.checkpoint} …')
    model = load_model(args.checkpoint, device)

    # ── Load eval samples ─────────────────────────────────────
    noisy_dir = Path(args.noisy_dir)
    clean_dir  = Path(args.clean_dir)
    print(f'Collecting {args.num_samples} sample pairs from {noisy_dir} …')
    pairs = collect_sample_pairs(noisy_dir, clean_dir, args.num_samples, args.seed)
    print(f'  Matched {len(pairs)} pairs; loading audio …')

    samples = []
    for nf, cf in tqdm(pairs, desc='Loading audio'):
        audio = load_audio_pair(nf, cf)
        if audio is not None:
            samples.append(audio)

    print(f'  {len(samples)} usable pairs loaded.')
    if not samples:
        print('ERROR: No usable audio pairs. Aborting.')
        sys.exit(1)

    # ── Layer groups ──────────────────────────────────────────
    layer_groups = build_layer_groups()
    param_bits   = compute_param_bits(model, layer_groups)

    # ── Run sweep ─────────────────────────────────────────────
    baseline, sweep = run_sweep(model, layer_groups, samples, device, QUANT_FNS)

    # ── Save CSV ──────────────────────────────────────────────
    df = save_sensitivity_csv(sweep, out_dir / 'sensitivity_matrix.csv')

    # ── Save heatmaps ─────────────────────────────────────────
    save_heatmap(df, 'delta_si_snr',
                 'ΔSI-SNR vs FP8 baseline (dB, negative = degradation)',
                 out_dir / 'sensitivity_heatmap_snr.png')
    save_heatmap(df, 'delta_ild',
                 'ΔILD error vs FP8 baseline (dB)',
                 out_dir / 'sensitivity_heatmap_ild.png')
    save_heatmap(df, 'delta_ipd',
                 'ΔIPD error vs FP8 baseline (degrees)',
                 out_dir / 'sensitivity_heatmap_ipd.png')

    # Composite heatmap: all three metrics side by side
    fig, axes = plt.subplots(1, 3, figsize=(24, max(8, len(layer_groups) * 0.45 + 2)))
    for ax, metric, label in zip(
        axes,
        ['delta_si_snr', 'delta_ild', 'delta_ipd'],
        ['ΔSI-SNR (dB)', 'ΔILD (dB)', 'ΔIPD (°)'],
    ):
        pivot = df.pivot(index='layer', columns='quant_config', values=metric)
        col_order = [c for c in ['INT8', 'INT4', 'INT3', 'FP4'] if c in pivot.columns]
        pivot = pivot[col_order]
        row_order = sorted(
            pivot.index,
            key=lambda x: (
                0 if x.startswith('enc_dw') else
                1 if x.startswith('enc_pw') else
                2 if x.startswith('dec_dw') else
                3 if x.startswith('dec_pw') else 4, x
            )
        )
        pivot = pivot.loc[row_order]
        sns.heatmap(pivot.astype(float), ax=ax, cmap='RdYlGn_r',
                    annot=True, fmt='.2f', linewidths=0.3,
                    cbar_kws={'label': label})
        ax.set_title(label)
        ax.set_xlabel('Quant config')
        ax.set_ylabel('Layer' if ax is axes[0] else '')
    fig.suptitle('BCCTN layer-wise quantisation sensitivity', fontsize=14)
    plt.tight_layout()
    fig.savefig(out_dir / 'sensitivity_heatmap.png', dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'Saved: {out_dir / "sensitivity_heatmap.png"}')

    # ── Mixed-precision recommendation ────────────────────────
    thresholds = {
        'delta_si_snr': args.thr_snr,
        'delta_ild':    args.thr_ild,
        'delta_ipd':    args.thr_ipd,
    }
    config = make_mixed_precision_config(sweep, param_bits, thresholds)
    mp_json = out_dir / 'mixed_precision_config.json'
    with open(mp_json, 'w') as f:
        json.dump(config, f, indent=2)
    print(f'Saved: {mp_json}')

    # ── Console summary ───────────────────────────────────────
    print_summary(baseline, sweep, config)

    # ── Save baseline metrics ─────────────────────────────────
    baseline_out = {
        'si_snr_l':  baseline['si_snr_l'],
        'si_snr_r':  baseline['si_snr_r'],
        'si_snr_avg': (baseline['si_snr_l'] + baseline['si_snr_r']) / 2,
        'ild':        baseline['ild'],
        'ipd':        baseline['ipd'],
        'num_samples': len(samples),
    }
    with open(out_dir / 'fp8_baseline.json', 'w') as f:
        json.dump(baseline_out, f, indent=2)
    print(f'\nDone. Results in {out_dir}/')


if __name__ == '__main__':
    main()
