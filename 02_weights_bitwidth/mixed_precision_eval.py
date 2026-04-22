#!/usr/bin/env python3
"""
Full evaluation with mixed-precision quantisation vs baselines.

Compares three configurations on the complete test set:
  1. Uniform W8 (INT8 all layers)
  2. Uniform W4 (INT4 all layers)
  3. Mixed-precision (per-layer assignment from mixed_precision_config.json)

Metrics: SI-SNR, ILD error, IPD error, and (optionally) MBSTOI.

Usage:
    python mixed_precision_eval.py --checkpoint model.ckpt
    python mixed_precision_eval.py --checkpoint model.ckpt --num_samples 200 --with_mbstoi
"""

import argparse
import json
import re
import sys
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import pandas as pd
import soundfile as sf
import torch
from tqdm import tqdm

# ──────────────────────────────────────────────────────────────
# Shared quantisation utilities (mirrors bitwidth_sweep.py)
# ──────────────────────────────────────────────────────────────

def _quant_int(t: torch.Tensor, bits: int) -> torch.Tensor:
    if t.is_complex():
        return torch.complex(
            _quant_int(t.real.contiguous(), bits),
            _quant_int(t.imag.contiguous(), bits),
        )
    qmax = 2 ** (bits - 1) - 1
    amax = t.abs().max().clamp(min=1e-8)
    scale = amax / qmax
    return torch.round(t / scale).clamp(-qmax, qmax) * scale


_FP4_VALS = torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0])


def _quant_fp4(t: torch.Tensor) -> torch.Tensor:
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
    idx = (norm.abs().unsqueeze(-1) - fp4.unsqueeze(0)).abs().argmin(dim=-1)
    return (fp4[idx] * sign * scale).reshape(shape)


def _get_quant_fn(bits: int, fmt: str = 'INT'):
    if fmt == 'FP' and bits == 4:
        return _quant_fp4
    return lambda t: _quant_int(t, bits)


# ──────────────────────────────────────────────────────────────
# Layer groups (must match bitwidth_sweep.py exactly)
# ──────────────────────────────────────────────────────────────

def build_layer_groups() -> dict:
    g = {}
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
    g['mattn_input_proj'] = ['mattn.input_proj.parametrizations.weight.original']
    g['mattn_in_proj_re'] = [
        'mattn.mattn.MultiheadAttention_re.parametrizations.in_proj_weight.original']
    g['mattn_in_proj_im'] = [
        'mattn.mattn.MultiheadAttention_im.parametrizations.in_proj_weight.original']
    g['mattn_out_proj_re'] = [
        'mattn.mattn.MultiheadAttention_re.out_proj.parametrizations.weight.original']
    g['mattn_out_proj_im'] = [
        'mattn.mattn.MultiheadAttention_im.out_proj.parametrizations.weight.original']
    g['mattn_output_proj'] = ['mattn.output_proj.parametrizations.weight.original']
    return g


# ──────────────────────────────────────────────────────────────
# Model loading
# ──────────────────────────────────────────────────────────────

def load_model(checkpoint_path: str, device: torch.device):
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
# Weight patching
# ──────────────────────────────────────────────────────────────

@contextmanager
def apply_quantisation(model: torch.nn.Module,
                       layer_groups: dict,
                       layer_config: dict):
    """
    Context manager: apply per-layer quantisation from layer_config dict.
    layer_config maps layer_name → {'bits': int, 'format': str}
    Only layers present in layer_config (with bits < 32) are quantised.
    """
    named = dict(model.named_parameters())
    saved: dict[str, torch.Tensor] = {}

    for layer_name, cfg in layer_config.items():
        bits = cfg.get('bits', 32)
        fmt  = cfg.get('format', 'INT')
        if bits >= 32 or layer_name not in layer_groups:
            continue
        qfn = _get_quant_fn(bits, fmt)
        for p in layer_groups[layer_name]:
            if p in named:
                saved[p] = named[p].data.clone()
                named[p].data.copy_(qfn(named[p].data))

    try:
        yield
    finally:
        for p, orig in saved.items():
            named[p].data.copy_(orig)


def uniform_config(layer_groups: dict, bits: int, fmt: str = 'INT') -> dict:
    return {name: {'bits': bits, 'format': fmt} for name in layer_groups}


# ──────────────────────────────────────────────────────────────
# Metrics
# ──────────────────────────────────────────────────────────────

def si_snr(clean: np.ndarray, enh: np.ndarray) -> float:
    c = clean - clean.mean()
    e = enh   - enh.mean()
    alpha = np.dot(e, c) / (np.dot(c, c) + 1e-8)
    s_t = alpha * c
    e_n = e - s_t
    val = 10.0 * np.log10((np.dot(s_t, s_t) + 1e-8) / (np.dot(e_n, e_n) + 1e-8))
    return float(np.clip(val, -30.0, 40.0))


def _stft(x, n_fft=128, hop=80, win=128):
    import librosa
    return librosa.stft(x, n_fft=n_fft, hop_length=hop, win_length=win)


def ild_error(cl, cr, el, er) -> float:
    eps = 1e-8
    CL = _stft(cl); CR = _stft(cr)
    EL = _stft(el); ER = _stft(er)
    ild_c = 20 * np.log10(np.abs(CL) + eps) - 20 * np.log10(np.abs(CR) + eps)
    ild_e = 20 * np.log10(np.abs(EL) + eps) - 20 * np.log10(np.abs(ER) + eps)
    energy = np.maximum(np.abs(CL) ** 2, np.abs(CR) ** 2)
    mx = energy.max(axis=1, keepdims=True).clip(min=eps)
    mask = 10 * np.log10(energy / mx) > -20
    err = np.abs(ild_c - ild_e)
    return float(err[mask].mean()) if mask.any() else 0.0


def ipd_error(cl, cr, el, er, f_max=1500, sr=16000) -> float:
    eps = 1e-8
    n_fft = 128
    CL = _stft(cl); CR = _stft(cr)
    EL = _stft(el); ER = _stft(er)
    max_bin = int(f_max / (sr / n_fft)) + 1
    CL, CR, EL, ER = CL[:max_bin], CR[:max_bin], EL[:max_bin], ER[:max_bin]
    energy = np.maximum(np.abs(CL) ** 2, np.abs(CR) ** 2)
    mx = energy.max(axis=1, keepdims=True).clip(min=eps)
    mask = 10 * np.log10(energy / mx) > -20
    ipd_c = np.angle(CL * np.conj(CR))
    ipd_e = np.angle(EL * np.conj(ER))
    err = np.abs(np.angle(np.exp(1j * (ipd_c - ipd_e)))) * 180.0 / np.pi
    return float(err[mask].mean()) if mask.any() else 0.0


def mbstoi_score(cl, cr, el, er, sr=16000):
    """Return MBSTOI ∈ [0, 1] or None on failure."""
    import math
    try:
        from MBSTOI.mbstoi import mbstoi
    except ImportError:
        return None
    min_len = min(len(cl), len(cr), len(el), len(er))
    target = max(4096, 256 * math.ceil(min_len / 256))

    def prep(s):
        s = s[:min_len].copy()
        if len(s) < target:
            s = np.pad(s, (0, target - len(s)))
        s -= s.mean()
        mx = np.abs(s).max()
        if mx > 0:
            s /= mx
        return s

    try:
        score = mbstoi(prep(cl), prep(cr), prep(el), prep(er), gridcoarseness=1)
        if np.isnan(score) or not (0 <= score <= 1):
            return None
        return float(score)
    except Exception:
        return None


# ──────────────────────────────────────────────────────────────
# Dataset helpers
# ──────────────────────────────────────────────────────────────

def collect_pairs(noisy_root: Path, clean_dir: Path,
                  n: int | None = None, seed: int = 42):
    pattern = re.compile(r'^(.+_az-?\d+)_snr.*$')
    noisy_files = sorted(noisy_root.glob('**/*.wav'))
    pairs = []
    for nf in noisy_files:
        m = pattern.match(nf.stem)
        if m:
            cf = clean_dir / (m.group(1) + '.wav')
            if cf.exists():
                pairs.append((nf, cf))
    import random
    rng = random.Random(seed)
    rng.shuffle(pairs)
    return pairs[:n] if n else pairs


def load_pair(noisy_path, clean_path):
    import librosa
    noisy, sr_n = sf.read(noisy_path)
    clean, sr_c = sf.read(clean_path)
    if noisy.ndim == 1 or clean.ndim == 1:
        return None
    if min(len(noisy), len(clean)) / sr_c < 0.3:
        return None
    if sr_n != 16000:
        noisy = np.stack(
            [librosa.resample(noisy[:, ch], orig_sr=sr_n, target_sr=16000)
             for ch in range(2)], axis=1)
    if sr_c != 16000:
        clean = np.stack(
            [librosa.resample(clean[:, ch], orig_sr=sr_c, target_sr=16000)
             for ch in range(2)], axis=1)
    return noisy.T.astype(np.float32), clean.T.astype(np.float32)


# ──────────────────────────────────────────────────────────────
# Evaluation engine
# ──────────────────────────────────────────────────────────────

def evaluate_config(model, samples, device, with_mbstoi: bool = False) -> dict:
    """Run inference on all samples and return averaged metrics."""
    si_snr_vals, ild_vals, ipd_vals, mbstoi_vals = [], [], [], []

    with torch.no_grad():
        for noisy_np, clean_np in samples:
            inp = torch.from_numpy(noisy_np).unsqueeze(0).to(device)
            enh_np = model(inp).cpu().numpy()[0]  # [2, T']

            T = min(clean_np.shape[1], enh_np.shape[1])
            c = clean_np[:, :T]
            e = enh_np[:, :T]

            si_snr_vals.append((si_snr(c[0], e[0]) + si_snr(c[1], e[1])) / 2)
            ild_vals.append(ild_error(c[0], c[1], e[0], e[1]))
            ipd_vals.append(ipd_error(c[0], c[1], e[0], e[1]))

            if with_mbstoi:
                score = mbstoi_score(c[0], c[1], e[0], e[1])
                if score is not None:
                    mbstoi_vals.append(score)

    result = {
        'si_snr_avg':  float(np.mean(si_snr_vals)),
        'si_snr_std':  float(np.std(si_snr_vals)),
        'ild_avg':     float(np.mean(ild_vals)),
        'ild_std':     float(np.std(ild_vals)),
        'ipd_avg':     float(np.mean(ipd_vals)),
        'ipd_std':     float(np.std(ipd_vals)),
        'n_samples':   len(si_snr_vals),
    }
    if mbstoi_vals:
        result['mbstoi_avg'] = float(np.mean(mbstoi_vals))
        result['mbstoi_std'] = float(np.std(mbstoi_vals))
    return result


# ──────────────────────────────────────────────────────────────
# Memory footprint calculator
# ──────────────────────────────────────────────────────────────

def compute_weight_bytes(model, layer_groups, layer_config: dict) -> dict:
    """
    Return bytes consumed by each configuration.
    FP8 baseline = 1 byte per parameter element.
    """
    named = dict(model.named_parameters())
    total_params = sum(
        t.numel()
        for name, t in named.items()
        if 'weight' in name or 'bias' in name
    )

    config_params = {name: 0 for name in layer_groups}
    for layer_name, paths in layer_groups.items():
        config_params[layer_name] = sum(
            named[p].numel() for p in paths if p in named)

    total_quant_params = sum(config_params.values())

    breakdown = {}
    for cfg_name, bit_override in layer_config.items():
        if isinstance(bit_override, dict):
            # Mixed-precision: per-layer bits
            bytes_used = 0
            for layer_name, lcfg in bit_override.items():
                n = config_params.get(layer_name, 0)
                b = lcfg.get('bits', 8)
                bytes_used += n * b // 8
            breakdown[cfg_name] = {
                'quant_weight_bytes': bytes_used,
                'fp8_baseline_bytes': total_quant_params,
                'savings_pct': round(100 * (1 - bytes_used / total_quant_params), 1)
                               if total_quant_params else 0,
            }
        else:
            # Uniform: bit_override is an int
            bits = bit_override
            bytes_used = total_quant_params * bits // 8
            breakdown[cfg_name] = {
                'quant_weight_bytes': bytes_used,
                'fp8_baseline_bytes': total_quant_params,
                'savings_pct': round(100 * (1 - bytes_used / total_quant_params), 1)
                               if total_quant_params else 0,
            }
    return breakdown


# ──────────────────────────────────────────────────────────────
# Report writer
# ──────────────────────────────────────────────────────────────

def format_row(label, metrics, mbstoi=True):
    mb = f"{metrics.get('mbstoi_avg', float('nan')):.4f}" if mbstoi else 'N/A'
    return (f"{label:<25} "
            f"SI-SNR={metrics['si_snr_avg']:>7.3f}±{metrics['si_snr_std']:.2f}  "
            f"ILD={metrics['ild_avg']:>6.3f}±{metrics['ild_std']:.2f}  "
            f"IPD={metrics['ipd_avg']:>6.2f}±{metrics['ipd_std']:.2f}  "
            f"MBSTOI={mb}")


def write_report(results: dict, mem: dict, out_path: Path):
    lines = [
        'BCCTN Mixed-Precision Quantisation Evaluation',
        '=' * 60,
        '',
    ]
    for cfg_name, metrics in results.items():
        lines.append(format_row(cfg_name, metrics,
                                mbstoi='mbstoi_avg' in metrics))
        m = mem.get(cfg_name, {})
        lines.append(f"  Memory: {m.get('quant_weight_bytes', 0) / 1024:.1f} KB  "
                     f"(FP8 baseline: {m.get('fp8_baseline_bytes', 0) / 1024:.1f} KB, "
                     f"savings: {m.get('savings_pct', 0):.1f}%)")
        lines.append('')

    # Delta vs FP8 baseline
    if 'FP8 baseline' in results and 'Mixed-precision' in results:
        b = results['FP8 baseline']
        mp = results['Mixed-precision']
        lines += [
            'Deltas (Mixed-precision − FP8 baseline):',
            f"  ΔSI-SNR: {mp['si_snr_avg'] - b['si_snr_avg']:+.3f} dB",
            f"  ΔILD   : {mp['ild_avg']    - b['ild_avg']:+.3f} dB",
            f"  ΔIPD   : {mp['ipd_avg']    - b['ipd_avg']:+.2f}°",
        ]
        if 'mbstoi_avg' in mp and 'mbstoi_avg' in b:
            lines.append(
                f"  ΔMBSTOI: {mp['mbstoi_avg'] - b['mbstoi_avg']:+.4f}")
        lines.append('')

    text = '\n'.join(lines)
    out_path.write_text(text)
    print(text)
    print(f'Saved: {out_path}')


# ──────────────────────────────────────────────────────────────
# Entry point
# ──────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description='BCCTN mixed-precision evaluation')
    p.add_argument('--checkpoint', default='model.ckpt')
    p.add_argument('--noisy_dir',
                   default='/raid/R12K41024/BCCTN/Dataset/noisy_testset',
                   help='Root of noisy test set (may be nested by SNR)')
    p.add_argument('--clean_dir',
                   default='/raid/R12K41024/BCCTN/Dataset/clean_testset')
    p.add_argument('--mp_config',
                   default='results/mixed_precision_config.json',
                   help='JSON produced by bitwidth_sweep.py')
    p.add_argument('--num_samples', type=int, default=None,
                   help='Limit number of samples (None = use all)')
    p.add_argument('--out_dir', default='results')
    p.add_argument('--with_mbstoi', action='store_true',
                   help='Compute MBSTOI (slow, ~0.5 s/sample)')
    p.add_argument('--seed', type=int, default=42)
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
    layer_groups = build_layer_groups()

    # ── Load eval samples ─────────────────────────────────────
    noisy_root = Path(args.noisy_dir)
    clean_dir  = Path(args.clean_dir)
    print(f'Collecting pairs from {noisy_root} …')
    pairs = collect_pairs(noisy_root, clean_dir, args.num_samples, args.seed)
    print(f'  Found {len(pairs)} pairs; loading audio …')
    samples = []
    for nf, cf in tqdm(pairs, desc='Loading'):
        audio = load_pair(nf, cf)
        if audio is not None:
            samples.append(audio)
    print(f'  {len(samples)} usable samples.')
    if not samples:
        print('ERROR: No usable audio pairs.')
        sys.exit(1)

    # ── Load mixed-precision config ───────────────────────────
    mp_config_path = Path(args.mp_config)
    if mp_config_path.exists():
        with open(mp_config_path) as f:
            raw_mp = json.load(f)
        # Extract per-layer bits (drop _summary key)
        mp_layer_config = {k: v for k, v in raw_mp.items()
                           if not k.startswith('_') and isinstance(v, dict)}
        print(f'Loaded mixed-precision config from {mp_config_path}')
        print('  Tier distribution:',
              raw_mp.get('_summary', {}).get('tier_counts', '—'))
    else:
        print(f'[warn] {mp_config_path} not found — using default mixed-precision '
              'heuristic (enc_dw→W8, enc_pw→W4, dec_dw→W8, dec_pw→W4, mha→W4)')
        mp_layer_config = {name: {'bits': 8 if 'dw' in name else 4, 'format': 'INT'}
                           for name in layer_groups}

    # ── Define evaluation configurations ──────────────────────
    configs = {
        'FP8 baseline':    None,          # no patching
        'Uniform INT8':    uniform_config(layer_groups, 8),
        'Uniform INT4':    uniform_config(layer_groups, 4),
        'Uniform INT3':    uniform_config(layer_groups, 3),
        'Mixed-precision': mp_layer_config,
    }

    # ── Run evaluations ───────────────────────────────────────
    results = {}
    for cfg_name, layer_cfg in configs.items():
        print(f'\n[{cfg_name}] Evaluating {len(samples)} samples …')
        if layer_cfg is None:
            # FP8 baseline — no quantisation applied
            metrics = evaluate_config(model, samples, device, args.with_mbstoi)
        else:
            with apply_quantisation(model, layer_groups, layer_cfg):
                metrics = evaluate_config(model, samples, device, args.with_mbstoi)
        results[cfg_name] = metrics
        print(f'  SI-SNR={metrics["si_snr_avg"]:.3f}  '
              f'ILD={metrics["ild_avg"]:.3f}  IPD={metrics["ipd_avg"]:.2f}°'
              + (f'  MBSTOI={metrics.get("mbstoi_avg", float("nan")):.4f}'
                 if args.with_mbstoi else ''))

    # ── Memory footprint ──────────────────────────────────────
    mem_cfgs = {
        'Uniform INT8':  8,
        'Uniform INT4':  4,
        'Uniform INT3':  3,
        'Mixed-precision': mp_layer_config,
    }
    mem = compute_weight_bytes(model, layer_groups, mem_cfgs)

    # ── Save JSON results ─────────────────────────────────────
    json_out = out_dir / 'mixed_precision_results.json'
    with open(json_out, 'w') as f:
        json.dump({'metrics': results, 'memory': mem}, f, indent=2)
    print(f'\nSaved: {json_out}')

    # ── Save text report ──────────────────────────────────────
    write_report(results, mem, out_dir / 'mixed_precision_eval.txt')

    # ── Save per-config CSV ───────────────────────────────────
    rows = []
    for cfg_name, m in results.items():
        rows.append({'config': cfg_name, **m,
                     **{f'mem_{k}': v for k, v in mem.get(cfg_name, {}).items()}})
    pd.DataFrame(rows).to_csv(out_dir / 'mixed_precision_summary.csv', index=False)
    print(f'Saved: {out_dir / "mixed_precision_summary.csv"}')

    print('\nDone.')


if __name__ == '__main__':
    main()
