"""
Per-layer L-Mul *local operator error* heatmap.

For each Conv2d / ConvTranspose2d / Linear that's been L-Mul-patched:
  - capture the input the layer actually receives during the full L-Mul forward
  - compute L-Mul output (what the patched forward produces)
  - compute real-mult output (F.conv2d / F.conv_transpose2d / F.linear with the
    SAME weight, bias, and input that the L-Mul layer used)
  - relative error = MSE(real - lmul) / mean(real ** 2)

This isolates each layer's L-Mul approximation error from upstream divergence.
(Codex review of an earlier two-model design caught that comparing whole-network
activations gets polluted by upstream errors.)

COVERAGE LIMITATION (read this before treating output as the full picture):
This script ONLY measures L-Mul error inside Conv2d, ConvTranspose2d, and Linear
modules — the ones `enable_all_lmul_multiply()` patches via module-level forward
override. It does NOT measure MHA QK / Score×V L-Mul error, because those happen
inside custom matmul functions (`_lmul_matmul`, `_q08_packet_lmul_matmul`) called
directly from `ComplexMultiheadAttention.forward`, not from a module forward hook.
So this is a "conv/linear heatmap", not a "full datapath heatmap". To extend to
MHA, add explicit instrumentation in complexLayers.py at the call sites of the
L-Mul matmul helpers.

Output: CSV sorted descending by relative error, with cumulative contribution.

Usage:
  CUDA_VISIBLE_DEVICES=3 python scripts/per_layer_error_heatmap.py \
    --ckpt /path/to/all_lmul_m4_qat_best.ckpt \
    --num_samples 24 \
    --output_csv results/heatmap/per_layer_error.csv \
    --hydra_overrides ... (all the v3/v4 overrides)
"""
import argparse
import os
import re
import sys
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
import torch.nn as nn
import torch.nn.functional as F

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from hydra import compose, initialize


def load_model(ckpt_path, overrides):
    # config_path is relative to this file (scripts/), repo config is one level up.
    with initialize(config_path="../config", version_base="1.1"):
        cfg = compose(config_name="config", overrides=list(overrides or []))
    from DCNN.trainer import DCNNLightningModule
    lm = DCNNLightningModule(cfg)
    sd = torch.load(ckpt_path, map_location="cpu", weights_only=False)["state_dict"]
    cleaned = {}
    for k, v in sd.items():
        if k.startswith("loss."):
            continue
        if k.startswith("model."):
            k = k[len("model."):]
        cleaned[k] = v
    missing = lm.model.load_state_dict(cleaned, strict=False)
    if missing.missing_keys:
        print(f"[heatmap] {len(missing.missing_keys)} missing keys; first few: {missing.missing_keys[:3]}")
    if missing.unexpected_keys:
        print(f"[heatmap] {len(missing.unexpected_keys)} unexpected keys; first few: {missing.unexpected_keys[:3]}")
    lm.model.eval()
    return lm.model


def register_diagnostic_hooks(model):
    """Register forward_pre_hook + forward_hook on every Conv2d/ConvTranspose2d/Linear.

    Returns (saved_dict, handles). saved_dict[name] = {"input": tensor, "output": tensor}
    """
    saved = {}
    handles = []

    def make_pre(name):
        def fn(module, inputs):
            # inputs is a tuple; first positional is the activation
            if inputs and isinstance(inputs[0], torch.Tensor):
                saved.setdefault(name, {})["input"] = inputs[0].detach()
        return fn

    def make_post(name):
        def fn(module, inputs, output):
            if isinstance(output, torch.Tensor):
                saved.setdefault(name, {})["output"] = output.detach()
        return fn

    for name, mod in model.named_modules():
        if isinstance(mod, (nn.Conv2d, nn.ConvTranspose2d, nn.Linear)):
            handles.append(mod.register_forward_pre_hook(make_pre(name)))
            handles.append(mod.register_forward_hook(make_post(name)))

    return saved, handles


def compute_real_mult_output(mod, inp):
    """Real-mult reference using PyTorch's stock F.* ops on the same (weight, bias, input).

    Both the L-Mul forward and this function read `mod.weight` / `mod.bias`, so any
    QAT fake-quant applied to those tensors is shared between them; the only
    difference is the multiplier (L-Mul vs real).
    """
    if isinstance(mod, nn.Conv2d):
        if mod.padding_mode != "zeros":
            padded = F.pad(inp, mod._reversed_padding_repeated_twice, mode=mod.padding_mode)
            return F.conv2d(padded, mod.weight, mod.bias, mod.stride, (0, 0),
                            mod.dilation, mod.groups)
        return F.conv2d(inp, mod.weight, mod.bias, mod.stride, mod.padding,
                        mod.dilation, mod.groups)
    if isinstance(mod, nn.ConvTranspose2d):
        return F.conv_transpose2d(inp, mod.weight, mod.bias, mod.stride, mod.padding,
                                  mod.output_padding, mod.groups, mod.dilation)
    if isinstance(mod, nn.Linear):
        return F.linear(inp, mod.weight, mod.bias)
    raise ValueError(f"unsupported module type {type(mod).__name__}")


def gather_samples(noisy_dir, clean_dir, num_samples, snr_filter):
    """Pick (clean, noisy) pairs spread across SNR conditions."""
    noisy_root = Path(noisy_dir)
    clean_root = Path(clean_dir)

    snr_dirs = []
    for s in snr_filter:
        cand = noisy_root / f"snr_{s}dB"
        if cand.exists():
            snr_dirs.append(cand)
    if not snr_dirs:
        snr_dirs = [noisy_root]

    samples = []
    per_snr = max(1, num_samples // max(1, len(snr_dirs)))
    for snr_dir in snr_dirs:
        n_here = 0
        for noisy_path in sorted(snr_dir.glob("*.wav")):
            # Noisy filename: <base>_snr{+,-,empty}X.Y.wav; clean: <base>.wav
            clean_name = re.sub(r"_snr[+\-]?[\d.]+\.wav$", ".wav", noisy_path.name)
            clean_path = clean_root / clean_name
            if not clean_path.exists():
                hits = list(clean_root.rglob(clean_name))
                if not hits:
                    continue
                clean_path = hits[0]
            samples.append((clean_path, noisy_path))
            n_here += 1
            if n_here >= per_snr:
                break
        if len(samples) >= num_samples:
            break

    return samples


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--noisy_dir", default="/raid/R12K41024/BCCTN/Dataset/noisy_testset")
    ap.add_argument("--clean_dir", default="/raid/R12K41024/BCCTN/Dataset/clean_testset")
    ap.add_argument("--num_samples", type=int, default=24,
                    help="number of (noisy, clean) pairs (~3 per SNR for 8 SNRs)")
    ap.add_argument("--snr_filter", nargs="*",
                    default=["-6", "-3", "0", "3", "6", "9", "12", "15"])
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--output_csv", required=True,
                    help="output table sorted by relative error")
    ap.add_argument("--hydra_overrides", nargs="*", default=[])
    args = ap.parse_args()

    print(f"[heatmap] loading model from {args.ckpt}")
    model = load_model(args.ckpt, args.hydra_overrides)
    model = model.to(args.device)

    print(f"[heatmap] gathering samples from {args.noisy_dir}")
    samples = gather_samples(args.noisy_dir, args.clean_dir,
                             args.num_samples, args.snr_filter)
    if not samples:
        raise RuntimeError("no calibration samples found")
    print(f"[heatmap] using {len(samples)} samples")

    # Locate every L-Mul-targeted layer (must enumerate before forward so we don't
    # rely on hook execution order being stable across PyTorch versions).
    module_by_name = {
        n: m for n, m in model.named_modules()
        if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d, nn.Linear))
    }
    print(f"[heatmap] {len(module_by_name)} target modules "
          f"(Conv2d/ConvTranspose2d/Linear)")

    saved, handles = register_diagnostic_hooks(model)
    layer_stats = {}  # name -> {mse_sum, ref_pwr_sum, n, module, out_shape}

    with torch.no_grad():
        for i, (_, noisy_path) in enumerate(samples):
            noisy_data, _ = sf.read(noisy_path)
            if noisy_data.ndim == 1:
                continue
            x = torch.tensor(noisy_data.T, dtype=torch.float32,
                             device=args.device).unsqueeze(0)
            # The L-Mul-patched forward will run; hooks save (input, lmul_output)
            saved.clear()
            _ = model(x)

            # Per-layer compare against real-mult on the SAME captured input.
            for name, mod in module_by_name.items():
                if name not in saved or "input" not in saved[name] or "output" not in saved[name]:
                    continue
                inp = saved[name]["input"]
                lmul_out = saved[name]["output"]
                try:
                    real_out = compute_real_mult_output(mod, inp)
                except Exception as e:
                    print(f"[heatmap] skip {name}: {e}")
                    continue
                if real_out.shape != lmul_out.shape:
                    print(f"[heatmap] shape mismatch {name}: "
                          f"lmul={tuple(lmul_out.shape)} real={tuple(real_out.shape)}")
                    continue
                lo = lmul_out.float()
                ro = real_out.float()
                mse = ((ro - lo) ** 2).mean().item()
                ref_pwr = (ro ** 2).mean().item()
                st = layer_stats.setdefault(name, {
                    "mse_sum": 0.0, "ref_pwr_sum": 0.0, "n": 0,
                    "module": mod, "out_shape": tuple(ro.shape),
                })
                st["mse_sum"] += mse
                st["ref_pwr_sum"] += ref_pwr
                st["n"] += 1

            if (i + 1) % 5 == 0:
                print(f"[heatmap] processed {i + 1}/{len(samples)}")

    for h in handles:
        h.remove()

    # Aggregate
    rows = []
    for name, st in layer_stats.items():
        if st["n"] == 0:
            continue
        mse = st["mse_sum"] / st["n"]
        ref_pwr = st["ref_pwr_sum"] / st["n"]
        rel = mse / (ref_pwr + 1e-12)
        rel_db = 10.0 * np.log10(rel + 1e-12)
        rows.append({
            "name": name,
            "type": type(st["module"]).__name__,
            "rel_error": rel,
            "rel_error_db": rel_db,
            "mse": mse,
            "ref_pwr": ref_pwr,
            "out_shape": st["out_shape"],
            "n_samples": st["n"],
        })

    # Sort descending by relative error (worst-offender first)
    rows.sort(key=lambda r: -r["rel_error"])

    # Cumulative absolute MSE share (assumes additive variance; only a rough guide)
    total_mse = sum(r["mse"] for r in rows) or 1e-12
    cum = 0.0
    for r in rows:
        cum += r["mse"]
        r["cumulative_pct"] = 100.0 * cum / total_mse

    os.makedirs(os.path.dirname(args.output_csv) or ".", exist_ok=True)
    with open(args.output_csv, "w") as fp:
        fp.write("rank,name,type,rel_error,rel_error_dB,mse,ref_pwr,cumulative_pct_by_mse,out_shape,n_samples\n")
        for i, r in enumerate(rows):
            fp.write(f"{i+1},{r['name']},{r['type']},{r['rel_error']:.6e},"
                     f"{r['rel_error_db']:.2f},{r['mse']:.6e},{r['ref_pwr']:.6e},"
                     f"{r['cumulative_pct']:.2f},\"{r['out_shape']}\",{r['n_samples']}\n")

    print(f"\n[heatmap] wrote {args.output_csv}")
    print("\n=== TOP 20 BY RELATIVE L-Mul ERROR ===")
    print(f"{'rank':>4} {'type':<18} {'name':<50} {'rel_err':>10} {'rel_dB':>7} {'cum%':>6}")
    for i, r in enumerate(rows[:20]):
        print(f"{i+1:>4} {r['type']:<18} {r['name']:<50} "
              f"{r['rel_error']:>10.3e} {r['rel_error_db']:>7.2f} "
              f"{r['cumulative_pct']:>6.1f}")


if __name__ == "__main__":
    main()
