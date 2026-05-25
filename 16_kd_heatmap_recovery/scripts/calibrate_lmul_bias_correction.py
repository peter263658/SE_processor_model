"""
Calibrate per-channel L-Mul vs real-mult bias-correction for Conv2d /
ConvTranspose2d layers in the all-L-Mul student.

What this measures
------------------
For each target layer, given the *same* input the L-Mul model actually feeds
into the layer, compute both:
  - L-Mul output (what the patched forward produces)
  - real-mult output (F.conv2d / F.conv_transpose2d with the SAME weight, bias)

Then accumulate per-output-channel δ[c]:
  δ[layer, c] = mean over (batch, spatial, samples) of (real_out[c] - lmul_out[c])

This is the *local* L-Mul arithmetic offset; it has NOTHING to do with teacher
distillation. δ is the constant additive offset to add to L-Mul's output so it
matches real-mult on average. It can be:
  - applied at runtime via BCCTN_BIAS_CORRECTION_PATH (toggleable, ablation-friendly)
  - baked into module.bias via --bake_ckpt_out (one-step deploy)

Scope (per codex guardrails 2026-05-25)
---------------------------------------
v1 only calibrates Conv2d + ConvTranspose2d.  Linear / MHA are skipped because:
  - D-probe showed ConvT carries the most L-Mul bias signal
  - keeping scope narrow makes the bake/runtime contract simpler

Sample count: N=128 default (codex: 64~256 is enough to see signal; if you need
1000+ to stabilise, the calibration is too sample-dependent to deploy).

Outputs
-------
  <output_pt>           dict {"_meta": {...}, layer_name: tensor[C_out]}
                        loadable by enable_all_lmul_multiply via BCCTN_BIAS_CORRECTION_PATH
  <bake_ckpt_out>       (optional) student ckpt with δ added into the layer's bias
                        parameter. Do NOT also enable runtime BCCTN_BIAS_CORRECTION_PATH
                        when loading this baked ckpt — that would double-apply.

Usage
-----
  CUDA_VISIBLE_DEVICES=6 python scripts/calibrate_lmul_bias_correction.py \\
    --ckpt /path/to/v4_ep5.ckpt \\
    --num_samples 128 \\
    --output_pt results/bias_correction/v4_ep5_convonly_N128.pt \\
    --bake_ckpt_out results/bias_correction/v4_ep5_convonly_N128_baked.ckpt \\
    --hydra_overrides <all student-side overrides matching the ckpt>
"""
from __future__ import annotations

import argparse
import datetime
import hashlib
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


def _ckpt_sha(path: str) -> str:
    h = hashlib.sha1()
    with open(path, "rb") as f:
        h.update(f.read(8 * 1024 * 1024))
    return h.hexdigest()[:16]


def load_student(ckpt_path: str, overrides):
    with initialize(config_path="../config", version_base="1.1"):
        cfg = compose(config_name="config", overrides=list(overrides))
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
    res = lm.model.load_state_dict(cleaned, strict=False)
    if res.missing_keys:
        print(f"[calib] {len(res.missing_keys)} missing keys; first: {res.missing_keys[:3]}")
    if res.unexpected_keys:
        print(f"[calib] {len(res.unexpected_keys)} unexpected keys; first: {res.unexpected_keys[:3]}")
    lm.model.eval()
    return lm


def compute_real_mult_output(mod: nn.Module, inp: torch.Tensor) -> torch.Tensor:
    """Real-mult reference via stock F.* ops, sharing this module's weight/bias."""
    if isinstance(mod, nn.Conv2d):
        if mod.padding_mode != "zeros":
            padded = F.pad(inp, mod._reversed_padding_repeated_twice, mode=mod.padding_mode)
            return F.conv2d(padded, mod.weight, mod.bias, mod.stride, (0, 0),
                            mod.dilation, mod.groups)
        return F.conv2d(inp, mod.weight, mod.bias, mod.stride, mod.padding,
                        mod.dilation, mod.groups)
    if isinstance(mod, nn.ConvTranspose2d):
        # ConvTranspose2d has no padding_mode attribute; use stride/padding/
        # output_padding/groups/dilation/bias directly per codex guardrail #2.
        return F.conv_transpose2d(inp, mod.weight, mod.bias, mod.stride, mod.padding,
                                  mod.output_padding, mod.groups, mod.dilation)
    if isinstance(mod, nn.Linear):
        return F.linear(inp, mod.weight, mod.bias)
    raise TypeError(f"unsupported module {type(mod).__name__} (v1 is conv-only)")


def gather_samples(noisy_dir, num_samples, snr_filter, sample_seed):
    noisy_root = Path(noisy_dir)
    rng = np.random.default_rng(int(sample_seed))
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
        candidates = sorted(snr_dir.glob("*.wav"))
        if candidates:
            order = rng.permutation(len(candidates))
        else:
            order = []
        n = 0
        for idx in order:
            noisy_path = candidates[int(idx)]
            samples.append(noisy_path)
            n += 1
            if n >= per_snr:
                break
        if len(samples) >= num_samples:
            break
    return samples


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True, help="student ckpt (v4 Ep5 etc.)")
    ap.add_argument("--noisy_dir",
                    default="/raid/R12K41024/BCCTN/Dataset/noisy_trainset",
                    help="calibration audio root. Defaults to trainset to avoid "
                         "test-set calibration leakage.")
    ap.add_argument("--num_samples", type=int, default=128,
                    help="codex guardrail: 64-256 enough. Default 128.")
    ap.add_argument("--sample_seed", type=int, default=42,
                    help="deterministic shuffle seed for calibration samples")
    ap.add_argument("--snr_filter", nargs="*",
                    default=["-6", "-3", "0", "3", "6", "9", "12", "15"])
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--output_pt", required=True,
                    help="where to write {layer_name: tensor[C_out]} dict")
    ap.add_argument("--bake_ckpt_out", default=None,
                    help="optional: write a new ckpt with δ baked into module.bias. "
                         "DO NOT enable BCCTN_BIAS_CORRECTION_PATH when loading "
                         "the baked ckpt — that would double-apply.")
    ap.add_argument("--include_linear", action="store_true",
                    help="v1 default skips Linear (codex guardrail #3). "
                         "Set this to also calibrate Linear layers.")
    ap.add_argument("--hydra_overrides", nargs="*", default=[])
    args = ap.parse_args()

    if not os.path.isfile(args.ckpt):
        raise FileNotFoundError(args.ckpt)

    print(f"[calib] loading student from {args.ckpt}")
    lm = load_student(args.ckpt, args.hydra_overrides)
    lm.model.to(args.device).eval()

    # Identify target layers (conv only by default; codex guardrail #3)
    target_types = [nn.Conv2d, nn.ConvTranspose2d]
    if args.include_linear:
        target_types.append(nn.Linear)
    target_modules = {
        name: mod for name, mod in lm.model.named_modules()
        if any(isinstance(mod, t) for t in target_types)
    }
    print(f"[calib] {len(target_modules)} target layers "
          f"({[t.__name__ for t in target_types]})")

    # Forward hooks: capture (input, lmul_output) for each target layer
    saved = {}
    handles = []
    def make_pre(name):
        def fn(module, inputs):
            if inputs and isinstance(inputs[0], torch.Tensor):
                saved.setdefault(name, {})["input"] = inputs[0].detach()
        return fn
    def make_post(name):
        def fn(module, inputs, output):
            if isinstance(output, torch.Tensor):
                saved.setdefault(name, {})["output"] = output.detach()
        return fn
    for name, mod in target_modules.items():
        handles.append(mod.register_forward_pre_hook(make_pre(name)))
        handles.append(mod.register_forward_hook(make_post(name)))

    samples = gather_samples(args.noisy_dir, args.num_samples, args.snr_filter,
                             args.sample_seed)
    if not samples:
        raise RuntimeError("no calibration samples found")
    print(f"[calib] using {len(samples)} samples")

    # Accumulators: per layer, running sum of (real - lmul) over channels.
    # Using float64 accumulator for numerical stability across many spatial bins.
    accum = {}      # name -> tensor[C_out] float64
    count = {}      # name -> total count (batch*spatial elements)

    with torch.no_grad():
        for i, noisy_path in enumerate(samples):
            noisy_data, _ = sf.read(noisy_path)
            if noisy_data.ndim == 1:
                continue
            x = torch.tensor(noisy_data.T, dtype=torch.float32,
                             device=args.device).unsqueeze(0)
            saved.clear()
            _ = lm.model(x)

            for name, mod in target_modules.items():
                if name not in saved or "input" not in saved[name] or "output" not in saved[name]:
                    continue
                inp = saved[name]["input"]
                lmul_out = saved[name]["output"]
                try:
                    real_out = compute_real_mult_output(mod, inp)
                except TypeError:
                    # Linear is filtered above but defensive
                    continue
                except Exception as e:
                    print(f"[calib] skip {name}: {e}")
                    continue
                if real_out.shape != lmul_out.shape:
                    continue

                diff = (real_out.float() - lmul_out.float())   # [B, C, ...] or [B, ..., C] for Linear
                # Reduce over all dims except the channel dim.
                # Conv2d/ConvT2d: output is [B, C, H, W] → reduce over (0, 2, 3)
                # Linear (defensive): output is [..., C] → reduce over all but last
                if isinstance(mod, (nn.Conv2d, nn.ConvTranspose2d)):
                    # sum over B, H, W
                    per_ch_sum = diff.sum(dim=(0, 2, 3))   # [C]
                    n = diff.shape[0] * diff.shape[2] * diff.shape[3]
                elif isinstance(mod, nn.Linear):
                    # treat all leading dims as batch
                    flat = diff.reshape(-1, diff.shape[-1])  # [N, C]
                    per_ch_sum = flat.sum(dim=0)
                    n = flat.shape[0]
                else:
                    continue

                if name not in accum:
                    accum[name] = per_ch_sum.double().cpu()
                    count[name] = n
                else:
                    accum[name] += per_ch_sum.double().cpu()
                    count[name] += n

            if (i + 1) % 16 == 0:
                print(f"[calib] processed {i+1}/{len(samples)}")

    for h in handles:
        h.remove()

    # Finalise δ = sum / count (per channel mean)
    delta_dict = {}
    summary = []
    for name in sorted(accum.keys()):
        delta = (accum[name] / count[name]).float()  # [C_out]
        delta_dict[name] = delta
        layer_type = type(target_modules[name]).__name__
        summary.append((name, layer_type, float(delta.abs().max()),
                        float(delta.abs().mean()),
                        float(delta.std())))

    summary.sort(key=lambda r: -r[3])  # by mean |δ|
    print("\n=== TOP 20 layers by mean |δ| ===")
    print(f"{'name':<55} {'type':<18} {'max|δ|':>10} {'mean|δ|':>10} {'std':>10}")
    for r in summary[:20]:
        print(f"{r[0]:<55} {r[1]:<18} {r[2]:>10.3e} {r[3]:>10.3e} {r[4]:>10.3e}")

    # Save .pt with metadata wrapped in a special key so the runtime loader
    # can ignore it; the loader only iterates dict items whose value is a tensor.
    # To keep enable_all_lmul_multiply's dict access simple, separate dict.
    out_payload = dict(delta_dict)
    out_payload["_meta"] = {
        "mode": "runtime_buffer" if args.bake_ckpt_out is None else "calibrated_and_baked",
        "source_ckpt": args.ckpt,
        "source_ckpt_sha1_prefix": _ckpt_sha(args.ckpt),
        "num_samples": len(samples),
        "noisy_dir": str(args.noisy_dir),
        "snr_filter": list(args.snr_filter),
        "include_linear": bool(args.include_linear),
        "hydra_overrides": list(args.hydra_overrides),
        "calibrated_at": datetime.datetime.now().isoformat(),
    }
    os.makedirs(os.path.dirname(args.output_pt) or ".", exist_ok=True)
    torch.save(out_payload, args.output_pt)
    print(f"\n[calib] saved bias-correction → {args.output_pt}")
    print(f"        meta: mode={out_payload['_meta']['mode']}, "
          f"N={out_payload['_meta']['num_samples']}, "
          f"ckpt_sha={out_payload['_meta']['source_ckpt_sha1_prefix']}")
    print("        NOTE: the runtime loader skips the '_meta' key automatically.")

    # Optional bake: load ckpt, add δ into module.bias, save new ckpt.
    # Codex guardrail #4: bake → DO NOT also set BCCTN_BIAS_CORRECTION_PATH.
    if args.bake_ckpt_out:
        print(f"\n[calib] baking δ into source ckpt → {args.bake_ckpt_out}")
        src = torch.load(args.ckpt, map_location="cpu", weights_only=False)
        sd = src["state_dict"]
        # Bias keys in BCCTN state_dict look like "model.<path>.bias" or
        # "<path>.bias". Find matching bias for each calibrated layer.
        n_baked = 0
        for name, delta in delta_dict.items():
            # The model's state_dict uses "model." prefix; layer names from
            # lm.model.named_modules() do NOT have that prefix. Try both.
            candidates = [f"model.{name}.bias", f"{name}.bias"]
            bias_key = next((k for k in candidates if k in sd), None)
            if bias_key is None:
                # Some convs have bias=False; in that case we cannot bake here
                # (would need to add a bias parameter — out of scope for v1).
                print(f"[bake] skip {name}: no .bias entry in state_dict "
                      f"(layer may have bias=False; use runtime buffer instead)")
                continue
            old = sd[bias_key].float()
            if old.shape != delta.shape:
                print(f"[bake] skip {name}: bias shape {tuple(old.shape)} "
                      f"!= δ shape {tuple(delta.shape)}")
                continue
            sd[bias_key] = old + delta.to(old.dtype)
            n_baked += 1
        src["state_dict"] = sd
        # Also stash provenance so we can detect double-application later
        src.setdefault("bcctn_bias_correction_baked_meta", out_payload["_meta"])
        os.makedirs(os.path.dirname(args.bake_ckpt_out) or ".", exist_ok=True)
        torch.save(src, args.bake_ckpt_out)
        print(f"[bake] baked {n_baked}/{len(delta_dict)} layers into "
              f"{args.bake_ckpt_out}")
        print(f"[bake] IMPORTANT: DO NOT also set BCCTN_BIAS_CORRECTION_PATH "
              f"when loading this ckpt (double-apply).")


if __name__ == "__main__":
    main()
