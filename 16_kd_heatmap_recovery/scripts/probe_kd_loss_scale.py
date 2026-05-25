"""
Loss-scale probe for KD: measure relative magnitudes of task_loss vs kd_loss
to calibrate `lambda_kd` BEFORE launching another KD scout.

Codex review of the first KD smoke caught that with lambda_kd=0.1, the KD term's
contribution to total loss was ~1e-5 of task_loss — essentially "KD turned off
in disguise". This probe gives a principled lambda recommendation by measuring
real per-batch loss magnitudes on a held-out subset of the training set.

For N batches (default 20), records:
  task_loss             (the BCCTN composite loss: SNR + STOI + ILD + IPD vs clean)
  kd_loss_waveform_mse  (student_out vs teacher_out, time-domain MSE)
  kd_loss_stft_l1       (student_out vs teacher_out, STFT magnitude L1)
  kd_loss_binaural      (same BinauralLoss instance applied with teacher as target;
                         this is signed like task_loss, magnitude-sized via abs())

Then reports:
  per-batch values
  aggregate stats (mean / median / min / max)
  lambda recommendations for each kd mode:
    - "equal contribution"  : λ * |kd_loss| == |task_loss|
    - "10% contribution"    : λ * |kd_loss| ==  0.10 * |task_loss|
    - "1% contribution"     : λ * |kd_loss| ==  0.01 * |task_loss|

Typical lambda ranges (from prior empirical runs):
  waveform_mse : O(1e3) - O(1e4)  to reach 1-10% contribution
  stft_l1      : O(1)  - O(50)
  binaural     : O(0.01) - O(0.3)  (BinauralLoss already at task_loss scale)

Notes:
  - Probes the student under no_grad; goal is loss magnitude comparison, not
    exact gradient direction.
  - Uses the same KDPrecomputedDataset the training pipeline does, to make
    sure (noisy, clean, teacher) triples line up identically.
  - Reproducible across re-runs by passing the same --seed.

Usage:
  CUDA_VISIBLE_DEVICES=3 python scripts/probe_kd_loss_scale.py \
    --ckpt /path/to/v4_or_kd.ckpt \
    --teacher_dir /raid/R12K41024/16_kd_heatmap_recovery/teacher_outputs/d1c3_epoch25_step100672_fp16 \
    --num_batches 20 --batch_size 8 \
    --hydra_overrides <ALL student-side overrides matching the ckpt>
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from hydra import compose, initialize


def load_student(ckpt_path: str, overrides):
    """Build the student model exactly like the trainer would, then load ckpt."""
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
        print(f"[probe] {len(res.missing_keys)} missing keys; "
              f"first: {res.missing_keys[:3]}")
    if res.unexpected_keys:
        print(f"[probe] {len(res.unexpected_keys)} unexpected keys; "
              f"first: {res.unexpected_keys[:3]}")
    return lm, cfg


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True, help="student ckpt path")
    ap.add_argument("--teacher_dir", required=True,
                    help="precomputed teacher output dir")
    ap.add_argument("--noisy_dir",
                    default="/raid/R12K41024/BCCTN/Dataset/noisy_trainset")
    ap.add_argument("--clean_dir",
                    default="/raid/R12K41024/BCCTN/Dataset/clean_trainset")
    ap.add_argument("--num_batches", type=int, default=20)
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--num_workers", type=int, default=2)
    ap.add_argument("--seed", type=int, default=12345)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--output_json", default=None,
                    help="optional path to dump full per-batch + agg stats")
    ap.add_argument("--hydra_overrides", nargs="*", default=[])
    args = ap.parse_args()

    if not os.path.isfile(args.ckpt):
        raise FileNotFoundError(args.ckpt)

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    print(f"[probe] loading student from {args.ckpt}")
    lm, _cfg = load_student(args.ckpt, args.hydra_overrides)
    lm.model.to(args.device).eval()
    lm.loss.to(args.device)

    from DCNN.datasets.kd_dataset import KDPrecomputedDataset
    from DCNN.utils.kd import (
        kd_loss_binaural,
        kd_loss_stft_l1,
        kd_loss_waveform_mse,
    )

    ds = KDPrecomputedDataset(
        noisy_dataset_dir=args.noisy_dir,
        target_dataset_dir=args.clean_dir,
        teacher_output_dir=args.teacher_dir,
        teacher_noisy_root=args.noisy_dir,
        mono=False,
    )
    loader = torch.utils.data.DataLoader(
        ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        drop_last=True,
    )

    per_batch = []
    print(f"[probe] running {args.num_batches} batches @ bs={args.batch_size}")
    with torch.no_grad():
        for i, batch in enumerate(loader):
            if i >= args.num_batches:
                break
            noisy, clean, teacher = batch
            noisy = noisy.to(args.device)
            clean = clean.to(args.device)
            # student forward
            out = lm.model(noisy)
            # cast teacher to student dtype + device at loss time
            teacher = teacher.to(out.device, dtype=out.dtype, non_blocking=True)

            task_l = lm.loss(out, clean)
            kd_wm = kd_loss_waveform_mse(out, teacher)
            kd_sl = kd_loss_stft_l1(out, teacher)
            # Binaural KD uses the same BinauralLoss instance; this is the
            # exact loss-shape the kd_mode="binaural" training branch will run.
            kd_bn = kd_loss_binaural(out, teacher, lm.loss)

            row = {
                "i": i,
                "task_loss": float(task_l.item()),
                "kd_waveform_mse": float(kd_wm.item()),
                "kd_stft_l1": float(kd_sl.item()),
                "kd_binaural": float(kd_bn.item()),
            }
            per_batch.append(row)
            print(f"  batch {i+1:2d}: task={row['task_loss']:.4f}  "
                  f"kd_wm={row['kd_waveform_mse']:.6e}  "
                  f"kd_sl={row['kd_stft_l1']:.6e}  "
                  f"kd_bn={row['kd_binaural']:.4f}")

    if not per_batch:
        raise RuntimeError("no batches processed; check loader / teacher_dir paths")

    arr_task = np.array([r["task_loss"] for r in per_batch])
    arr_task_abs = np.abs(arr_task)
    arr_kd_wm = np.array([r["kd_waveform_mse"] for r in per_batch])
    arr_kd_sl = np.array([r["kd_stft_l1"] for r in per_batch])
    arr_kd_bn = np.array([r["kd_binaural"] for r in per_batch])

    def stats(name, a):
        print(f"  {name:24s} mean={a.mean():.6e}  median={np.median(a):.6e}  "
              f"min={a.min():.6e}  max={a.max():.6e}")

    print("\n=== AGGREGATE STATS ===")
    stats("task_loss",        arr_task)
    stats("|task_loss|",      arr_task_abs)
    stats("kd_waveform_mse",  arr_kd_wm)
    stats("kd_stft_l1",       arr_kd_sl)
    stats("kd_binaural",      arr_kd_bn)

    print("\n=== LAMBDA RECOMMENDATIONS ===")
    print("  Using median |task_loss| and median |kd_loss|; \"contribution\" means")
    print("  how big λ·|kd_loss| is relative to |task_loss|.")
    print("  (kd_binaural is signed like task_loss; we use its abs for sizing.)")
    t = float(np.median(arr_task_abs))

    def recommend(kd_name, kd_arr):
        m = float(np.median(np.abs(kd_arr)))
        print(f"\n  --- {kd_name} ---")
        print(f"    median |task_loss|     = {t:.4e}")
        print(f"    median |{kd_name}|     = {m:.4e}")
        print(f"    λ for equal contribution (100%): {t/m:.4e}")
        print(f"    λ for 30% contribution        : {0.30 * t/m:.4e}")
        print(f"    λ for 10% contribution        : {0.10 * t/m:.4e}")
        print(f"    λ for  3% contribution        : {0.03 * t/m:.4e}")
        print(f"    λ for  1% contribution        : {0.01 * t/m:.4e}")

    recommend("kd_waveform_mse", arr_kd_wm)
    recommend("kd_stft_l1",      arr_kd_sl)
    recommend("kd_binaural",     arr_kd_bn)

    if args.output_json:
        os.makedirs(os.path.dirname(args.output_json) or ".", exist_ok=True)
        # Recompute medians for json payload to include binaural
        med_wm = float(np.median(np.abs(arr_kd_wm)))
        med_sl = float(np.median(np.abs(arr_kd_sl)))
        med_bn = float(np.median(np.abs(arr_kd_bn)))
        payload = {
            "ckpt": args.ckpt,
            "teacher_dir": args.teacher_dir,
            "num_batches": len(per_batch),
            "batch_size": args.batch_size,
            "per_batch": per_batch,
            "agg": {
                "task_loss_median": float(np.median(arr_task)),
                "task_loss_abs_median": t,
                "kd_waveform_mse_median": float(np.median(arr_kd_wm)),
                "kd_stft_l1_median": float(np.median(arr_kd_sl)),
                "kd_binaural_median": float(np.median(arr_kd_bn)),
                "kd_binaural_abs_median": med_bn,
            },
            "lambda_recommendations": {
                "kd_waveform_mse": {
                    "equal": t / med_wm,
                    "10pct": 0.1 * t / med_wm,
                    "3pct":  0.03 * t / med_wm,
                    "1pct":  0.01 * t / med_wm,
                },
                "kd_stft_l1": {
                    "equal": t / med_sl,
                    "10pct": 0.1 * t / med_sl,
                    "3pct":  0.03 * t / med_sl,
                    "1pct":  0.01 * t / med_sl,
                },
                "kd_binaural": {
                    "equal": t / med_bn,
                    "30pct": 0.30 * t / med_bn,
                    "10pct": 0.10 * t / med_bn,
                    "3pct":  0.03 * t / med_bn,
                    "1pct":  0.01 * t / med_bn,
                },
            },
        }
        with open(args.output_json, "w") as f:
            json.dump(payload, f, indent=2)
        print(f"\n[probe] wrote {args.output_json}")


if __name__ == "__main__":
    main()
