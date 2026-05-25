"""
Pre-compute teacher model outputs for the entire training dataset.

Run once; reuse for any number of KD training runs. The savings vs live teacher
forward in the training loop are roughly:
   live-teacher KD wall-clock ≈ 2x student-only training (teacher forward ~= student forward)
   precompute KD wall-clock ≈ student-only + a one-time precompute cost

Output:
  <output_dir>/<noisy_basename>.pt    # fp16 tensor of shape [2, T]
  <output_dir>/_meta.json             # summary (teacher ckpt path, hash, sample count)

A KD-aware dataloader should key on `Path(noisy_path).relative_to(noisy_root)`
with a `.pt` suffix to find the corresponding teacher output. Determinism:
BCCTN training dataloader yields deterministic (clean, noisy) pairs by
sorted-name index, no on-the-fly aug, so this 1:1 keyed mapping is stable.

Usage (sample 100 files for smoke test):
  CUDA_VISIBLE_DEVICES=3 python scripts/precompute_teacher_outputs.py \
    --teacher_ckpt /raid/R12K41024/14_low_power_arch_qat/.../epoch=25-step=100672.ckpt \
    --output_dir /raid/R12K41024/16_kd_heatmap_recovery/teacher_outputs/d1c3 \
    --limit 100 \
    --hydra_overrides ...

Full run: drop --limit and let it churn through ~30k samples (~2-3h on H100).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path

import torch
import torchaudio
from tqdm import tqdm

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from hydra import compose, initialize


def _ckpt_hash(path: str) -> str:
    """SHA1 prefix of the ckpt's bytes, for provenance metadata."""
    h = hashlib.sha1()
    with open(path, "rb") as f:
        # Cap at first 8 MB for speed; that's enough to fingerprint a Lightning ckpt header.
        h.update(f.read(8 * 1024 * 1024))
    return h.hexdigest()[:16]


def load_teacher(teacher_overrides, ckpt_path, device):
    """Build the teacher model via Hydra compose + ckpt load. Frozen, eval mode."""
    # config_path is resolved relative to THIS file (scripts/), repo config is one level up.
    with initialize(config_path="../config", version_base="1.1"):
        cfg = compose(config_name="config", overrides=list(teacher_overrides))
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
        print(f"[precompute] {len(res.missing_keys)} missing keys; "
              f"first few: {res.missing_keys[:3]}")
    if res.unexpected_keys:
        print(f"[precompute] {len(res.unexpected_keys)} unexpected keys; "
              f"first few: {res.unexpected_keys[:3]}")
    lm.model.eval()
    for p in lm.model.parameters():
        p.requires_grad_(False)
    lm.model.to(device)
    return lm.model


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--teacher_ckpt", required=True)
    ap.add_argument("--noisy_dir", required=True,
                    help="root dir of noisy training wavs (e.g., noisy_trainset)")
    ap.add_argument("--clean_dir", default=None,
                    help="root dir of clean wavs (for metadata only; not used for forward)")
    ap.add_argument("--output_dir", required=True,
                    help="<output_dir>/<rel_path_to_noisy>.pt files. "
                         "Subdirectory structure is preserved to avoid same-basename collisions.")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--dtype", default="fp16", choices=["fp16", "fp32"])
    ap.add_argument("--limit", type=int, default=0,
                    help="0 = all files; >0 = process only first N (for smoke test). "
                         "Default 0 = full run; recommend --limit 32 first to sanity-check.")
    ap.add_argument("--skip_existing", action=argparse.BooleanOptionalAction, default=True,
                    help="when set, skip files whose .pt already exists (resume-friendly)")
    ap.add_argument("--hydra_overrides", nargs="*", default=[],
                    help="full Hydra overrides for the TEACHER architecture, "
                         "e.g. d1+c3 (real-mult conv, packet+L-Mul attention)")
    args = ap.parse_args()

    if not os.path.isfile(args.teacher_ckpt):
        raise FileNotFoundError(f"teacher ckpt not found: {args.teacher_ckpt}")

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    save_dtype = torch.float16 if args.dtype == "fp16" else torch.float32

    print(f"[precompute] teacher ckpt: {args.teacher_ckpt}")
    teacher_hash = _ckpt_hash(args.teacher_ckpt)
    print(f"[precompute] teacher fingerprint (sha1[:16]): {teacher_hash}")
    print(f"[precompute] noisy dir: {args.noisy_dir}")
    print(f"[precompute] output dir: {out_dir}")
    print(f"[precompute] save dtype: {save_dtype}")

    teacher = load_teacher(args.hydra_overrides, args.teacher_ckpt, args.device)

    noisy_root = Path(args.noisy_dir)
    noisy_paths = sorted(noisy_root.rglob("*.wav"))
    if args.limit > 0:
        noisy_paths = noisy_paths[:args.limit]
    print(f"[precompute] {len(noisy_paths)} noisy files to process")

    skipped = 0
    written = 0
    errored = 0
    start = time.time()
    first_output_shape = None
    first_sr = None

    with torch.no_grad():
        for noisy_path in tqdm(noisy_paths, desc="teacher forward"):
            # Preserve relative path so e.g. "snr_-3dB/p225_001.wav" and
            # "snr_3dB/p225_001.wav" don't collide (the trainset happens to be
            # flat right now but other datasets may not be).
            rel = noisy_path.relative_to(noisy_root)
            out_path = out_dir / rel.with_suffix(".pt")
            if args.skip_existing and out_path.exists():
                skipped += 1
                continue
            out_path.parent.mkdir(parents=True, exist_ok=True)
            try:
                noisy_signal, sr = torchaudio.load(str(noisy_path))  # [2, T] float32
                if noisy_signal.dim() == 1 or noisy_signal.shape[0] < 2:
                    print(f"[precompute] skip non-stereo: {rel}")
                    continue
                x = noisy_signal.unsqueeze(0).to(args.device)  # [1, 2, T]
                y = teacher(x)                                  # [1, 2, T]
                y = y.squeeze(0).to(save_dtype).cpu()           # [2, T]
                if first_output_shape is None:
                    first_output_shape = tuple(y.shape)
                    first_sr = sr
                # Atomic write: write to .tmp then rename, so a crashed run
                # doesn't leave partially written files that --skip_existing
                # would later mistake for completed.
                tmp_path = out_path.with_suffix(".tmp.pt")
                torch.save(y, tmp_path)
                tmp_path.replace(out_path)
                written += 1
            except Exception as e:
                errored += 1
                print(f"[precompute] error on {rel}: {e}")

    elapsed = time.time() - start

    import datetime
    script_path = Path(__file__).resolve()
    script_mtime = datetime.datetime.fromtimestamp(script_path.stat().st_mtime).isoformat()
    meta_path = out_dir / "_meta.json"
    meta = {
        "teacher_ckpt": args.teacher_ckpt,
        "teacher_sha1_prefix": teacher_hash,
        "noisy_dir": str(args.noisy_dir),
        "clean_dir": str(args.clean_dir) if args.clean_dir else None,
        "output_dir": str(out_dir),
        "save_dtype": str(save_dtype),
        "first_output_shape": list(first_output_shape) if first_output_shape else None,
        "first_sample_rate": first_sr,
        "hydra_overrides": list(args.hydra_overrides),
        "n_processed_this_run": written,
        "n_skipped_existing": skipped,
        "n_errored": errored,
        "n_total_noisy_files": len(noisy_paths),
        "elapsed_seconds": round(elapsed, 1),
        "rel_path_preserved": True,
        "script_path": str(script_path),
        "script_mtime": script_mtime,
        "completed_at": datetime.datetime.now().isoformat(),
    }
    # If meta already exists (resume after a smoke test), preserve prior runs
    # but keep the top-level counters describing the most recent run.
    if meta_path.exists():
        try:
            existing = json.load(open(meta_path))
            history = existing.get("history", [])
            prior = {k: existing.get(k) for k in
                     ("completed_at", "n_processed_this_run", "n_skipped_existing",
                      "n_errored", "elapsed_seconds", "n_total_noisy_files")}
            history.append(prior)
            meta["history"] = history
            meta["n_files_present"] = len(list(out_dir.rglob("*.pt")))
            with open(meta_path, "w") as f:
                json.dump(meta, f, indent=2)
        except Exception as e:
            print(f"[precompute] could not merge existing meta: {e}")
            with open(meta_path, "w") as f:
                json.dump(meta, f, indent=2)
    else:
        with open(meta_path, "w") as f:
            json.dump(meta, f, indent=2)

    print(f"\n[precompute] done. "
          f"written={written}, skipped(existing)={skipped}, errored={errored}, "
          f"elapsed={elapsed:.1f}s, dir={out_dir}")
    print(f"[precompute] meta written to {meta_path}")


if __name__ == "__main__":
    main()
