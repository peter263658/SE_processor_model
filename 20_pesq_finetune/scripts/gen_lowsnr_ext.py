#!/usr/bin/env python
"""Extend the binaural VCTK/TIMIT test sets with -8/-10 dB SNR points.

Method: the generator (BCCTN/prepare_data.py) renders noisy = clean + noise * k
with k = sqrt(P_speech / (P_noise * 10^(SNR/10))) and NO post-normalization, so
the noise instance of an existing SNR dir can be recovered as
    noise_ref = noisy_ref - clean
and rescaled exactly to a new target SNR T:
    noisy_T = clean + noise_ref * 10^((SNR_ref - T) / 20)
This reuses the same utterance crop, azimuth/HRIR render, and noise draw as the
reference SNR condition (we use -6 dB, the closest one), so the new points are
methodologically identical to the shipped test set. Only 16-bit quantization of
the stored reference wavs (~-90 dB) separates this from a native generator run.

Outputs PCM_16 wavs (same as the original pipeline). Clipping is counted and
reported; the original pipeline had the same clipping exposure at low SNR.
"""

import argparse
from pathlib import Path

import numpy as np
import soundfile as sf

REF_SNR = -6.0


def synth_dataset(noisy_ref_dir: Path, clean_dir: Path, out_root: Path,
                  targets=(-8.0, -10.0)):
    ref_files = sorted(noisy_ref_dir.glob("*.wav"))
    if not ref_files:
        raise SystemExit(f"no wavs in {noisy_ref_dir}")
    stats = {t: {"files": 0, "clipped_samples": 0, "total_samples": 0} for t in targets}
    missing_clean = 0

    for t in targets:
        (out_root / f"snr_{int(t)}dB").mkdir(parents=True, exist_ok=True)

    for nf in ref_files:
        # p225_p225_009_az-40_snr-6.0.wav -> clean p225_p225_009_az-40.wav
        stem = nf.stem
        base = stem.rsplit("_snr", 1)[0]
        cf = clean_dir / f"{base}.wav"
        if not cf.exists():
            missing_clean += 1
            continue
        noisy, sr_n = sf.read(nf, dtype="float64", always_2d=True)
        clean, sr_c = sf.read(cf, dtype="float64", always_2d=True)
        assert sr_n == sr_c, f"sr mismatch {nf}"
        n = min(len(noisy), len(clean))
        noisy, clean = noisy[:n], clean[:n]
        noise = noisy - clean

        for t in targets:
            gain = 10.0 ** ((REF_SNR - t) / 20.0)
            out = clean + noise * gain
            clipped = int(np.sum(np.abs(out) > 1.0))
            out = np.clip(out, -1.0, 1.0)
            name = f"{base}_snr{t:+.1f}.wav"
            sf.write(out_root / f"snr_{int(t)}dB" / name, out, sr_n, subtype="PCM_16")
            st = stats[t]
            st["files"] += 1
            st["clipped_samples"] += clipped
            st["total_samples"] += out.size

    for t, st in stats.items():
        pct = 100.0 * st["clipped_samples"] / max(st["total_samples"], 1)
        print(f"  snr {t:+.0f} dB: {st['files']} files, clipped {st['clipped_samples']} "
              f"samples ({pct:.4f}%)")
    if missing_clean:
        print(f"  WARNING: {missing_clean} noisy files had no matching clean file")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_root", default="/raid/R12K41024/20_pesq_finetune/data/lowsnr_ext")
    args = ap.parse_args()
    out_root = Path(args.out_root)

    sets = {
        "vctk": (Path("/raid/R12K41024/BCCTN/Dataset/noisy_testset/snr_-6dB"),
                 Path("/raid/R12K41024/BCCTN/Dataset/clean_testset")),
        "timit": (Path("/raid/R12K41024/BCCTN/Dataset/noisy_testset_timit/snr_-6dB"),
                  Path("/raid/R12K41024/BCCTN/Dataset/clean_testset_timit")),
    }
    for name, (noisy_dir, clean_dir) in sets.items():
        print(f"[{name}] ref={noisy_dir}")
        synth_dataset(noisy_dir, clean_dir, out_root / name)


if __name__ == "__main__":
    main()
