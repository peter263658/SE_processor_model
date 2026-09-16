#!/usr/bin/env python
"""
Speech harmonic-fidelity probe (companion to thd_probe.py).

Pure-tone THD answers the ANSI S3.22 / IEC 60118-7 style question.  This script
answers the *speech* version of the same worry -- "are the pitch harmonics
damaged, or is extra energy appearing between them?" -- by comparing the
enhanced output against the CLEAN binaural reference on the harmonic grid of
the talker's own F0 track.

Per voiced frame (n_fft=1024, hop=256, Hann; F0 from librosa.pyin on clean):
  harm_err_db  : mean |20log10(|X_enh|/|X_clean|)| on bins k*F0  (harmonic damage)
  harm_bias_db : signed version of the same (systematic over/under-shoot)
  ihr_db       : 10log10(inter-harmonic energy / harmonic energy)
                 -- how much energy sits BETWEEN the harmonics.  Reported for
                 clean / noisy / enhanced; enhanced should sit near clean and
                 well below noisy if harmonics survive and noise is removed.
  her          : harmonic energy / total energy in the frame

Enhanced waveforms are also cached as .npy so fp32 and hw runs can be diffed
directly (quantization noise-to-signal ratio).

Usage:
  python speech_harmonic_probe.py --mode fp32 --n_files 30 --out results/thd/speech_fp32.json
"""

import argparse
import json
import os
import sys

import numpy as np
import soundfile as sf
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from thd_probe import (FS, PKG, FP32_OVERRIDES, HW_OVERRIDES, build_model,  # noqa: E402
                       run_model)

N_FFT = 1024
HOP = 256
FMIN, FMAX = 65.0, 400.0


def stft_mag(x):
    import librosa
    return np.abs(librosa.stft(x.astype(np.float64), n_fft=N_FFT,
                               hop_length=HOP, window="hann", center=True))


def harmonic_stats(clean, noisy, enh, tol_bins=1, floor_db=40.0):
    """All inputs are 1-D, same length.  Returns per-file averaged stats.

    The enhancer has no absolute-level guarantee, so `enh` is first aligned to
    `clean` by the least-squares scalar gain; otherwise a constant broadband
    offset would masquerade as harmonic damage.  Harmonic bins more than
    `floor_db` below the strongest harmonic of the frame are excluded -- their
    log-ratio is dominated by residual noise, not by harmonic fidelity.
    """
    import librosa

    denom = float(np.dot(clean, clean))
    align = float(np.dot(enh, clean)) / denom if denom > 0 else 1.0
    align_db = 20 * np.log10(abs(align) + 1e-12)
    enh = enh * align

    f0, voiced, _ = librosa.pyin(clean.astype(np.float64), fmin=FMIN, fmax=FMAX,
                                 sr=FS, frame_length=N_FFT, hop_length=HOP)
    Mc, Mn, Me = stft_mag(clean), stft_mag(noisy), stft_mag(enh)
    n_fr = min(len(f0), Mc.shape[1], Mn.shape[1], Me.shape[1])
    df = FS / N_FFT
    n_bins = Mc.shape[0]

    acc = {k: [] for k in ("harm_err_db", "harm_bias_db", "gain_align_db", "her_clean", "her_enh",
                           "ihr_clean_db", "ihr_noisy_db", "ihr_enh_db")}
    for t in range(n_fr):
        if not voiced[t] or not np.isfinite(f0[t]):
            continue
        mask = np.zeros(n_bins, dtype=bool)
        k = 1
        while k * f0[t] < FS / 2:
            c = int(round(k * f0[t] / df))
            if c < n_bins:
                mask[max(0, c - tol_bins):min(n_bins, c + tol_bins + 1)] = True
            k += 1
        if mask.sum() == 0:
            continue
        c_h, e_h = Mc[mask, t], Me[mask, t]
        keep = c_h > (10 ** (-floor_db / 20.0)) * max(c_h.max(), 1e-12)
        if keep.sum() == 0:
            continue
        d = 20 * np.log10((e_h[keep] + 1e-12) / (c_h[keep] + 1e-12))
        acc["harm_err_db"].append(np.mean(np.abs(d)))
        acc["harm_bias_db"].append(np.mean(d))
        acc["gain_align_db"].append(align_db)

        def split(M):
            eh = float(np.sum(M[mask, t] ** 2))
            et = float(np.sum(M[:, t] ** 2))
            return eh, max(et - eh, 0.0)

        for tag, M in (("clean", Mc), ("noisy", Mn), ("enh", Me)):
            eh, ei = split(M)
            if tag in ("clean", "enh"):
                acc[f"her_{tag}"].append(eh / max(eh + ei, 1e-30))
            acc[f"ihr_{tag}_db"].append(10 * np.log10((ei + 1e-30) / (eh + 1e-30)))

    if not acc["harm_err_db"]:
        return None
    return {k: float(np.mean(v)) for k, v in acc.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", required=True, choices=["fp32", "hw"])
    ap.add_argument("--ckpt", default=os.path.join(
        PKG, "ckpt", "strict_core_pow2_blockfloat_deploy_ep3_vctk0.8357_timit0.7898.ckpt"))
    ap.add_argument("--noisy_dir", default="/raid/R12K41024/BCCTN/Dataset/noisy_testset/snr_0dB")
    ap.add_argument("--clean_dir", default="/raid/R12K41024/BCCTN/Dataset/clean_testset")
    ap.add_argument("--n_files", type=int, default=30)
    ap.add_argument("--cache_dir", default=None)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(args.ckpt,
                        FP32_OVERRIDES if args.mode == "fp32" else HW_OVERRIDES,
                        device)

    files = sorted(os.listdir(args.noisy_dir))[:args.n_files]
    cache = args.cache_dir or os.path.join(os.path.dirname(os.path.abspath(args.out)),
                                           f"enh_{args.mode}")
    os.makedirs(cache, exist_ok=True)

    per_file = []
    for i, fn in enumerate(files):
        stem = fn.rsplit("_snr", 1)[0]
        cpath = os.path.join(args.clean_dir, stem + ".wav")
        if not os.path.exists(cpath):
            continue
        noisy, sr_n = sf.read(os.path.join(args.noisy_dir, fn))
        clean, sr_c = sf.read(cpath)
        if noisy.ndim == 1 or clean.ndim == 1 or sr_n != FS or sr_c != FS:
            continue
        n = min(len(noisy), len(clean))
        noisy, clean = noisy[:n], clean[:n]
        enh = run_model(model, np.ascontiguousarray(noisy.T, dtype=np.float32), device)
        np.save(os.path.join(cache, stem + ".npy"), enh.astype(np.float32))

        chan = []
        for ch in range(2):
            st = harmonic_stats(clean[:, ch], noisy[:, ch], enh[ch])
            if st:
                chan.append(st)
        if not chan:
            continue
        rec = {"file": stem}
        rec.update({k: float(np.mean([c[k] for c in chan])) for k in chan[0]})
        per_file.append(rec)
        print(f"[{args.mode}] {i+1}/{len(files)} {stem}  "
              f"harm_err={rec['harm_err_db']:.2f} dB  "
              f"IHR clean/noisy/enh = {rec['ihr_clean_db']:.1f}/"
              f"{rec['ihr_noisy_db']:.1f}/{rec['ihr_enh_db']:.1f} dB", flush=True)

    keys = [k for k in per_file[0] if k != "file"]
    summary = {k: float(np.mean([r[k] for r in per_file])) for k in keys}
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w") as fh:
        json.dump({"mode": args.mode, "n_files": len(per_file),
                   "cache_dir": cache, "summary": summary,
                   "per_file": per_file}, fh, indent=2)
    print(f"\n[{args.mode}] SUMMARY over {len(per_file)} files:")
    for k in keys:
        print(f"    {k:16s} = {summary[k]:+.4f}")
    print(f"[{args.mode}] wrote {args.out}")


if __name__ == "__main__":
    main()
