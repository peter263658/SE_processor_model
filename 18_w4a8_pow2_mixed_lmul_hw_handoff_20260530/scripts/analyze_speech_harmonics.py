#!/usr/bin/env python
"""
Offline analysis over the enhanced-waveform caches written by
speech_harmonic_probe.py.  Adds the two things the per-mode runs cannot give:

  1. the NOISY-input baseline for the same harmonic metrics, so
     "enhanced vs clean" can be read against "noisy vs clean" -- the only way
     an absolute dB number means anything;
  2. the direct fp32 <-> hw waveform difference, i.e. the quantization
     noise-to-signal ratio of the deployed zero-multiplier datapath.

No model is loaded; this runs on CPU from the cached .npy files.
"""

import argparse
import json
import os

import numpy as np
import soundfile as sf

FS = 16000
N_FFT = 1024
HOP = 256
FMIN, FMAX = 65.0, 400.0
FLOOR_DB = 40.0


def stft_mag(x):
    import librosa
    return np.abs(librosa.stft(np.ascontiguousarray(x, dtype=np.float64),
                               n_fft=N_FFT, hop_length=HOP,
                               window="hann", center=True))


def f0_track(clean):
    import librosa
    f0, voiced, _ = librosa.pyin(np.ascontiguousarray(clean, dtype=np.float64),
                                 fmin=FMIN, fmax=FMAX, sr=FS,
                                 frame_length=N_FFT, hop_length=HOP)
    return f0, voiced


def harmonic_mask(f0_t, n_bins, tol_bins=1):
    df = FS / N_FFT
    m = np.zeros(n_bins, dtype=bool)
    k = 1
    while k * f0_t < FS / 2:
        c = int(round(k * f0_t / df))
        if c < n_bins:
            m[max(0, c - tol_bins):min(n_bins, c + tol_bins + 1)] = True
        k += 1
    return m


def compare(clean, test, f0, voiced):
    """Harmonic-grid comparison of `test` against `clean` (both 1-D)."""
    d0 = float(np.dot(clean, clean))
    a = float(np.dot(test, clean)) / d0 if d0 > 0 else 1.0
    test = test * a

    Mc, Mt = stft_mag(clean), stft_mag(test)
    n_fr = min(len(f0), Mc.shape[1], Mt.shape[1])
    n_bins = Mc.shape[0]

    err, werr, bias, ihr, her = [], [], [], [], []
    for t in range(n_fr):
        if not voiced[t] or not np.isfinite(f0[t]):
            continue
        m = harmonic_mask(f0[t], n_bins)
        if not m.any():
            continue
        c_h, t_h = Mc[m, t], Mt[m, t]
        keep = c_h > (10 ** (-FLOOR_DB / 20.0)) * max(c_h.max(), 1e-12)
        if not keep.any():
            continue
        d = 20 * np.log10((t_h[keep] + 1e-12) / (c_h[keep] + 1e-12))
        w = c_h[keep] ** 2
        err.append(float(np.mean(np.abs(d))))
        werr.append(float(np.sum(w * np.abs(d)) / np.sum(w)))
        bias.append(float(np.mean(d)))
        eh = float(np.sum(Mt[m, t] ** 2))
        et = float(np.sum(Mt[:, t] ** 2))
        ihr.append(10 * np.log10((max(et - eh, 0.0) + 1e-30) / (eh + 1e-30)))
        her.append(eh / max(et, 1e-30))
    if not err:
        return None
    return {"harm_err_db": float(np.mean(err)),
            "harm_err_w_db": float(np.mean(werr)),
            "harm_bias_db": float(np.mean(bias)),
            "ihr_db": float(np.mean(ihr)),
            "her": float(np.mean(her)),
            "align_db": 20 * float(np.log10(abs(a) + 1e-12))}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fp32_dir", default="results/thd/enh_fp32")
    ap.add_argument("--hw_dir", default="results/thd/enh_hw")
    ap.add_argument("--noisy_dir", default="/raid/R12K41024/BCCTN/Dataset/noisy_testset/snr_0dB")
    ap.add_argument("--clean_dir", default="/raid/R12K41024/BCCTN/Dataset/clean_testset")
    ap.add_argument("--out", default="results/thd/speech_summary.json")
    args = ap.parse_args()

    stems = sorted(os.path.splitext(f)[0] for f in os.listdir(args.fp32_dir)
                   if f.endswith(".npy"))
    rows, qn = [], []
    for i, stem in enumerate(stems):
        fp32 = np.load(os.path.join(args.fp32_dir, stem + ".npy"))
        hwp = os.path.join(args.hw_dir, stem + ".npy")
        if not os.path.exists(hwp):
            continue
        hw = np.load(hwp)
        clean, _ = sf.read(os.path.join(args.clean_dir, stem + ".wav"))
        nz = [f for f in os.listdir(args.noisy_dir) if f.startswith(stem + "_snr")]
        noisy, _ = sf.read(os.path.join(args.noisy_dir, nz[0]))
        n = min(len(clean), len(noisy), fp32.shape[1], hw.shape[1])
        clean, noisy, fp32, hw = clean[:n], noisy[:n], fp32[:, :n], hw[:, :n]

        # quantization noise-to-signal, hw relative to fp32 (identical weights)
        qn.append(10 * np.log10(
            (np.sum((hw - fp32.astype(np.float64)) ** 2) + 1e-30) /
            (np.sum(fp32.astype(np.float64) ** 2) + 1e-30)))

        rec = {"file": stem}
        ok = True
        for ch in range(2):
            f0, voiced = f0_track(clean[:, ch])
            for tag, sig in (("noisy", noisy[:, ch]),
                             ("fp32", fp32[ch]), ("hw", hw[ch])):
                st = compare(clean[:, ch], np.asarray(sig, dtype=np.float64), f0, voiced)
                if st is None:
                    ok = False
                    break
                for k, v in st.items():
                    rec.setdefault(f"{tag}_{k}", []).append(v)
            if not ok:
                break
        if not ok:
            continue
        rec = {k: (float(np.mean(v)) if isinstance(v, list) else v)
               for k, v in rec.items()}
        rows.append(rec)
        print(f"{i+1}/{len(stems)} {stem}: harm_err noisy/fp32/hw = "
              f"{rec['noisy_harm_err_db']:.2f}/{rec['fp32_harm_err_db']:.2f}/"
              f"{rec['hw_harm_err_db']:.2f} dB", flush=True)

    keys = [k for k in rows[0] if k != "file"]
    summary = {k: float(np.mean([r[k] for r in rows])) for k in keys}
    summary["quant_noise_to_signal_db"] = float(np.mean(qn))
    with open(args.out, "w") as fh:
        json.dump({"n_files": len(rows), "summary": summary, "per_file": rows}, fh, indent=2)

    print(f"\n=== harmonic fidelity, {len(rows)} files, VCTK 0 dB SNR ===")
    hdr = f"{'metric':22s} {'noisy':>10s} {'fp32':>10s} {'hw':>10s}"
    print(hdr); print("-" * len(hdr))
    for m in ("harm_err_db", "harm_err_w_db", "harm_bias_db", "ihr_db", "her"):
        print(f"{m:22s} " + " ".join(f"{summary[f'{t}_{m}']:10.3f}"
                                     for t in ("noisy", "fp32", "hw")))
    print(f"\nquantization noise-to-signal (hw vs fp32): "
          f"{summary['quant_noise_to_signal_db']:.2f} dB")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
