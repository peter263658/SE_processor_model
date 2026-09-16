#!/usr/bin/env python
"""
Pre-render VoiceBank+DEMAND (mono) to BINAURAL 2-channel wavs via an HRIR, ONCE,
so eval just loads the 2ch files instead of re-convolving every run.

No re-mix: the EXISTING standard clean + premixed-noisy mono files are each
convolved through the same HRIR (first 2 ch = L/R in-ear). Per-file amplitude
normalization is intentionally NOT applied here -- it stays in the eval so the
prerendered files match the on-the-fly binaural_az0 path bit-for-bit.

  python gen_vb_binaural.py --clean_dir <mono clean> --noisy_dir <mono noisy> \
      --hrir <az wav> --out_dir <dest>
"""
import argparse, glob, os
import numpy as np
import librosa
import soundfile as sf
import scipy.signal as sps

SR = 16000


def load_hrir(path, sr=SR):
    a, s = librosa.load(path, sr=None, mono=False)  # [ch, n]; ch0=L, ch1=R in-ear
    if s != sr:
        a = np.stack([librosa.resample(a[c], orig_sr=s, target_sr=sr) for c in range(a.shape[0])])
    return a[0].astype(np.float32), a[1].astype(np.float32)


def render(mono, hl, hr):
    l = sps.convolve(mono, hl)
    r = sps.convolve(mono, hr)
    n = min(len(l), len(r))
    return np.stack([l[:n], r[:n]]).astype(np.float32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--clean_dir", required=True)
    ap.add_argument("--noisy_dir", required=True)
    ap.add_argument("--hrir", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--clean_subdir", default="clean_testset_wav")
    ap.add_argument("--noisy_subdir", default="noisy_testset_wav")
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    out_clean = os.path.join(args.out_dir, args.clean_subdir)
    out_noisy = os.path.join(args.out_dir, args.noisy_subdir)
    os.makedirs(out_clean, exist_ok=True)
    os.makedirs(out_noisy, exist_ok=True)

    hl, hr = load_hrir(args.hrir)
    print(f"HRIR {args.hrir}  L/R taps={len(hl)}/{len(hr)}")

    clean_files = sorted(glob.glob(os.path.join(args.clean_dir, "*.wav")))
    if args.limit:
        clean_files = clean_files[:args.limit]
    print(f"{len(clean_files)} clean files")

    n_done = 0
    for i, cf in enumerate(clean_files):
        name = os.path.basename(cf)
        nf = os.path.join(args.noisy_dir, name)
        if not os.path.exists(nf):
            continue
        clean_m, _ = librosa.load(cf, sr=SR, mono=True)
        noisy_m, _ = librosa.load(nf, sr=SR, mono=True)
        n = min(len(clean_m), len(noisy_m))
        clean_m, noisy_m = clean_m[:n], noisy_m[:n]
        cb = render(clean_m, hl, hr)
        nb = render(noisy_m, hl, hr)
        m = min(cb.shape[1], nb.shape[1])
        cb, nb = cb[:, :m], nb[:, :m]
        # soundfile expects [n, ch]. FLOAT subtype = lossless 32-bit float:
        # 16-bit PCM (the .wav default) would quantize AND clip the post-render
        # samples (HRIR convolution overshoots +/-1), making prerendered != the
        # float32 on-the-fly path. FLOAT keeps them numerically identical.
        sf.write(os.path.join(out_clean, name), cb.T, SR, subtype="FLOAT")
        sf.write(os.path.join(out_noisy, name), nb.T, SR, subtype="FLOAT")
        n_done += 1
        if (i + 1) % 200 == 0:
            print(f"  {i + 1}/{len(clean_files)}", flush=True)

    # tiny manifest for traceability
    with open(os.path.join(args.out_dir, "RENDER_INFO.txt"), "w") as f:
        f.write(f"hrir={args.hrir}\nsrc_clean={args.clean_dir}\nsrc_noisy={args.noisy_dir}\n"
                f"files={n_done}\nsr={SR}\nnote=2ch binaural render, no normalization, no remix\n")
    print(f"done: {n_done} pairs -> {args.out_dir}")


if __name__ == "__main__":
    main()
