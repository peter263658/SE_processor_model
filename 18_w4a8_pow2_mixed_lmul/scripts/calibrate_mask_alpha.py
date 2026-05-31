"""Calibrate per-frequency-bin per-channel alpha for L-Mul gain bias.

Loads a ckpt, runs forward on N calibration samples (mid-SNR VCTK), computes:
    alpha[bin, channel] = median over samples of (|STFT(clean)| / |STFT(enhanced)|)
The median is robust against outliers (silent frames etc.). Saves to .npy.

Usage:
  CUDA_VISIBLE_DEVICES=3 python scripts/calibrate_mask_alpha.py \
      --ckpt /path/to/epoch=3-step=968.ckpt \
      --noisy_dir /raid/R12K41024/BCCTN/Dataset/noisy_testset \
      --clean_dir /raid/R12K41024/BCCTN/Dataset/clean_testset \
      --num_samples 30 \
      --snr_filter 0 3 \
      --output /raid/R12K41024/15_all_lmul_mixed_mult_eval/results/mask_alpha_probe/alpha.npy
"""
import argparse, os, sys
from pathlib import Path
import numpy as np
import soundfile as sf
import librosa
import torch
from omegaconf import OmegaConf
from hydra import compose, initialize

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))


def load_model(ckpt_path, overrides):
    # config_path is resolved relative to THIS file's location (scripts/).
    # Repo's config/ lives at scripts/../config.
    with initialize(config_path="../config", version_base="1.1"):
        cfg = compose(config_name="config", overrides=list(overrides or []))
    from DCNN.models.binaural_attention_model import BinauralAttentionDCNN
    from DCNN.trainer import DCNNLightningModule
    lm = DCNNLightningModule(cfg)
    sd = torch.load(ckpt_path, map_location="cpu", weights_only=False)["state_dict"]
    # Strip "model." prefix
    cleaned = {}
    for k, v in sd.items():
        if k.startswith("loss."):
            continue
        if k.startswith("model."):
            k = k[len("model."):]
        cleaned[k] = v
    lm.model.load_state_dict(cleaned, strict=False)
    lm.model.eval()
    return lm.model


def calibrate(model, noisy_dir, clean_dir, num_samples, snr_filter, n_fft, hop,
              device="cuda"):
    """Compute alpha[bin, channel] from a set of (clean, noisy) pairs."""
    noisy_root = Path(noisy_dir)
    clean_root = Path(clean_dir)
    # Walk noisy dir
    snr_dirs = []
    if snr_filter:
        for s in snr_filter:
            cand1 = noisy_root / f"snr_{s}dB"
            cand2 = noisy_root / str(s)
            if cand1.exists():
                snr_dirs.append(cand1)
            elif cand2.exists():
                snr_dirs.append(cand2)
    if not snr_dirs:
        snr_dirs = [p for p in noisy_root.rglob("*") if p.is_dir()] or [noisy_root]

    import re as _re
    samples = []
    for snr_dir in snr_dirs:
        for noisy_path in sorted(snr_dir.glob("*.wav")):
            # Noisy filename: <base>_snr{+,-,empty}X.Y.wav; Clean: <base>.wav
            clean_name = _re.sub(r"_snr[+\-]?[\d.]+\.wav$", ".wav", noisy_path.name)
            clean_path = clean_root / clean_name
            if not clean_path.exists():
                hits = list(clean_root.rglob(clean_name))
                if not hits:
                    continue
                clean_path = hits[0]
            samples.append((clean_path, noisy_path))
            if len(samples) >= num_samples:
                break
        if len(samples) >= num_samples:
            break

    print(f"[calibrate] using {len(samples)} samples from SNR dirs: "
          f"{[str(d.relative_to(noisy_root)) for d in snr_dirs[:3]]}")

    model = model.to(device).eval()
    ratios = []  # list of [F, 2] arrays
    with torch.no_grad():
        for i, (clean_path, noisy_path) in enumerate(samples):
            clean_data, sr = sf.read(clean_path)
            noisy_data, _ = sf.read(noisy_path)
            if clean_data.ndim == 1 or noisy_data.ndim == 1:
                continue
            noisy_t = torch.tensor(noisy_data.T, dtype=torch.float32).unsqueeze(0).to(device)
            enh = model(noisy_t).cpu().numpy()[0].astype(np.float32)  # [2, T]
            # Align lengths
            T = min(clean_data.shape[0], enh.shape[1])
            clean_data = clean_data[:T]
            enh = enh[:, :T]
            # Compute per-bin magnitude ratio for each channel
            per_bin = np.zeros((n_fft // 2 + 1, 2), dtype=np.float64)
            valid = np.zeros(2, dtype=np.int64)
            for ch in range(2):
                S_clean = librosa.stft(np.ascontiguousarray(clean_data[:, ch].astype(np.float32)),
                                       n_fft=n_fft, hop_length=hop)
                S_enh = librosa.stft(enh[ch], n_fft=n_fft, hop_length=hop)
                num = np.abs(S_clean).mean(axis=1)  # [F]
                den = np.abs(S_enh).mean(axis=1)    # [F]
                mask = den > 1e-8
                # Per-bin ratio; bins where enhanced is silent → keep ratio as 1
                r = np.ones_like(num)
                r[mask] = num[mask] / den[mask]
                per_bin[:, ch] = r
                valid[ch] = mask.sum()
            ratios.append(per_bin)
            if (i + 1) % 10 == 0:
                print(f"[calibrate] processed {i+1}/{len(samples)}")

    if not ratios:
        raise RuntimeError("no samples processed")

    arr = np.stack(ratios, axis=0)  # [N, F, 2]
    alpha_median = np.median(arr, axis=0)  # [F, 2], robust
    # Clip insane values
    alpha_median = np.clip(alpha_median, 0.5, 2.0)
    print(f"[calibrate] alpha shape={alpha_median.shape}, "
          f"L: min={alpha_median[:,0].min():.3f} max={alpha_median[:,0].max():.3f} med={np.median(alpha_median[:,0]):.3f}, "
          f"R: min={alpha_median[:,1].min():.3f} max={alpha_median[:,1].max():.3f} med={np.median(alpha_median[:,1]):.3f}")
    return alpha_median.astype(np.float32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--noisy_dir", required=True)
    ap.add_argument("--clean_dir", required=True)
    ap.add_argument("--num_samples", type=int, default=30)
    ap.add_argument("--snr_filter", nargs="*", default=["0", "3"])
    ap.add_argument("--n_fft", type=int, default=128)  # matches model's win_len
    ap.add_argument("--hop", type=int, default=80)     # matches model's win_inc
    ap.add_argument("--output", required=True)
    ap.add_argument("--hydra_overrides", nargs="*", default=[])
    args = ap.parse_args()

    model = load_model(args.ckpt, args.hydra_overrides)
    alpha = calibrate(model, args.noisy_dir, args.clean_dir,
                      args.num_samples, args.snr_filter,
                      args.n_fft, args.hop)
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    np.save(args.output, alpha)
    print(f"[calibrate] saved alpha -> {args.output}  shape={alpha.shape}  dtype={alpha.dtype}")


if __name__ == "__main__":
    main()
