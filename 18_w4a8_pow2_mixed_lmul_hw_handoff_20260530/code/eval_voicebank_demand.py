#!/usr/bin/env python
"""
VoiceBank+DEMAND zero-shot eval for the binaural BCCTN deploy model.

The standard VB+DEMAND test set is MONO. The model takes 2-channel binaural
input, so each mono file (clean and noisy) is binaural-rendered through the
anechoic HRIR at az=0 (frontal single source), then metrics are reported
per-ear (L/R averaged). Reports PESQ (nb + wb), STOI, ESTOI, SI-SDR for both
the noisy baseline and the enhanced output, plus the improvement delta.

Model is constructed with the SAME hydra overrides + env vars as the deploy
eval (run_calibrated_e4m3_activation_eval.sh -> run_apot_descale_eval.sh), so
this scores the true deployed datapath (calibrated_pow2_e4m3 + delayed APoT
descale + outlier L-Mul m4 + qkv g4 + BN fold + PReLU pow2).
"""
import os, sys, argparse, glob, csv
import numpy as np
import soundfile as sf
import librosa
import scipy.signal as sps
import torch

SR = 16000


def load_hrir_az0(path, sr=SR):
    a, s = librosa.load(path, sr=None, mono=False)  # [ch, n]; ch0=L in-ear, ch1=R in-ear
    if s != sr:
        a = np.stack([librosa.resample(a[c], orig_sr=s, target_sr=sr) for c in range(a.shape[0])])
    return a[0].astype(np.float32), a[1].astype(np.float32)


def render_binaural(mono, hl, hr):
    l = sps.convolve(mono, hl)
    r = sps.convolve(mono, hr)
    n = min(len(l), len(r))
    return np.stack([l[:n], r[:n]]).astype(np.float32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--clean_dir", required=True)
    ap.add_argument("--noisy_dir", required=True)
    ap.add_argument("--hrir", required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--input_mode", default="binaural_az0",
                    choices=["binaural_az0", "mono_dup"],
                    help="binaural_az0: HRIR-render mono->2ch (in-distribution input, "
                         "but metric domain is HRIR-colored, NOT standard-mono-comparable). "
                         "mono_dup: duplicate mono->L=R (OOD input for the binaural model, "
                         "but metrics are on the true mono signal = directly comparable to "
                         "mono SE papers).")
    ap.add_argument("--hydra_overrides", nargs="*", default=[])
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device={device}")

    # ---- build model EXACTLY like eval.py:evaluate_model_improved ----
    from hydra import compose, initialize
    from hydra.core.global_hydra import GlobalHydra
    GlobalHydra.instance().clear()
    initialize(config_path="config")
    config = compose(config_name="config", overrides=list(args.hydra_overrides))
    from DCNN.trainer import DCNNLightningModule
    model = DCNNLightningModule(config)
    model.eval()
    checkpoint = torch.load(args.ckpt, map_location=device)
    model.load_state_dict(checkpoint["state_dict"])
    model = model.to(device)

    # ---- metrics ----
    from pesq import pesq
    from pystoi import stoi
    from torchmetrics.functional.audio import scale_invariant_signal_distortion_ratio as si_sdr

    hl, hr = load_hrir_az0(args.hrir)

    def pe_pesq(ref2, deg2, mode):
        vals = []
        for c in range(2):
            try:
                vals.append(pesq(SR, ref2[c].astype(np.float32), deg2[c].astype(np.float32), mode))
            except Exception:
                pass
        return float(np.mean(vals)) if vals else float("nan")

    def pe_stoi(ref2, deg2, extended):
        try:
            return float(np.mean([stoi(ref2[c], deg2[c], SR, extended=extended) for c in range(2)]))
        except Exception:
            return float("nan")

    def pe_sisdr(ref2, deg2):
        try:
            return float(np.mean([
                float(si_sdr(torch.tensor(deg2[c]), torch.tensor(ref2[c]))) for c in range(2)
            ]))
        except Exception:
            return float("nan")

    clean_files = sorted(glob.glob(os.path.join(args.clean_dir, "*.wav")))
    if args.limit:
        clean_files = clean_files[:args.limit]
    print(f"{len(clean_files)} clean files in {args.clean_dir}")

    rows = []
    for i, cf in enumerate(clean_files):
        name = os.path.basename(cf)
        nf = os.path.join(args.noisy_dir, name)
        if not os.path.exists(nf):
            continue
        clean_m, _ = librosa.load(cf, sr=SR, mono=True)
        noisy_m, _ = librosa.load(nf, sr=SR, mono=True)
        n = min(len(clean_m), len(noisy_m))
        clean_m, noisy_m = clean_m[:n], noisy_m[:n]
        if n < int(0.3 * SR):
            continue
        if args.input_mode == "mono_dup":
            # duplicate mono -> L=R (metrics land on the true mono signal)
            clean_b = np.stack([clean_m, clean_m]).astype(np.float32)
            noisy_b = np.stack([noisy_m, noisy_m]).astype(np.float32)
        else:
            clean_b = render_binaural(clean_m, hl, hr)
            noisy_b = render_binaural(noisy_m, hl, hr)
        m = min(clean_b.shape[1], noisy_b.shape[1])
        clean_b, noisy_b = clean_b[:, :m], noisy_b[:, :m]
        # normalize the PAIR by the noisy peak -> 0.99 (preserves clean/noisy SNR,
        # matches the model's training input range). SI-SDR is scale-invariant anyway.
        g = 0.99 / max(np.max(np.abs(noisy_b)), 1e-9)
        clean_b *= g
        noisy_b *= g
        with torch.no_grad():
            enh = model(torch.tensor(noisy_b, dtype=torch.float32).unsqueeze(0).to(device))
            enh = enh.cpu().numpy()[0].astype(np.float32)
        mm = min(enh.shape[1], clean_b.shape[1])
        enh, clean_c, noisy_c = enh[:, :mm], clean_b[:, :mm], noisy_b[:, :mm]

        row = {"file": name}
        row["PESQ_nb_noisy"] = pe_pesq(clean_c, noisy_c, "nb")
        row["PESQ_nb_enh"] = pe_pesq(clean_c, enh, "nb")
        row["PESQ_wb_noisy"] = pe_pesq(clean_c, noisy_c, "wb")
        row["PESQ_wb_enh"] = pe_pesq(clean_c, enh, "wb")
        row["STOI_noisy"] = pe_stoi(clean_c, noisy_c, False)
        row["STOI_enh"] = pe_stoi(clean_c, enh, False)
        row["ESTOI_noisy"] = pe_stoi(clean_c, noisy_c, True)
        row["ESTOI_enh"] = pe_stoi(clean_c, enh, True)
        row["SISDR_noisy"] = pe_sisdr(clean_c, noisy_c)
        row["SISDR_enh"] = pe_sisdr(clean_c, enh)
        rows.append(row)
        if (i + 1) % 100 == 0:
            print(f"  {i + 1}/{len(clean_files)} processed", flush=True)

    if not rows:
        print("ERROR: no pairs processed")
        sys.exit(1)

    keys = list(rows[0].keys())
    with open(os.path.join(args.out, "vbdemand_per_file.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)

    def avg(k):
        return float(np.nanmean([r[k] for r in rows]))

    print(f"\n===== VoiceBank+DEMAND (input_mode={args.input_mode}, per-ear) ZERO-SHOT =====")
    print(f"files: {len(rows)}")
    lines = [f"VoiceBank+DEMAND input_mode={args.input_mode} per-ear zero-shot", f"files: {len(rows)}"]
    for metric in ["PESQ_nb", "PESQ_wb", "STOI", "ESTOI", "SISDR"]:
        no, en = avg(metric + "_noisy"), avg(metric + "_enh")
        msg = f"{metric:8s}: noisy={no:.4f}  enh={en:.4f}  delta={en - no:+.4f}"
        print(msg)
        lines.append(msg)
    with open(os.path.join(args.out, "vbdemand_averages.txt"), "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"\nwrote {args.out}/vbdemand_averages.txt + vbdemand_per_file.csv")


if __name__ == "__main__":
    main()
