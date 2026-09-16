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


# ---- module-level metrics (so multiprocessing workers can pickle/call them) ----
from pesq import pesq as _pesq
from pystoi import stoi as _stoi


def _pe_pesq(ref2, deg2, mode):
    vals = []
    for c in range(2):
        try:
            vals.append(_pesq(SR, ref2[c].astype(np.float32), deg2[c].astype(np.float32), mode))
        except Exception:
            pass
    return float(np.mean(vals)) if vals else float("nan")


def _pe_stoi(ref2, deg2, extended):
    try:
        return float(np.mean([_stoi(ref2[c], deg2[c], SR, extended=extended) for c in range(2)]))
    except Exception:
        return float("nan")


def _si_sdr_np(preds, target):
    # numpy port of torchmetrics scale_invariant_signal_distortion_ratio
    # (zero_mean=False). Avoids torch in forked workers; matches to ~1e-5 dB.
    eps = float(np.finfo(np.float32).eps)
    preds = preds.astype(np.float32); target = target.astype(np.float32)
    alpha = (float(np.sum(target * preds)) + eps) / (float(np.sum(target ** 2)) + eps)
    ts = alpha * target
    noise = ts - preds
    val = (float(np.sum(ts ** 2)) + eps) / (float(np.sum(noise ** 2)) + eps)
    return float(10.0 * np.log10(val))


def _pe_sisdr(ref2, deg2):
    try:
        return float(np.mean([_si_sdr_np(deg2[c], ref2[c]) for c in range(2)]))
    except Exception:
        return float("nan")


def compute_metrics_row(payload):
    """CPU metric worker. payload = (name, clean_c, noisy_c, enh) -> row dict."""
    name, clean_c, noisy_c, enh = payload
    return {
        "file": name,
        "PESQ_nb_noisy": _pe_pesq(clean_c, noisy_c, "nb"),
        "PESQ_nb_enh":   _pe_pesq(clean_c, enh, "nb"),
        "PESQ_wb_noisy": _pe_pesq(clean_c, noisy_c, "wb"),
        "PESQ_wb_enh":   _pe_pesq(clean_c, enh, "wb"),
        "STOI_noisy":  _pe_stoi(clean_c, noisy_c, False),
        "STOI_enh":    _pe_stoi(clean_c, enh, False),
        "ESTOI_noisy": _pe_stoi(clean_c, noisy_c, True),
        "ESTOI_enh":   _pe_stoi(clean_c, enh, True),
        "SISDR_noisy": _pe_sisdr(clean_c, noisy_c),
        "SISDR_enh":   _pe_sisdr(clean_c, enh),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--clean_dir", required=True)
    ap.add_argument("--noisy_dir", required=True)
    ap.add_argument("--hrir", required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--metric_workers", type=int, default=16,
                    help="CPU workers for the metric loop (PESQ/STOI/SISDR). "
                         "GPU inference stays in the main process; metrics run in "
                         "a pool. Keep <=32 per eval (shared box).")
    ap.add_argument("--input_mode", default="binaural_az0",
                    choices=["binaural_az0", "mono_dup", "binaural_prerendered"],
                    help="binaural_az0: HRIR-render mono->2ch on the fly (HRIR-colored, "
                         "NOT standard-mono-comparable). mono_dup: duplicate mono->L=R "
                         "(true-mono metrics, comparable to mono SE papers). "
                         "binaural_prerendered: clean_dir/noisy_dir already hold 2ch "
                         "HRIR-rendered wavs (gen_vb_binaural.py) -> load directly, no "
                         "re-convolution. Numerically identical to binaural_az0.")
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
    # Drop loss.* keys (PESQ/STOI filter buffers): not used at eval, and the
    # PESQ-finetuned ckpts carry loss.pesq_loss.* that older ckpts lack. Filter
    # + strict=False keeps load robust while still surfacing real model.* drift.
    state_dict = {k: v for k, v in checkpoint["state_dict"].items()
                  if not k.startswith("loss.")}
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    model_missing = [k for k in missing if not k.startswith("loss.")]
    model_unexpected = [k for k in unexpected if not k.startswith("loss.")]
    if model_missing or model_unexpected:
        raise RuntimeError(
            f"model weight mismatch loading {args.ckpt}:\n"
            f"  missing={model_missing}\n  unexpected={model_unexpected}")
    model = model.to(device)

    # HRIR only needed for on-the-fly render; prerendered/mono_dup skip it.
    hl = hr = None
    if args.input_mode == "binaural_az0":
        hl, hr = load_hrir_az0(args.hrir)

    clean_files = sorted(glob.glob(os.path.join(args.clean_dir, "*.wav")))
    if args.limit:
        clean_files = clean_files[:args.limit]
    print(f"{len(clean_files)} clean files in {args.clean_dir}")

    def infer(cf):
        """Load/render + GPU inference -> (name, clean_c, noisy_c, enh) or None."""
        name = os.path.basename(cf)
        nf = os.path.join(args.noisy_dir, name)
        if not os.path.exists(nf):
            return None
        if args.input_mode == "binaural_prerendered":
            cb, _ = librosa.load(cf, sr=SR, mono=False)
            nb, _ = librosa.load(nf, sr=SR, mono=False)
            n = min(cb.shape[1], nb.shape[1])
            if n < int(0.3 * SR):
                return None
            clean_b = cb[:, :n].astype(np.float32)
            noisy_b = nb[:, :n].astype(np.float32)
        else:
            clean_m, _ = librosa.load(cf, sr=SR, mono=True)
            noisy_m, _ = librosa.load(nf, sr=SR, mono=True)
            n = min(len(clean_m), len(noisy_m))
            clean_m, noisy_m = clean_m[:n], noisy_m[:n]
            if n < int(0.3 * SR):
                return None
            if args.input_mode == "mono_dup":
                clean_b = np.stack([clean_m, clean_m]).astype(np.float32)
                noisy_b = np.stack([noisy_m, noisy_m]).astype(np.float32)
            else:
                clean_b = render_binaural(clean_m, hl, hr)
                noisy_b = render_binaural(noisy_m, hl, hr)
        m = min(clean_b.shape[1], noisy_b.shape[1])
        clean_b, noisy_b = clean_b[:, :m], noisy_b[:, :m]
        # normalize the PAIR by the noisy peak -> 0.99 (preserves clean/noisy SNR).
        g = 0.99 / max(np.max(np.abs(noisy_b)), 1e-9)
        clean_b = clean_b * g
        noisy_b = noisy_b * g
        with torch.no_grad():
            enh = model(torch.tensor(noisy_b, dtype=torch.float32).unsqueeze(0).to(device))
            enh = enh.cpu().numpy()[0].astype(np.float32)
        mm = min(enh.shape[1], clean_b.shape[1])
        return (name, clean_b[:, :mm], noisy_b[:, :mm], enh[:, :mm])

    # GPU inference stays in the main process; CPU metrics run in a pool so the
    # two overlap (metrics for file i compute while inference does file i+1).
    rows = []
    workers = max(1, int(args.metric_workers))
    ntot = len(clean_files)
    if workers <= 1:
        for i, cf in enumerate(clean_files):
            p = infer(cf)
            if p is None:
                continue
            rows.append(compute_metrics_row(p))
            if (i + 1) % 100 == 0:
                print(f"  {i + 1}/{ntot} processed", flush=True)
    else:
        import multiprocessing as mp
        # forkserver: workers are forked from a CLEAN server process, so they do
        # NOT inherit the main process's CUDA context / held thread locks (a
        # plain fork after model.to(cuda) deadlocks the pool).
        ctx = mp.get_context("forkserver")
        print(f"metric pool: {workers} CPU workers (forkserver)", flush=True)
        with ctx.Pool(workers) as pool:
            pending, done = [], 0
            for cf in clean_files:
                p = infer(cf)
                if p is None:
                    continue
                pending.append(pool.apply_async(compute_metrics_row, (p,)))
                while len(pending) >= 2 * workers:   # bound in-flight memory
                    rows.append(pending.pop(0).get()); done += 1
                    if done % 100 == 0:
                        print(f"  {done}/{ntot} processed", flush=True)
            for ar in pending:
                rows.append(ar.get())

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
