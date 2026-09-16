"""Minimal, parallel eval for the from-scratch causal-512 model.

Loads the model the same way the legacy eval does, but computes enhanced MBSTOI /
STOI / ILD in a process pool (single logical run, no racing output tree). Noisy
MBSTOI baselines are already known (dirnoise90 0.8505, isotropic 0.7861) so only
the enhanced signal is scored, halving the CPU-bound MBSTOI cost.
"""
import os, sys, glob, re, argparse, json, warnings
warnings.simplefilter("ignore")
import numpy as np, soundfile as sf, torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from omegaconf import OmegaConf
from DCNN.trainer import DCNNLightningModule
from MBSTOI.mbstoi import mbstoi
from pystoi import stoi as pystoi_stoi
from ild_metric import compute_ild_error_with_mask
from concurrent.futures import ProcessPoolExecutor

SR = 16000
SNRS = [-6, -3, 0, 3, 6, 9, 12, 15]


def _metrics(p):
    cl0, cl1, el, er = p
    mb = st = il = None
    try:
        v = mbstoi(cl0, cl1, el, er, gridcoarseness=1)
        if np.isfinite(v): mb = float(v)
    except Exception: pass
    try:
        st = 0.5 * (pystoi_stoi(cl0, el, SR, extended=False) + pystoi_stoi(cl1, er, SR, extended=False))
    except Exception: pass
    try:
        iv = compute_ild_error_with_mask(cl0, cl1, el, er)
        if iv is not None and np.isfinite(iv): il = float(iv)
    except Exception: pass
    return mb, st, il


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--test_dir", required=True)
    ap.add_argument("--clean_dir", required=True)
    ap.add_argument("--config", default="config/config.yaml")
    ap.add_argument("--limit", type=int, default=750)
    ap.add_argument("--workers", type=int, default=32)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cfg = OmegaConf.load(a.config)
    lm = DCNNLightningModule(cfg)
    sd = torch.load(a.ckpt, map_location=dev)["state_dict"]
    sd = {k[len("model."):] if k.startswith("model.") else k: v for k, v in sd.items()}
    miss, _ = lm.model.load_state_dict(sd, strict=False)
    miss = [m for m in miss if not m.startswith("loss.")]
    assert not miss, f"missing keys: {miss[:5]}"
    model = lm.model.to(dev).eval()

    rx = re.compile(r'_snr[-+]?[\d.]+\.wav$')
    os.makedirs(a.out, exist_ok=True)
    rows = {}
    for snr in SNRS:
        d = os.path.join(a.test_dir, f"snr_{snr}dB")
        if not os.path.isdir(d):
            continue
        files = sorted(glob.glob(f"{d}/*.wav"))[:a.limit]
        payloads = []
        for f in files:
            cl = os.path.join(a.clean_dir, rx.sub(".wav", os.path.basename(f)))
            if not os.path.exists(cl):
                continue
            noisy, _ = sf.read(f, dtype="float32", always_2d=True)
            clean, _ = sf.read(cl, dtype="float32", always_2d=True)
            with torch.no_grad():
                enh = model(torch.tensor(noisy.T, dtype=torch.float32).unsqueeze(0).to(dev))
            enh = enh.cpu().numpy()[0]
            L = clean.shape[0]
            enh = enh[:, :L] if enh.shape[1] >= L else np.pad(enh, ((0, 0), (0, L - enh.shape[1])))
            payloads.append((clean[:, 0].copy(), clean[:, 1].copy(), enh[0].copy(), enh[1].copy()))
        with ProcessPoolExecutor(max_workers=a.workers) as ex:
            res = list(ex.map(_metrics, payloads))
        mb = [r[0] for r in res if r[0] is not None]
        st = [r[1] for r in res if r[1] is not None]
        il = [r[2] for r in res if r[2] is not None]
        rows[snr] = dict(n=len(mb),
                         mbstoi=float(np.mean(mb)) if mb else float("nan"),
                         stoi=float(np.mean(st)) if st else float("nan"),
                         ild=float(np.mean(il)) if il else float("nan"))
        print(f"SNR {snr:+d}: MBSTOI {rows[snr]['mbstoi']:.4f}  STOI {rows[snr]['stoi']:.4f}  "
              f"ILD {rows[snr]['ild']:.3f}  n={rows[snr]['n']}", flush=True)

    avg = lambda k: float(np.nanmean([rows[s][k] for s in rows]))
    summ = {str(s): rows[s] for s in rows}
    summ.update(avg_mbstoi=avg("mbstoi"), avg_stoi=avg("stoi"), avg_ild=avg("ild"))
    json.dump(summ, open(os.path.join(a.out, "summary.json"), "w"), indent=2)
    print(f"\nAVG  MBSTOI {avg('mbstoi'):.4f}  STOI {avg('stoi'):.4f}  ILD {avg('ild'):.3f}", flush=True)
    print("EVAL_MIN_DONE", flush=True)


if __name__ == "__main__":
    main()
