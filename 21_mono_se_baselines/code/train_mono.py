"""Train a mono-SE baseline (SNU BS-TGRU net or NTU CNN) on mono-ear VCTK.

FP32, SI-SDR waveform loss (favorable upper bound for the mono paradigm; no
quantization).  Saves best/last checkpoints and a JSONL training log.

`--loss mse` and `--snr_band` step this down to the training regime the NTU
JSSC'20 paper actually reports (Sec. VI-B: 620 utterances mixed at a single
0 dB SNR; the loss is never stated, so MSE on the magnitude spectrum -- the
objective of the Lai et al. APSIPA model it simplifies -- stands in for it).

Usage:
  python train_mono.py --arch snu --output_dir ../results/snu
  python train_mono.py --arch ntu --output_dir ../results/ntu
  python train_mono.py --arch ntu --loss mse --snr_band 0.2 \
      --output_dir ../results/ntu_aspub
"""

import argparse
import json
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from mono_dataset import MonoEarDataset
from models.snu_bstgru import SNUBstgruNet
from models.ntu_cnn import NTUCnnNet

DATA_ROOT = Path("/raid/R12K41024/BCCTN/Dataset")


def si_sdr(est, ref, eps=1e-8):
    """Scale-invariant SDR in dB, per batch element. est/ref: (B, N)."""
    ref = ref - ref.mean(dim=-1, keepdim=True)
    est = est - est.mean(dim=-1, keepdim=True)
    alpha = (est * ref).sum(-1, keepdim=True) / (ref.pow(2).sum(-1, keepdim=True) + eps)
    target = alpha * ref
    noise = est - target
    return 10 * torch.log10((target.pow(2).sum(-1) + eps) / (noise.pow(2).sum(-1) + eps))


def mag_mse(est, ref, model):
    """MSE between enhanced and clean magnitude spectra, in the model's own STFT.

    The mask is applied to the noisy spectrum before the iSTFT, so this is the
    classic mask/spectral-mapping objective of the pre-SI-SDR era.
    """
    kw = dict(n_fft=model.N_FFT, hop_length=model.HOP, window=model.window,
              center=True, return_complex=True)
    return (torch.stft(est, **kw).abs() - torch.stft(ref, **kw).abs()).pow(2).mean()


def build_model(arch):
    return {"snu": SNUBstgruNet, "ntu": NTUCnnNet}[arch]()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arch", choices=["snu", "ntu"], required=True)
    ap.add_argument("--output_dir", required=True)
    ap.add_argument("--epochs", type=int, default=80)
    ap.add_argument("--batch_size", type=int, default=128)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--patience", type=int, default=10)
    ap.add_argument("--num_workers", type=int, default=8)
    ap.add_argument("--limit_train", type=int, default=0, help="debug: cap train samples")
    ap.add_argument("--loss", choices=["sisdr", "mse"], default="sisdr",
                    help="sisdr = modern waveform loss; mse = magnitude-spectrum MSE")
    ap.add_argument("--snr_band", type=float, default=None,
                    help="keep only train/val mixes with |SNR| <= this many dB")
    ap.add_argument("--noisy_train_dir", default=None,
                    help="override the noisy train set (e.g. a directional-"
                         "interferer set from 22_directional_interferer), so the "
                         "baseline can be retrained on exactly the data our own "
                         "model is retrained on")
    ap.add_argument("--noisy_val_dir", default=None)
    ap.add_argument("--init_from", default=None,
                    help="warm-start from an existing checkpoint. Needed so the "
                         "baseline gets the same treatment as our own model when "
                         "both are fine-tuned onto a new acoustic condition -- a "
                         "cold-started baseline against a warm-started candidate "
                         "is not a fair comparison")
    args = ap.parse_args()

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    log_path = out / "train_log.jsonl"
    torch.manual_seed(0)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(args.arch).to(device)
    if args.init_from:
        model.load_state_dict(torch.load(args.init_from, map_location=device)["state_dict"])
        print(f"warm-started from {args.init_from}", flush=True)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"arch={args.arch} params={n_params:,} device={device}", flush=True)

    train_ds = MonoEarDataset(args.noisy_train_dir or DATA_ROOT / "noisy_trainset",
                              DATA_ROOT / "clean_trainset", snr_band=args.snr_band)
    val_ds = MonoEarDataset(args.noisy_val_dir or DATA_ROOT / "noisy_valset",
                            DATA_ROOT / "clean_valset", snr_band=args.snr_band)
    if args.limit_train:
        train_ds = torch.utils.data.Subset(train_ds, range(args.limit_train))
    print(f"train={len(train_ds)} val={len(val_ds)} mono samples", flush=True)

    train_dl = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                          num_workers=args.num_workers, pin_memory=True, drop_last=True)
    val_dl = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                        num_workers=args.num_workers, pin_memory=True)

    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode="max", factor=0.5, patience=3)

    def objective(est, clean):
        return -si_sdr(est, clean).mean() if args.loss == "sisdr" else mag_mse(est, clean, model)

    best_val, bad_epochs = float("-inf"), 0
    for epoch in range(1, args.epochs + 1):
        model.train()
        t0, tr_loss, n_batches = time.time(), 0.0, 0
        for noisy, clean in train_dl:
            noisy, clean = noisy.to(device, non_blocking=True), clean.to(device, non_blocking=True)
            est = model(noisy)
            loss = objective(est, clean)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
            tr_loss += loss.item()
            n_batches += 1
        tr_loss /= max(n_batches, 1)

        model.eval()
        val_sum, val_n, val_obj, val_bn = 0.0, 0, 0.0, 0
        with torch.no_grad():
            for noisy, clean in val_dl:
                noisy, clean = noisy.to(device), clean.to(device)
                est = model(noisy)
                s = si_sdr(est, clean)
                val_sum += s.sum().item()
                val_n += s.numel()
                val_obj += objective(est, clean).item() * noisy.shape[0]
                val_bn += noisy.shape[0]
        val_sisdr = val_sum / max(val_n, 1)
        val_obj /= max(val_bn, 1)
        # Select on the objective actually being trained -- an SI-SDR-selected
        # checkpoint would smuggle the modern loss back into an MSE run.
        monitor = val_sisdr if args.loss == "sisdr" else -val_obj
        sched.step(monitor)

        rec = {"epoch": epoch, "train_sisdr": round(-tr_loss, 3) if args.loss == "sisdr" else None,
               "train_loss": round(tr_loss, 6), "val_loss": round(val_obj, 6),
               "val_sisdr": round(val_sisdr, 3), "lr": opt.param_groups[0]["lr"],
               "sec": round(time.time() - t0, 1)}
        print(json.dumps(rec), flush=True)
        with open(log_path, "a") as f:
            f.write(json.dumps(rec) + "\n")

        ckpt = {"arch": args.arch, "epoch": epoch, "val_sisdr": val_sisdr,
                "val_loss": val_obj, "loss": args.loss, "snr_band": args.snr_band,
                "n_params": n_params, "state_dict": model.state_dict()}
        torch.save(ckpt, out / f"last_{args.arch}.pt")
        if monitor > best_val:
            best_val, bad_epochs = monitor, 0
            torch.save(ckpt, out / f"best_{args.arch}.pt")
        else:
            bad_epochs += 1
            if bad_epochs >= args.patience:
                print(f"early stop at epoch {epoch} (best monitor {best_val:.4f})", flush=True)
                break

    print(f"done: best monitor {best_val:.4f} ({args.loss}) -> {out / f'best_{args.arch}.pt'}", flush=True)


if __name__ == "__main__":
    main()
