"""Train LBCCN (Wang et al., ICASSP 2025) on the BCCTN binaural dataset.

Model/loss are the official release (github.com/jywanng/LBCCN @ 12aa002);
data pipeline mirrors BCCTN so the comparison is apples-to-apples.
Paper training config: Adam, lr 1e-4, 2 s / 16 kHz segments, trained to
convergence (~20-50 epochs). Loss weights are the official loss.py defaults.
"""
import argparse
import os

import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping, TQDMProgressBar
from pytorch_lightning import loggers as pl_loggers
from torch.utils.data import DataLoader

from model import Model
from loss import BinauralLoss
from dataset import BaseDataset


COMPONENT_NAMES = [
    "voice_snr", "noise_snr", "voice_stoi", "noise_stoi",
    "voice_ild", "noise_ild", "voice_ipd", "noise_ipd",
    "fullband_ild", "fullband_ipd",
]


class LBCCNModule(pl.LightningModule):
    def __init__(self, lr=1e-4, band_num=40,
                 ild_weight=0.5, ipd_weight=5.0, stoi_weight=5.0,
                 snr_loss_weight=0.5):
        super().__init__()
        self.save_hyperparameters()
        self.model = Model(band_num=band_num)
        self.loss = BinauralLoss(
            ild_weight=ild_weight, ipd_weight=ipd_weight,
            stoi_weight=stoi_weight, snr_loss_weight=snr_loss_weight,
            band_num=band_num)
        self.lr = lr

    def forward(self, x):
        return self.model(x)

    def _step(self, batch, stage):
        noisy, clean = batch
        noise_target = noisy - clean
        voice, noise, vlc, vrc, nlc, nrc = self.model(noisy)
        loss, comps = self.loss(voice, noise, vlc, vrc, nlc, nrc,
                                clean, noise_target)

        self.log(f"{stage}_loss", loss, on_step=False, on_epoch=True,
                 prog_bar=True, sync_dist=False)
        for name, value in zip(COMPONENT_NAMES, comps):
            if torch.is_tensor(value):
                value = value.detach().float().mean()
            self.log(f"{stage}_{name}", value, on_step=False, on_epoch=True)
        return loss

    def training_step(self, batch, batch_idx):
        return self._step(batch, "train")

    def validation_step(self, batch, batch_idx):
        self._step(batch, "validation")

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    root = "/raid/R12K41024/BCCTN/Dataset"
    p.add_argument("--noisy_train", default=f"{root}/noisy_trainset")
    p.add_argument("--clean_train", default=f"{root}/clean_trainset")
    p.add_argument("--noisy_val", default=f"{root}/noisy_valset")
    p.add_argument("--clean_val", default=f"{root}/clean_valset")
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--band_num", type=int, default=40)
    p.add_argument("--n_workers", type=int, default=4)
    p.add_argument("--patience", type=int, default=10)
    p.add_argument("--grad_clip", type=float, default=0.0,
                   help="0 disables clipping (paper default)")
    p.add_argument("--resume", default=None, help="ckpt path to resume from")
    p.add_argument("--out_dir", default="../logs")
    p.add_argument("--limit_batches", type=float, default=1.0,
                   help="fraction of batches per epoch (smoke tests)")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    pl.seed_everything(args.seed)

    train_ds = BaseDataset(args.noisy_train, args.clean_train)
    val_ds = BaseDataset(args.noisy_val, args.clean_val)
    train_dl = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                          num_workers=args.n_workers, pin_memory=True,
                          drop_last=False)
    val_dl = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                        num_workers=args.n_workers, pin_memory=True,
                        drop_last=False)

    module = LBCCNModule(lr=args.lr, band_num=args.band_num)
    n_params = sum(x.numel() for x in module.model.parameters())
    print(f"LBCCN parameters: {n_params/1e3:.1f} K "
          f"(paper reports 38.0 K)")

    ckpt_cb = ModelCheckpoint(monitor="validation_loss", mode="min",
                              save_top_k=1, save_last=True,
                              filename="lbccn-{epoch:02d}-{validation_loss:.3f}")
    es_cb = EarlyStopping(monitor="validation_loss", mode="min",
                          min_delta=0.0, patience=args.patience)
    tb_logger = pl_loggers.TensorBoardLogger(save_dir=args.out_dir,
                                             name="lbccn")

    trainer = pl.Trainer(
        max_epochs=args.epochs,
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        devices=1,
        callbacks=[ckpt_cb, es_cb, TQDMProgressBar(refresh_rate=50)],
        logger=tb_logger,
        gradient_clip_val=args.grad_clip if args.grad_clip > 0 else None,
        log_every_n_steps=100,
        limit_train_batches=args.limit_batches,
        limit_val_batches=args.limit_batches,
        detect_anomaly=False,
    )
    trainer.fit(module, train_dl, val_dl, ckpt_path=args.resume)
    print(f"best checkpoint: {ckpt_cb.best_model_path} "
          f"(val loss {ckpt_cb.best_model_score})")


if __name__ == "__main__":
    main()
