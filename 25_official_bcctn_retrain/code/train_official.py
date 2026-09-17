#!/usr/bin/env python3
"""Train the OFFICIAL BCCTN architecture (VikasTokala/BCCTN, 11.09 M) on our dataset, resumably.

Model and loss are imported from the pinned official submodule
(../../23_official_bcctn_ckpt_eval/official_bcctn); hyper-parameters are the official ones
(config_official.yaml). What differs from the official trainer, on purpose:
  * checkpoints hold the FULL training state (official uses save_weights_only=True): model, the loss
    module's buffers (it is a submodule, so they are in state_dict), Adam state, MultiStepLR state,
    epoch / global_step / loop progress  ->  `last.ckpt` resumes exactly;
  * every epoch is kept (post-hoc MBSTOI screening) and `last.ckpt` is rewritten each epoch;
  * re-running the same command auto-resumes from <out>/checkpoints/last.ckpt (server powers off 07:00);
  * written for pytorch-lightning 2.x (official code targets 1.x: `gpus=`, `*_epoch_end`);
  * ONE device only.
State-dict keys are `model.*` / `loss.*`, i.e. loadable by 23_.../code/eval_official.py unchanged.

  CUDA_VISIBLE_DEVICES=<one gpu> python train_official.py --out ../outputs/official_recipe
  smoke:  ... --out ../outputs/smoke --limit_train_batches 20 --limit_val_batches 5 --max_epochs 2
"""
import argparse
import os
import sys
from pathlib import Path

import yaml

HERE = Path(__file__).resolve().parent
OFFICIAL_DIR = (HERE / ".." / ".." / "23_official_bcctn_ckpt_eval" / "official_bcctn").resolve()
sys.dont_write_bytecode = True          # the official repo tracks its own .pyc files
sys.path.insert(0, str(OFFICIAL_DIR))

import pytorch_lightning as pl          # noqa: E402
import torch                            # noqa: E402
import torchaudio                       # noqa: E402
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint  # noqa: E402
from pytorch_lightning.loggers import CSVLogger, TensorBoardLogger            # noqa: E402

import DCNN.loss as official_loss_mod                                          # noqa: E402
from DCNN.loss import BinauralLoss                                             # noqa: E402
from DCNN.models.binaural_attention_model import BinauralAttentionDCNN        # noqa: E402

official_loss_mod.print = lambda *a, **k: None   # the official loss prints every step


class PairedDataset(torch.utils.data.Dataset):
    """Official BaseDataset pairing (sorted file order) + a hard check that each pair really matches."""

    def __init__(self, noisy_dir, clean_dir):
        self.noisy = sorted(str(p) for p in Path(noisy_dir).rglob("*.wav"))
        self.clean = sorted(str(p) for p in Path(clean_dir).rglob("*.wav"))
        assert len(self.noisy) == len(self.clean) > 0, (len(self.noisy), len(self.clean))
        for n, c in zip(self.noisy, self.clean):
            cs = Path(c).stem
            assert Path(n).stem.startswith(cs + "_snr") or Path(n).stem == cs, f"pair mismatch: {n} vs {c}"

    def __len__(self):
        return len(self.noisy)

    def __getitem__(self, i):
        noisy, _ = torchaudio.load(self.noisy[i])
        clean, _ = torchaudio.load(self.clean[i])
        return noisy, clean


class OfficialBCCTN(pl.LightningModule):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.save_hyperparameters({"cfg": cfg})
        self.model = BinauralAttentionDCNN(**cfg["model"])
        self.loss = BinauralLoss(**cfg["loss"])

    def forward(self, x):
        return self.model(x)

    def _step(self, batch, name):
        x, y = batch
        loss = self.loss(self.model(x), y)
        self.log(name, loss, on_step=(name == "train_loss"), on_epoch=True, prog_bar=True,
                 batch_size=x.shape[0], sync_dist=False)
        return loss

    def training_step(self, batch, _):
        return self._step(batch, "train_loss")

    def validation_step(self, batch, _):
        return self._step(batch, "validation_loss")

    def configure_optimizers(self):
        t = self.cfg["training"]
        opt = torch.optim.Adam(self.parameters(), lr=t["learning_rate"])
        sch = torch.optim.lr_scheduler.MultiStepLR(opt, t["learning_rate_decay_steps"], t["learning_rate_decay_values"])
        return [opt], [sch]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=str(HERE / "config_official.yaml"))
    ap.add_argument("--out", required=True)
    ap.add_argument("--max_epochs", type=int, default=None)
    ap.add_argument("--limit_train_batches", type=float, default=1.0)
    ap.add_argument("--limit_val_batches", type=float, default=1.0)
    a = ap.parse_args()
    cfg = yaml.safe_load(open(a.config))
    t, d = cfg["training"], cfg["dataset"]
    assert torch.cuda.device_count() <= 1, "expose exactly ONE GPU via CUDA_VISIBLE_DEVICES (server rule)"
    pl.seed_everything(t["seed"], workers=True)
    out = Path(a.out).resolve(); ckdir = out / "checkpoints"; ckdir.mkdir(parents=True, exist_ok=True)
    yaml.safe_dump(cfg, open(out / "config_used.yaml", "w"))

    def loader(noisy, clean, shuffle):
        return torch.utils.data.DataLoader(PairedDataset(noisy, clean), batch_size=t["batch_size"], shuffle=shuffle,
                                           num_workers=t["n_workers"], pin_memory=t["pin_memory"], drop_last=False,
                                           persistent_workers=True)
    train = loader(d["noisy_training_dataset_dir"], d["target_training_dataset_dir"], True)
    val = loader(d["noisy_validation_dataset_dir"], d["target_validation_dataset_dir"], False)

    module = OfficialBCCTN(cfg)
    n = sum(p.numel() for p in module.model.parameters())
    print(f"[train] model params {n:,} (expect 11,086,488) | train {len(train.dataset)} val {len(val.dataset)} "
          f"| batch {t['batch_size']} lr {t['learning_rate']} epochs {t['n_epochs']}")
    assert n == 11_086_488, n
    callbacks = [
        ModelCheckpoint(dirpath=ckdir, filename="all-{epoch:02d}-{validation_loss:.4f}", monitor="validation_loss",
                        save_top_k=-1, save_last=True, save_weights_only=False, auto_insert_metric_name=True),
        LearningRateMonitor(logging_interval="epoch"),
    ]
    if t.get("early_stopping"):
        from pytorch_lightning.callbacks import EarlyStopping
        callbacks.append(EarlyStopping("validation_loss", min_delta=0.01, patience=3))
    trainer = pl.Trainer(max_epochs=a.max_epochs or t["n_epochs"], accelerator="gpu" if torch.cuda.is_available() else "cpu",
                         devices=1, callbacks=callbacks, log_every_n_steps=50,
                         logger=[CSVLogger(out, name="csv"), TensorBoardLogger(out, name="tb")],
                         limit_train_batches=a.limit_train_batches, limit_val_batches=a.limit_val_batches,
                         num_sanity_val_steps=2)
    last = ckdir / "last.ckpt"
    if last.exists():
        ck = torch.load(last, map_location="cpu")
        print(f"[train] RESUMING from {last}: epoch {ck['epoch']} step {ck['global_step']} "
              f"optimizer_states={len(ck.get('optimizer_states', []))} lr_schedulers={len(ck.get('lr_schedulers', []))}")
        del ck
    else:
        print("[train] fresh start")
    trainer.fit(module, train, val, ckpt_path=str(last) if last.exists() else None)
    print(f"[train] finished: epoch {trainer.current_epoch} step {trainer.global_step}")


if __name__ == "__main__":
    main()
