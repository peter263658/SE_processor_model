"""MetricGAN PESQ finetune on the W4A8 DEPLOY architecture.

Unlike train_metricgan.py (FP32-512 generator), the generator here MUST go
through the exact deploy hardware surgery (mixed-precision W4A8 QAT, APoT/pow2
block-float, MHA/outlier L-Mul, grouped decoder, pow2 BN/PReLU, top-4 prob
packet). We get that for free by letting the standard DCNNLightningModule build
and QAT-wrap the model, then warm-starting it with the same _load_checkpoint the
BaseTrainer uses. We then hand the fully-wrapped generator to MetricGANModule.

Goal: lift wideband PESQ past the (reproduced) SNU baseline WHILE holding the
MBSTOI >= 0.835 deploy gate and keeping ILD positive -- MetricGAN's learned
PESQ critic gives a gentler, less MBSTOI-destroying gradient than cranking the
torch_pesq weight (which drops MBSTOI to ~0.82).

Usage (Hydra overrides via run_metricgan_deploy.sh):
  python train_metricgan_deploy.py \
      training.train_checkpoint_path='<deploy ep3 ckpt>' \
      +model.gan_weight=16 +training.lr_d=5e-4 \
      model.pesq_weight=0 model.stoi_weight=10 model.ild_weight=4 ... \
      <full W4A8 override stack> hydra.run.dir=<run dir>
"""

import os

import hydra
from omegaconf import DictConfig
import warnings
warnings.simplefilter("ignore")

import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping

from DCNN.datasets import create_torch_dataloaders
from DCNN.trainer import DCNNLightningModule
from DCNN.utils.base_trainer import _load_checkpoint
from DCNN.metricgan import MetricGANModule


@hydra.main(config_path="config", config_name="config", version_base="1.1")
def train(config: DictConfig):
    dl_train, dl_val, dl_test = create_torch_dataloaders(config)

    # Build the deploy generator through the standard module so ALL the W4A8
    # QAT / L-Mul / APoT surgery is applied identically to the finetune path.
    dcnn_lm = DCNNLightningModule(config)
    ckpt_path = config["training"].get("train_checkpoint_path", None)
    if ckpt_path:
        print(f"[metricgan-deploy] warm-start generator from {ckpt_path}")
        _load_checkpoint(dcnn_lm.model, ckpt_path)
    else:
        print("[metricgan-deploy] WARNING: no warm-start ckpt; generator at init")
    generator = dcnn_lm.model

    module = MetricGANModule(generator, config)

    # Optional discriminator warm init (BCCTN_D_INIT_CKPT): load disc.* weights
    # from a prior MetricGAN checkpoint (e.g. the FP32-512 run's critic, or a
    # D-warm-up run) so G never trains against a random critic. Env var, not a
    # Hydra override, because ckpt paths contain '=' characters.
    d_init = os.environ.get("BCCTN_D_INIT_CKPT", "").strip()
    if d_init:
        sd = torch.load(d_init, map_location="cpu")["state_dict"]
        dsd = {k[len("disc."):]: v for k, v in sd.items() if k.startswith("disc.")}
        if not dsd:
            raise RuntimeError(f"BCCTN_D_INIT_CKPT has no disc.* keys: {d_init}")
        module.disc.load_state_dict(dsd, strict=True)
        print(f"[metricgan-deploy] D warm-initialized from {d_init} ({len(dsd)} tensors)")

    # Keep every epoch so the PESQ/MBSTOI Pareto can be swept post-hoc (the
    # gate-holding operating point is whichever epoch first crosses SNU PESQ
    # while MBSTOI is still >= 0.835).
    callbacks = [
        ModelCheckpoint(monitor="validation_loss", save_last=True,
                        save_top_k=-1, every_n_epochs=1, save_weights_only=True),
    ]
    es_cfg = config["training"].get("early_stopping", {}) or {}
    if es_cfg.get("enabled", False):
        callbacks.append(EarlyStopping("validation_loss",
                                       es_cfg.get("min_delta", 0.0),
                                       es_cfg.get("patience_in_epochs", 10)))

    trainer = pl.Trainer(
        max_epochs=config["training"]["n_epochs"],
        accelerator="cuda" if torch.cuda.is_available() else "cpu",
        devices=1,
        callbacks=callbacks,
        num_sanity_val_steps=0,
        limit_train_batches=config["training"].get("limit_train_batches", 1.0),
        limit_val_batches=config["training"].get("limit_val_batches", 1.0),
        logger=pl.loggers.TensorBoardLogger(save_dir="logs"),
    )
    trainer.fit(module, dl_train, val_dataloaders=dl_val)


if __name__ == "__main__":
    train()
