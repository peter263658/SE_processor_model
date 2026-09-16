"""MetricGAN PESQ finetune entry point for the FP32-512 BCCTN.

Warm-starts the generator from a checkpoint (env/config
`training.train_checkpoint_path`), then trains G with the standard
BinauralLoss anchors + a MetricGAN PESQ critic (see DCNN/metricgan.py).

Usage (Hydra overrides as usual):
  python train_metricgan.py \
      +model.gan_weight=12 \
      training.learning_rate=2e-5 \
      training.n_epochs=20 \
      training.train_checkpoint_path=<generator warm-start ckpt> \
      hydra.run.dir=<run dir>
"""

import hydra
from omegaconf import DictConfig
import warnings
warnings.simplefilter("ignore")

import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping

from DCNN.datasets import create_torch_dataloaders
from DCNN.models.binaural_attention_model import BinauralAttentionDCNN
from DCNN.metricgan import MetricGANModule


def _warm_start(model, ckpt_path):
    ckpt = torch.load(ckpt_path, map_location="cpu")
    state = {}
    for k, v in ckpt["state_dict"].items():
        if k.startswith("model."):
            k = k[len("model."):]
        state[k] = v
    result = model.load_state_dict(state, strict=False)
    loaded = len(state) - len(result.unexpected_keys)
    print(f"[warm-start] loaded {loaded}/{len(state)} tensors; "
          f"{len(result.missing_keys)} model keys left at init")
    if loaded < len(state) // 2:
        raise RuntimeError("warm-start loaded under half the tensors; key mismatch?")


@hydra.main(config_path="config", config_name="config", version_base="1.1")
def train(config: DictConfig):
    dl_train, dl_val, dl_test = create_torch_dataloaders(config)

    generator = BinauralAttentionDCNN(**config["model"])
    ckpt_path = config["training"].get("train_checkpoint_path")
    if ckpt_path:
        _warm_start(generator, ckpt_path)

    module = MetricGANModule(generator, config)

    es_cfg = config["training"]["early_stopping"]
    callbacks = [
        ModelCheckpoint(monitor="validation_loss", save_last=True,
                        save_top_k=5, save_weights_only=True),
    ]
    if es_cfg["enabled"]:
        callbacks.append(EarlyStopping("validation_loss",
                                       es_cfg["min_delta"],
                                       es_cfg["patience_in_epochs"]))

    trainer = pl.Trainer(
        max_epochs=config["training"]["n_epochs"],
        accelerator="cuda" if torch.cuda.is_available() else "cpu",
        devices=1,
        callbacks=callbacks,
        logger=pl.loggers.TensorBoardLogger(save_dir="logs"),
    )
    trainer.fit(module, dl_train, val_dataloaders=dl_val)


if __name__ == "__main__":
    train()
