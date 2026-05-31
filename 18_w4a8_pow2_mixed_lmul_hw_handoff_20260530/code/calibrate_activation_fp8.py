"""Collect fixed FP8 activation scales for calibrated_e4m3 mode.

The model still runs the validated dynamic-scaled activation path while this
script records each quantization boundary. The emitted JSON is then used by
BCCTN_ACTIVATION_FP8_MODE=calibrated_e4m3 so runtime needs no running amax.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import torch
from hydra import compose, initialize
from hydra.core.global_hydra import GlobalHydra

from DCNN.datasets import create_torch_dataloaders
from DCNN.trainer import DCNNLightningModule
from DCNN.utils.fp8_qat import (
    get_activation_calibration_stats,
    reset_activation_calibration_stats,
)


def _move_input_to_device(batch, device):
    if isinstance(batch, (tuple, list)):
        x = batch[0]
    else:
        x = batch
    return x.to(device, non_blocking=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--split", choices=("train", "val"), default="train")
    parser.add_argument("--batches", type=int, default=64)
    parser.add_argument("--quantile", type=float, default=0.999)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--hydra_overrides", nargs="*", default=[])
    args = parser.parse_args()

    os.environ["BCCTN_ACTIVATION_FP8_MODE"] = "calibrate_e4m3"
    os.environ["BCCTN_ACTIVATION_CALIB_QUANTILE"] = str(args.quantile)
    torch.manual_seed(args.seed)

    GlobalHydra.instance().clear()
    initialize(config_path="config", version_base="1.1")
    config = compose(config_name="config", overrides=list(args.hydra_overrides))

    train_loader, val_loader, _ = create_torch_dataloaders(config)
    loader = train_loader if args.split == "train" else val_loader

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    lm = DCNNLightningModule(config)
    ckpt = torch.load(args.ckpt, map_location="cpu")
    lm.load_state_dict(ckpt["state_dict"])
    lm.eval().to(device)

    reset_activation_calibration_stats()
    with torch.no_grad():
        for batch_idx, batch in enumerate(loader):
            if batch_idx >= args.batches:
                break
            x = _move_input_to_device(batch, device)
            _ = lm.model(x)
            if (batch_idx + 1) % 8 == 0:
                print(f"[act-calib] processed {batch_idx + 1}/{args.batches} batches")

    keys = get_activation_calibration_stats(max_val=448.0)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "format_version": 1,
        "mode": "calibrated_e4m3",
        "source_ckpt": str(Path(args.ckpt).resolve()),
        "split": args.split,
        "requested_batches": args.batches,
        "quantile": args.quantile,
        "stat": "max_of_batch_quantile",
        "max_fp8_value": 448.0,
        "scale_definition": "scale=448/calib_amax, descale=calib_amax/448",
        "keys": keys,
    }
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, sort_keys=True)

    descales = [v["descale"] for v in keys.values()]
    print(f"[act-calib] wrote {out_path}")
    print(f"[act-calib] keys={len(keys)} quantile={args.quantile}")
    if descales:
        print(
            "[act-calib] descale range "
            f"[{min(descales):.6e}, {max(descales):.6e}]"
        )


if __name__ == "__main__":
    main()
