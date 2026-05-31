#!/usr/bin/env python3
"""Analytical MAC counter for P2 skip-subset variants on top of P1 g42.

Index identity does not affect MAC, only skip_select_ratio does, so this uses
first_n indices to build the reduced-shape models.
"""

from __future__ import annotations

import sys
from pathlib import Path

from omegaconf import OmegaConf
from torch import nn

REPO = Path(__file__).resolve().parents[1]
SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(SCRIPT_DIR))

from DCNN.models.model import DCNN  # noqa: E402
from count_p1_macs import count_macs_analytical, is_decoder_pointwise  # noqa: E402


VARIANTS = [
    ("g42_s100", 4, 2, 1.0),
    ("g42_s075", 4, 2, 0.75),
    ("g42_s050", 4, 2, 0.50),
]


def build_model(gx: int, gs: int, ratio: float) -> nn.Module:
    cfg = OmegaConf.load(REPO / "config" / "config.yaml")
    cfg.model.attention_softmax.mode = "shift_topk_pow2_clamp8_nomantissa"
    cfg.training.top_k_attention = 4
    cfg.model.decoder_grouped_pointwise.mode = "grouped"
    cfg.model.decoder_grouped_pointwise.groups_x = gx
    cfg.model.decoder_grouped_pointwise.groups_skip = gs
    cfg.model.decoder_grouped_pointwise.skip_select_ratio = ratio
    cfg.model.decoder_grouped_pointwise.skip_select_mode = "first_n"
    cfg.model.decoder_grouped_pointwise.skip_indices_path = None
    model = DCNN(**cfg.model)
    model.eval()
    return model


def main() -> None:
    rows = []
    base_total = base_dec_pw = None
    for name, gx, gs, ratio in VARIANTS:
        model = build_model(gx, gs, ratio)
        captured = count_macs_analytical(model)
        total = sum(captured.values())
        dec_pw = sum(v for k, v in captured.items() if is_decoder_pointwise(k))
        if base_total is None:
            base_total = total
            base_dec_pw = dec_pw
        rows.append((name, ratio, dec_pw, total))
        del model

    print("| variant | ratio | dec_pw MAC | delta dec_pw | full_conv MAC | delta full_conv |")
    print("|---|---:|---:|---:|---:|---:|")
    for name, ratio, dec_pw, total in rows:
        d_dec = 100.0 * (dec_pw / base_dec_pw - 1.0)
        d_total = 100.0 * (total / base_total - 1.0)
        print(
            f"| {name} | {ratio:.2f} | {dec_pw:,} | {d_dec:+.2f}% | "
            f"{total:,} | {d_total:+.2f}% |"
        )


if __name__ == "__main__":
    main()
