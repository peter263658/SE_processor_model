#!/usr/bin/env python3
"""
Convert a baseline (untied) ComplexMHA checkpoint into a weight-tied
checkpoint usable by the 09_weight_tied_mha model.

Untied baseline has separate weights for `MultiheadAttention_re` and
`MultiheadAttention_im`. The tied model has only `MultiheadAttention_re`.
We initialise the tied weight as the average of the two, dropping the `_im`
keys. Averaging is a symmetric warm-start; fine-tuning will refine.

Usage:
    python checkpoint_surgery_for_tied.py \
        --src /raid/R12K41024/09_weight_tied_mha/epoch=134-step=130680.ckpt \
        --dst /raid/R12K41024/09_weight_tied_mha/baseline_tied_init.ckpt
"""
import argparse
import torch
from collections import OrderedDict
from pathlib import Path


MHA_KEY_FRAGMENTS = (
    ".MultiheadAttention_re.",
    ".MultiheadAttention_im.",
)


def average_mha_weights(state_dict):
    """For every key matching `mattn.MultiheadAttention_re.X`, average it with
    the corresponding `mattn.MultiheadAttention_im.X` and drop the `_im` key.
    """
    new_state = OrderedDict()
    averaged = []
    dropped_im = []
    re_only_kept = []

    # Index _im keys for fast lookup.
    im_keys = {k for k in state_dict if ".MultiheadAttention_im." in k}

    for k, v in state_dict.items():
        if ".MultiheadAttention_im." in k:
            dropped_im.append(k)
            continue

        if ".MultiheadAttention_re." in k:
            paired = k.replace(".MultiheadAttention_re.", ".MultiheadAttention_im.")
            if paired in state_dict:
                v_im = state_dict[paired]
                if v_im.shape != v.shape:
                    raise RuntimeError(
                        f"Shape mismatch averaging {k}: re={tuple(v.shape)} im={tuple(v_im.shape)}"
                    )
                new_state[k] = 0.5 * (v + v_im)
                averaged.append(k)
            else:
                new_state[k] = v
                re_only_kept.append(k)
            continue

        new_state[k] = v

    return new_state, averaged, dropped_im, re_only_kept


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True, type=Path)
    ap.add_argument("--dst", required=True, type=Path)
    args = ap.parse_args()

    print(f"Loading source checkpoint: {args.src}")
    ckpt = torch.load(args.src, map_location="cpu")

    if "state_dict" in ckpt:
        sd = ckpt["state_dict"]
        wrapped = True
    else:
        sd = ckpt
        wrapped = False

    new_sd, averaged, dropped_im, re_only_kept = average_mha_weights(sd)

    print(f"\n  Averaged MHA_re/MHA_im pairs : {len(averaged)} keys")
    for k in averaged:
        print(f"    avg({k},  → {k.replace('.MultiheadAttention_re.', '.MultiheadAttention_im.')})")
    print(f"\n  Dropped MHA_im keys           : {len(dropped_im)}")
    print(f"  Kept MHA_re-only keys (no _im pair): {len(re_only_kept)}")

    if wrapped:
        out = {**ckpt, "state_dict": new_sd}
    else:
        out = new_sd

    args.dst.parent.mkdir(parents=True, exist_ok=True)
    torch.save(out, args.dst)
    print(f"\nSaved tied-init checkpoint: {args.dst}")
    print(f"  Original keys : {len(sd)}")
    print(f"  New keys      : {len(new_sd)}")


if __name__ == "__main__":
    main()
