#!/usr/bin/env python3
"""Identify which checkpoint/architecture produced each FP32 "original BCCTN" reference score.

Prints trainable-parameter count (BN running stats and loss buffers excluded), epoch, and the
architecture-defining tensor shapes for every candidate checkpoint, plus the Hydra config of its
training run when present. See docs/RESULTS_FP32_REFERENCE_IDENTITY_2026-09-16.md.

Run (CPU):  CUDA_VISIBLE_DEVICES= ~/miniforge3/envs/bcctn/bin/python scripts/identify_fp32_reference_ckpts.py
"""
import os, re, torch

B = "/raid/R12K41024/BCCTN"
CANDIDATES = [
    ("official VikasTokala/BCCTN Trained_model.ckpt", f"{B}/DCNN/Checkpoints/Trained_model.ckpt"),
    ("BCCTN/results_May21 (0.8519/0.8114) <- BCCTN/run_eval.sh MODEL_CHECKPOINT",
     f"{B}/outputs/2025-06-02/00-25-51/logs/lightning_logs/version_0/checkpoints/epoch=137-step=133584.ckpt"),
    ("20_pesq_finetune fp32_512_ref_full (0.8535/0.8134) <- run_queue_eval.sh",
     f"{B}/outputs/2025-05-10/19-39-07/logs/lightning_logs/version_0/checkpoints/epoch=98-step=95832.ckpt"),
    ("(5,1)-kernel fft-512 config = the 3,973,288 in RESULTS_ARCH_PROGRESSION (never evaluated as reference)",
     f"{B}/outputs/2025-05-11/16-34-17/logs/lightning_logs/version_0/checkpoints/epoch=94-step=91960.ckpt"),
]
BUF = ("running_mean", "running_var", "num_batches_tracked")
KEYS = ("fft_len", "win_len", "win_inc", "embed_dim", "num_heads", "kernel_num", "use_depthwise")


def hydra_cfg(ckpt):
    run = ckpt.split("/logs/lightning_logs/")[0]
    p = os.path.join(run, ".hydra", "config.yaml")
    if not os.path.exists(p):
        return None
    txt = open(p).read()
    out = {}
    for k in KEYS:
        m = re.search(rf"^\s*{k}:\s*(.*)$", txt, re.M)
        if m:
            out[k] = m.group(1).strip() or "(list)"
    m = re.search(r"kernel_num:\n((?:\s*- \d+\n)+)", txt)
    if m:
        out["kernel_num"] = [int(x) for x in re.findall(r"- (\d+)", m.group(1))]
    return out


for label, path in CANDIDATES:
    print(f"\n{label}\n  {path}")
    if not os.path.exists(path):
        print("  MISSING"); continue
    ck = torch.load(path, map_location="cpu")
    sd = ck.get("state_dict", ck)
    sd = {k[6:] if k.startswith("model.") else k: v for k, v in sd.items()
          if not k.startswith("loss") and not k.endswith(BUF)}
    enc = [k for k in sd if k.startswith("encoder") and sd[k].dim() == 4 and k.endswith("_r.weight")]
    inp = [k for k in sd if "input_proj" in k and k.endswith("weight")]
    mha = [k for k in sd if "in_proj_weight" in k]
    print(f"  params {sum(v.numel() for v in sd.values()):,}   epoch {ck.get('epoch')}")
    print(f"  encoder stages {len(enc)}  first conv {tuple(sd[enc[0]].shape)}  depthwise {any('depthwise' in k for k in sd)}")
    print(f"  input_proj {tuple(sd[inp[0]].shape) if inp else None}  MHA in_proj {tuple(sd[mha[0]].shape)}")
    print(f"  hydra: {hydra_cfg(path)}")
