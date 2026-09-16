#!/usr/bin/env python3
"""Paper-style eval of a legacy (pre-09) FP32 BCCTN checkpoint with the deploy-model pipeline.

The model is rebuilt from the training run's own Hydra config (`<run_dir>/.hydra/config.yaml`,
`model:` section) using the model code of a chosen variant dir (`--code_dir`), loaded with
strict=True, then handed to dir 18's evaluate_model_object (same 750-pair subset, metric
workers, metrics and paper-style tables as every deploy result).

  CUDA_VISIBLE_DEVICES=1 python eval_legacy_ckpt.py --code_dir /raid/R12K41024/main_model \
      --run_dir /raid/R12K41024/main_model/outputs/2025-06-16/00-32-24 \
      --ckpt .../epoch=98-step=95832.ckpt --output_dir ../results/P3_fft128_E512_dw \
      --vctk_test_dir ... --vctk_clean_dir ... [--timit_test_dir ... --timit_clean_dir ...]
"""
import argparse
import importlib
import os
import sys

import yaml

HERE = os.path.dirname(os.path.abspath(__file__))
DIR18_CODE = os.path.normpath(os.path.join(HERE, "..", "..", "18_w4a8_pow2_mixed_lmul_hw_handoff_20260530", "code"))
PAPER_SNRS = [-6, -3, 0, 3, 6, 9, 12, 15]
sys.dont_write_bytecode = True


def _purge_dcnn():
    for m in [m for m in sys.modules if m == "DCNN" or m.startswith("DCNN.")]:
        del sys.modules[m]


def build_legacy_model(code_dir, run_dir, ckpt, device, ignore_missing_prefixes=()):
    import torch
    cfg = yaml.safe_load(open(os.path.join(run_dir, ".hydra", "config.yaml")))["model"]
    print(f"[legacy] code_dir {code_dir}\n[legacy] run cfg: " + ", ".join(
        f"{k}={cfg[k]}" for k in ("fft_len", "win_len", "win_inc", "embed_dim", "num_heads", "kernel_num",
                                  "use_depthwise", "attn_win") if k in cfg))
    sys.path.insert(0, code_dir); _purge_dcnn()
    mod = importlib.import_module("DCNN.models.binaural_attention_model")
    model = mod.BinauralAttentionDCNN(**cfg)
    ck = torch.load(ckpt, map_location="cpu")
    sd = {k[len("model."):]: v for k, v in ck["state_dict"].items() if k.startswith("model.")}
    if ignore_missing_prefixes:
        # tolerate ONLY vestigial modules the code constructs but never calls (e.g. bcctn_backup_v1's
        # encoder_l/encoder_r); anything else missing or unexpected still raises
        missing, unexpected = model.load_state_dict(sd, strict=False)
        bad = [k for k in missing if not k.startswith(tuple(ignore_missing_prefixes))]
        if bad or unexpected:
            raise RuntimeError(f"state_dict mismatch: missing={bad} unexpected={unexpected}")
        print(f"[legacy] ignored {len(missing)} vestigial keys under {ignore_missing_prefixes}")
    else:
        model.load_state_dict(sd, strict=True)
    n = sum(p.numel() for n_, p in model.named_parameters() if not n_.startswith(tuple(ignore_missing_prefixes or ())))
    print(f"[legacy] loaded strict=True: {n:,} params, epoch {ck.get('epoch')}  <- {ckpt}")
    sys.path.remove(code_dir); _purge_dcnn()
    return model.to(device).eval(), n


def import_dir18_eval():
    sys.path.insert(0, DIR18_CODE); sys.path.insert(0, HERE)
    import bcctn18_eval
    return bcctn18_eval


def evaluate_dataset(ev, model, device, name, test_dir, clean_dir, output_dir, limit_pairs, metric_workers):
    results_by_snr = {}
    ds_out = os.path.join(output_dir, name.lower()); os.makedirs(ds_out, exist_ok=True)
    print(f"\n===== EVALUATING {name} DATASET =====")
    for snr in PAPER_SNRS:
        print(f"\n===== Evaluating {name} at SNR = {snr} dB =====")
        snr_test_dir = os.path.join(test_dir, f"snr_{snr}dB")
        if not os.path.exists(snr_test_dir):
            snr_test_dir = test_dir
        res = ev.evaluate_model_object(model, device, snr_test_dir, clean_dir, os.path.join(ds_out, f"snr_{snr}dB"),
                                       specific_snr=snr, batch_size=8, limit_pairs=limit_pairs, save_audio=False,
                                       metric_workers=metric_workers)
        if res:
            results_by_snr[snr] = res
    ev.generate_paper_style_table(results_by_snr, os.path.join(output_dir, f"{name.lower()}_paper_style_results.csv"), name)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--code_dir", required=True); ap.add_argument("--run_dir", required=True); ap.add_argument("--ckpt", required=True)
    ap.add_argument("--vctk_test_dir", required=True); ap.add_argument("--vctk_clean_dir", required=True)
    ap.add_argument("--timit_test_dir"); ap.add_argument("--timit_clean_dir")
    ap.add_argument("--output_dir", required=True); ap.add_argument("--limit_pairs", type=int, default=750)
    ap.add_argument("--metric_workers", type=int, default=32)
    ap.add_argument("--ignore_missing_prefixes", default="", help="comma-separated key prefixes of vestigial modules to leave uninitialised (must be unused in forward)")
    a = ap.parse_args()
    import torch
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, n = build_legacy_model(a.code_dir, a.run_dir, a.ckpt, device,
                                  tuple(x for x in a.ignore_missing_prefixes.split(',') if x))
    ev = import_dir18_eval()
    os.makedirs(a.output_dir, exist_ok=True)
    with open(os.path.join(a.output_dir, "model_info.txt"), "w") as f:
        f.write(f"code_dir {a.code_dir}\nrun_dir {a.run_dir}\nckpt {a.ckpt}\nparams {n}\n")
    evaluate_dataset(ev, model, device, "VCTK", a.vctk_test_dir, a.vctk_clean_dir, a.output_dir, a.limit_pairs, a.metric_workers)
    if a.timit_test_dir and a.timit_clean_dir:
        evaluate_dataset(ev, model, device, "TIMIT", a.timit_test_dir, a.timit_clean_dir, a.output_dir, a.limit_pairs, a.metric_workers)
    print(f"\n[legacy] done -> {a.output_dir}")


if __name__ == "__main__":
    main()
