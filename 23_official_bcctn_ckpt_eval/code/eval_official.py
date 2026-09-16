#!/usr/bin/env python3
"""Paper-style evaluation of the OFFICIAL BCCTN checkpoint (VikasTokala/BCCTN, 11.09 M params).

The model is built with the official repository code itself (submodule ../official_bcctn, pinned)
so the architecture is exactly the released one: complex MHA directly at embed 1024 (no in/out
projection) + complex Linear 1024->1024, conv kernels (5,2), fft 512 / win 400 / hop 100.
Everything downstream — pair discovery, the 750-pair subset, GPU inference + CPU metric-worker
pool, MBSTOI/STOI/PESQ/SegSNR/ILD/IPD/ITD/IC, the paper-style tables — is dir 18's eval.py
(imported via the bcctn18_eval.py symlink), i.e. identical to every deploy-model result.

Usage (see ../run_eval.sh):
  CUDA_VISIBLE_DEVICES=1 python eval_official.py --output_dir ../results/official_ckpt_full \
      --vctk_test_dir ... --vctk_clean_dir ... [--timit_test_dir ... --timit_clean_dir ...] \
      --limit_pairs 750 --metric_workers 32
"""
import argparse
import hashlib
import os
import sys

# the official repo tracks its own __pycache__/*.pyc; never rewrite them (keeps the submodule clean)
sys.dont_write_bytecode = True

HERE = os.path.dirname(os.path.abspath(__file__))
OFFICIAL_DIR = os.path.normpath(os.path.join(HERE, "..", "official_bcctn"))
DIR18_CODE = os.path.normpath(os.path.join(HERE, "..", "..",
                              "18_w4a8_pow2_mixed_lmul_hw_handoff_20260530", "code"))
DEFAULT_CKPT = os.path.join(OFFICIAL_DIR, "DCNN", "Checkpoints", "Trained_model.ckpt")
OFFICIAL_MD5 = "991c7096f84d8b8dbf7d591277ba66ea"  # == /raid/R12K41024/BCCTN/DCNN/Checkpoints/Trained_model.ckpt
# official config/model.yaml verbatim; every other hyper-parameter is DCNN.__init__'s default
OFFICIAL_MODEL_CFG = dict(binaural=True, attention=True, use_clstm=True, use_cbn=True)
PAPER_SNRS = [-6, -3, 0, 3, 6, 9, 12, 15]


def _purge_dcnn_modules():
    for m in [m for m in sys.modules if m == "DCNN" or m.startswith("DCNN.")]:
        del sys.modules[m]


def build_official_model(ckpt_path, device):
    """Instantiate BinauralAttentionDCNN from the official code and load the checkpoint strictly."""
    import torch
    md5 = hashlib.md5(open(ckpt_path, "rb").read()).hexdigest()
    print(f"[official] ckpt {ckpt_path}\n[official] md5 {md5} ({'matches' if md5 == OFFICIAL_MD5 else 'DIFFERS FROM'} GitHub copy)")
    sys.path.insert(0, OFFICIAL_DIR)
    _purge_dcnn_modules()
    from DCNN.models.binaural_attention_model import BinauralAttentionDCNN
    model = BinauralAttentionDCNN(**OFFICIAL_MODEL_CFG)
    ck = torch.load(ckpt_path, map_location="cpu")
    sd = {k[len("model."):]: v for k, v in ck["state_dict"].items() if k.startswith("model.")}
    model.load_state_dict(sd, strict=True)  # raises on any missing/unexpected key
    n_params = sum(p.numel() for p in model.parameters())
    print(f"[official] loaded strict=True: {n_params:,} params (numel; complex tensors count 1), epoch {ck.get('epoch')}")
    # detach the official package so nothing else resolves `DCNN` to it
    sys.path.remove(OFFICIAL_DIR)
    _purge_dcnn_modules()
    return model.to(device).eval()


def import_dir18_eval():
    sys.path.insert(0, DIR18_CODE)   # for its `from MBSTOI...` imports; inherited by spawned workers
    sys.path.insert(0, HERE)         # bcctn18_eval.py symlink -> dir-18 eval.py
    import bcctn18_eval
    return bcctn18_eval


def evaluate_dataset(ev, model, device, name, test_dir, clean_dir, output_dir, limit_pairs, metric_workers):
    """Mirror of dir-18 evaluate_at_paper_snrs, with the model object passed in."""
    results_by_snr = {}
    ds_out = os.path.join(output_dir, name.lower())
    os.makedirs(ds_out, exist_ok=True)
    print(f"\n===== EVALUATING {name} DATASET =====")
    for snr in PAPER_SNRS:
        print(f"\n===== Evaluating {name} at SNR = {snr} dB =====")
        snr_test_dir = os.path.join(test_dir, f"snr_{snr}dB")
        if not os.path.exists(snr_test_dir):
            print(f"SNR subdirectory not found: {snr_test_dir}\nUsing flat structure with specific_snr parameter...")
            snr_test_dir = test_dir
        res = ev.evaluate_model_object(
            model, device, snr_test_dir, clean_dir, os.path.join(ds_out, f"snr_{snr}dB"),
            specific_snr=snr, batch_size=8, limit_pairs=limit_pairs, save_audio=False,
            metric_workers=metric_workers)
        if res:
            results_by_snr[snr] = res
    ev.generate_paper_style_table(results_by_snr,
                                  os.path.join(output_dir, f"{name.lower()}_paper_style_results.csv"), name)
    return results_by_snr


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model_checkpoint", default=DEFAULT_CKPT)
    ap.add_argument("--vctk_test_dir", required=True)
    ap.add_argument("--vctk_clean_dir", required=True)
    ap.add_argument("--timit_test_dir", default=None)
    ap.add_argument("--timit_clean_dir", default=None)
    ap.add_argument("--output_dir", required=True)
    ap.add_argument("--limit_pairs", type=int, default=750)
    ap.add_argument("--metric_workers", type=int, default=32)
    args = ap.parse_args()

    import torch
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[official] device {device}")
    model = build_official_model(args.model_checkpoint, device)
    ev = import_dir18_eval()
    os.makedirs(args.output_dir, exist_ok=True)

    evaluate_dataset(ev, model, device, "VCTK", args.vctk_test_dir, args.vctk_clean_dir,
                     args.output_dir, args.limit_pairs, args.metric_workers)
    if args.timit_test_dir and args.timit_clean_dir:
        evaluate_dataset(ev, model, device, "TIMIT", args.timit_test_dir, args.timit_clean_dir,
                         args.output_dir, args.limit_pairs, args.metric_workers)
    print(f"\n[official] done -> {args.output_dir}")


if __name__ == "__main__":
    main()
