"""Paper-style evaluation of a trained LBCCN checkpoint.

Reuses BCCTN/eval.py's metric functions (MBSTOI, STOI, PESQ wb/nb, SegSNR,
masked ILD/IPD errors) verbatim so numbers are directly comparable to every
BCCTN result table in this repo. Only the model loading differs: LBCCN is
wrapped so `model(noisy) -> enhanced` like BCCTN.

Usage:
  python eval.py --model_checkpoint ../logs/lbccn/version_N/checkpoints/best.ckpt \
      --output_dir ../results --paper_style_eval
"""
import argparse
import importlib.util
import os
import re
import sys
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

BCCTN_DIR = "/raid/R12K41024/BCCTN"
sys.path.insert(0, BCCTN_DIR)  # for `from MBSTOI.mbstoi import mbstoi`
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

_spec = importlib.util.spec_from_file_location(
    "bcctn_eval", os.path.join(BCCTN_DIR, "eval.py"))
bcctn_eval = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(bcctn_eval)

from train import LBCCNModule  # noqa: E402


class LBCCNWrapper(torch.nn.Module):
    """Adapts LBCCN's 6-tuple output to BCCTN's `model(x) -> enhanced` API."""

    def __init__(self, lbccn_model):
        super().__init__()
        self.lbccn = lbccn_model

    def forward(self, x):
        voice, _noise, _vlc, _vrc, _nlc, _nrc = self.lbccn(x)
        return voice


def load_model(checkpoint_path, device):
    module = LBCCNModule.load_from_checkpoint(checkpoint_path,
                                              map_location=device)
    model = LBCCNWrapper(module.model).to(device)
    model.eval()
    return model


# --- multiprocessing worker: model is tiny (38 K), so each worker loads its
# --- own CPU copy and runs the SAME bcctn_eval.safe_evaluate_file per pair.
_WORKER_MODEL = None


def _worker_init(checkpoint_path):
    global _WORKER_MODEL
    torch.set_num_threads(1)
    _WORKER_MODEL = load_model(checkpoint_path, torch.device("cpu"))


def _worker_eval(pair):
    clean_path, noisy_path = pair
    return bcctn_eval.safe_evaluate_file(
        clean_path, noisy_path, _WORKER_MODEL, torch.device("cpu"))


def evaluate_model(model, device, test_dataset_path, clean_dataset_path,
                   output_dir, specific_snr=None, limit_pairs=None,
                   pool=None):
    """BCCTN/eval.py::evaluate_model_improved with the model passed in.

    If `pool` is given, per-pair evaluation fans out to CPU workers (same
    per-file function, so numbers are identical to the serial path).
    """
    os.makedirs(output_dir, exist_ok=True)

    clean_files = list(Path(clean_dataset_path).glob('**/*.wav'))
    noisy_files = list(Path(test_dataset_path).glob('**/*.wav'))
    print(f"Found {len(clean_files)} clean files and {len(noisy_files)} noisy files")

    def parse_tag(stem):
        m = re.search(r'_az(-?\d+)(?:_snr([-+]?\d+(?:\.\d+)?))?$', stem)
        az = int(m.group(1)) if m else None
        snr = float(m.group(2)) if m and m.group(2) else None
        base = stem.split('_az')[0]
        return base, az, snr

    clean_dict = {}
    for f in clean_files:
        base, az, _ = parse_tag(f.stem)
        if az is not None:
            clean_dict[(base, az)] = f

    noisy_dict = {}
    for f in noisy_files:
        base, az, snr = parse_tag(f.stem)
        if az is None:
            continue
        if specific_snr is not None:
            if snr is None or abs(float(snr) - float(specific_snr)) > 0.1:
                continue
        noisy_dict[(base, az, snr)] = f

    pairs = [(clean_dict[(base, az)], noisy_dict[(base, az, snr)])
             for (base, az, snr) in noisy_dict.keys() if (base, az) in clean_dict]
    pairs.sort(key=lambda x: str(x[0]))

    if limit_pairs is not None and 0 < limit_pairs < len(pairs):
        print(f"Limiting evaluation to {limit_pairs} pairs (out of {len(pairs)})")
        pairs = pairs[:limit_pairs]
    print(f"Found {len(pairs)} matching clean/noisy pairs (azimuth-aligned)")
    if not pairs:
        print("ERROR: No matching clean/noisy pairs found!")
        return None

    metric_keys = [
        'SegSNR_noisy_L', 'SegSNR_noisy_R', 'SegSNR_noisy_Avg',
        'SegSNR_L', 'SegSNR_R', 'SegSNR_Avg',
        'SegSNR_impr_L', 'SegSNR_impr_R', 'SegSNR_impr_Avg',
        'MBSTOI_noisy', 'MBSTOI', 'STOI_noisy', 'STOI',
        'PESQ_noisy', 'PESQ', 'PESQ_nb_noisy', 'PESQ_nb',
        'ILD_error_noisy', 'ILD_error',
        'IPD_error_noisy', 'IPD_error',
        'IPD_error_noisy_deg', 'IPD_error_deg',
    ]
    results = {'filename': []}
    results.update({k: [] for k in metric_keys})

    if pool is not None:
        pair_results_iter = tqdm(pool.imap(_worker_eval, pairs, chunksize=4),
                                 total=len(pairs), desc="Processing files")
    else:
        pair_results_iter = (
            bcctn_eval.safe_evaluate_file(clean_path, noisy_path, model, device)
            for clean_path, noisy_path in tqdm(pairs, desc="Processing files"))

    for pair_results in pair_results_iter:
        if pair_results:
            for key in results.keys():
                if key in pair_results:
                    results[key].append(pair_results[key])
                elif key != 'filename':
                    results[key].append(float('nan'))

    if not results['filename']:
        print("Warning: no files successfully processed!")
        return None

    import pandas as pd
    results_df = pd.DataFrame(results)
    results_df.to_csv(os.path.join(output_dir, "detailed_results.csv"),
                      index=False)

    average_results = {
        'Average SegSNR Noisy (L)': np.nanmean(results_df['SegSNR_noisy_L']),
        'Average SegSNR Noisy (R)': np.nanmean(results_df['SegSNR_noisy_R']),
        'Average SegSNR Noisy': np.nanmean(results_df['SegSNR_noisy_Avg']),
        'Average MBSTOI Noisy': np.nanmean(results_df['MBSTOI_noisy']),
        'Average STOI Noisy': np.nanmean(results_df['STOI_noisy']),
        'Average PESQ Noisy': np.nanmean(results_df['PESQ_noisy']),
        'Average PESQ (nb) Noisy': np.nanmean(results_df['PESQ_nb_noisy']),
        'Average ILD Error Noisy (dB)': np.nanmean(results_df['ILD_error_noisy']),
        'Average IPD Error Noisy (rad)': np.nanmean(results_df['IPD_error_noisy']),
        'Average IPD Error Noisy (deg)': np.nanmean(results_df['IPD_error_noisy_deg']),
        'Average SegSNR (L)': np.nanmean(results_df['SegSNR_L']),
        'Average SegSNR (R)': np.nanmean(results_df['SegSNR_R']),
        'Average SegSNR': np.nanmean(results_df['SegSNR_Avg']),
        'Average SegSNR Improvement': np.nanmean(results_df['SegSNR_impr_Avg']),
        'Average MBSTOI': np.nanmean(results_df['MBSTOI']),
        'Average STOI': np.nanmean(results_df['STOI']),
        'Average PESQ': np.nanmean(results_df['PESQ']),
        'Average PESQ (nb)': np.nanmean(results_df['PESQ_nb']),
        'Average ILD Error (dB)': np.nanmean(results_df['ILD_error']),
        'Average IPD Error (rad)': np.nanmean(results_df['IPD_error']),
        'Average IPD Error (deg)': np.nanmean(results_df['IPD_error_deg']),
        'Files Processed': len(results_df),
        'Total Files': len(pairs),
        'Processing Rate': f"{len(results_df) / len(pairs) * 100:.1f}%",
    }

    with open(os.path.join(output_dir, "average_results.txt"), 'w') as f:
        for metric, value in average_results.items():
            line = (f"{metric}: {value:.4f}" if isinstance(value, float)
                    else f"{metric}: {value}")
            f.write(line + "\n")
            print(line)

    return average_results


def evaluate_at_paper_snrs(model, device, test_dir, clean_dir, output_dir,
                           dataset_name, limit_pairs=750, pool=None):
    paper_snrs = [-6, -3, 0, 3, 6, 9, 12, 15]
    results_by_snr = {}
    ds_output_dir = os.path.join(output_dir, dataset_name.lower())
    os.makedirs(ds_output_dir, exist_ok=True)

    for snr in paper_snrs:
        print(f"\n===== Evaluating {dataset_name} at SNR = {snr} dB =====")
        snr_test_dir = os.path.join(test_dir, f"snr_{snr}dB")
        if not os.path.exists(snr_test_dir):
            print(f"SNR subdirectory not found: {snr_test_dir}; "
                  f"using flat structure with specific_snr")
            snr_test_dir = test_dir
        snr_results = evaluate_model(
            model, device, snr_test_dir, clean_dir,
            os.path.join(ds_output_dir, f"snr_{snr}dB"),
            specific_snr=snr, limit_pairs=limit_pairs, pool=pool)
        if snr_results:
            results_by_snr[snr] = snr_results

    bcctn_eval.generate_paper_style_table(
        results_by_snr,
        os.path.join(output_dir, f"{dataset_name.lower()}_paper_style_results.csv"),
        dataset_name)
    return results_by_snr


def main():
    root = "/raid/R12K41024/BCCTN/Dataset"
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model_checkpoint", required=True)
    p.add_argument("--output_dir", required=True)
    p.add_argument("--paper_style_eval", action="store_true")
    p.add_argument("--vctk_test_dir", default=f"{root}/noisy_testset")
    p.add_argument("--vctk_clean_dir", default=f"{root}/clean_testset")
    p.add_argument("--timit_test_dir", default=f"{root}/noisy_testset_timit")
    p.add_argument("--timit_clean_dir", default=f"{root}/clean_testset_timit")
    p.add_argument("--test_dataset_path")
    p.add_argument("--clean_dataset_path")
    p.add_argument("--specific_snr", type=float)
    p.add_argument("--limit_pairs", type=int, default=750)
    p.add_argument("--n_workers", type=int, default=1,
                   help=">1 fans per-pair metric computation out to CPU "
                        "workers (model is 38 K, CPU inference is cheap); "
                        "identical numbers, much faster wall-clock")
    args = p.parse_args()

    pool = None
    if args.n_workers > 1:
        import multiprocessing as mp
        device = torch.device("cpu")
        model = None
        print(f"Evaluating with {args.n_workers} CPU workers")
        pool = mp.get_context("spawn").Pool(
            args.n_workers, initializer=_worker_init,
            initargs=(args.model_checkpoint,))
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Evaluating on {device}")
        model = load_model(args.model_checkpoint, device)
        n_params = sum(x.numel() for x in model.parameters())
        print(f"LBCCN parameters: {n_params/1e3:.1f} K")

    try:
        if args.paper_style_eval:
            evaluate_at_paper_snrs(model, device, args.vctk_test_dir,
                                   args.vctk_clean_dir, args.output_dir,
                                   "VCTK", limit_pairs=args.limit_pairs,
                                   pool=pool)
            if os.path.isdir(args.timit_test_dir):
                evaluate_at_paper_snrs(model, device, args.timit_test_dir,
                                       args.timit_clean_dir, args.output_dir,
                                       "TIMIT", limit_pairs=args.limit_pairs,
                                       pool=pool)
            else:
                print(f"TIMIT test dir not found ({args.timit_test_dir}); "
                      f"skipped")
        else:
            if not (args.test_dataset_path and args.clean_dataset_path):
                p.error("basic evaluation requires --test_dataset_path and "
                        "--clean_dataset_path")
            evaluate_model(model, device, args.test_dataset_path,
                           args.clean_dataset_path, args.output_dir,
                           specific_snr=args.specific_snr,
                           limit_pairs=args.limit_pairs, pool=pool)
    finally:
        if pool is not None:
            pool.close()
            pool.join()


if __name__ == "__main__":
    main()
