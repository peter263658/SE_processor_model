# Handoff: MBSTOI / Metric Parallelization

Date: 2026-05-14
Target: `11_weight_tied_mha_fast_mbstoi/`
Spec source: `09_weight_tied_mha/SPEC_MBSTOI_PARALLEL_2026-05-14.md`

## Summary

- Split `eval.py` per-pair evaluation into:
  - `gpu_prepare_pair(...)`: file I/O, validation, resampling, model inference, optional enhanced-audio write.
  - `compute_metrics_for_pair(...)`: CPU-only MBSTOI, PESQ, SegSNR, ILD, IPD metrics.
- Added `--metric_workers` CLI flag. Default remains `1`, which uses the inline legacy-compatible path.
- Added `spawn`-based `ProcessPoolExecutor` for `metric_workers > 1`.
- Limits pending metric work to `2 * metric_workers`.
- Keeps `detailed_results.csv` rows in original pair order.
- Added BLAS/OpenMP single-thread limits at import time and in worker init.
- Added `METRIC_WORKERS` passthrough to the two strict softmax RTL eval scripts.
- Kept `sf.read(...)` default dtype for clean/noisy inputs, matching the original legacy path and avoiding float32 drift in CPU metric calculations. The model input tensor is still converted to float32 at inference time.

## Main Code Locations

- `eval.py:1-41`: thread env limits, `threadpoolctl`, process-pool imports.
- `eval.py:482-492`: `_worker_init`.
- `eval.py:495-574`: `gpu_prepare_pair`.
- `eval.py:577-722`: `compute_metrics_for_pair`.
- `eval.py:725-732`: refactored `safe_evaluate_file`.
- `eval.py:746-841`: parallel producer/consumer loop and worker-failure fallback.
- `eval.py:845-988`: `metric_workers` in `evaluate_model_improved`.
- `eval.py:1065-1158`: `metric_workers` pass-through for paper-style SNR eval.
- `eval.py:1266-1307`: CLI flag and dispatch.
- `softmax_rtl_shift_based/run_eval_shift_strict.sh:17-38`: `METRIC_WORKERS` passthrough.
- `softmax_rtl_taylor_relu/run_eval_exp_free_strict.sh:23-44`: `METRIC_WORKERS` passthrough.

## Verification Run

Syntax:

```bash
python -m py_compile 11_weight_tied_mha_fast_mbstoi/eval.py
```

CLI/import sanity in project env:

```bash
cd /raid/R12K41024/11_weight_tied_mha_fast_mbstoi
NUMBA_CACHE_DIR=/tmp/numba_cache conda run --no-capture-output -n bcctn \
  python eval.py --help
```

End-to-end 2-pair serial vs parallel sanity:

```bash
cd /raid/R12K41024/11_weight_tied_mha_fast_mbstoi

NUMBA_CACHE_DIR=/tmp/numba_cache CUDA_VISIBLE_DEVICES=0 \
conda run --no-capture-output -n bcctn python eval.py \
  --model_checkpoint outputs/finetune_strict/2026-05-10_02-40-16/logs/lightning_logs/version_0/checkpoints/epoch=45-step=44528.ckpt \
  --test_dataset_path /raid/R12K41024/BCCTN/Dataset/noisy_testset/snr_-3dB \
  --clean_dataset_path /raid/R12K41024/BCCTN/Dataset/clean_testset \
  --output_dir /tmp/mbstoi_serial_smoke2_11 \
  --specific_snr -3 \
  --batch_size 1 \
  --limit_pairs 2 \
  --metric_workers 1 \
  --hydra_overrides training.qat_scope=full training.top_k_attention=4 model.attention_softmax.mode=shift

NUMBA_CACHE_DIR=/tmp/numba_cache CUDA_VISIBLE_DEVICES=0 \
conda run --no-capture-output -n bcctn python eval.py \
  --model_checkpoint outputs/finetune_strict/2026-05-10_02-40-16/logs/lightning_logs/version_0/checkpoints/epoch=45-step=44528.ckpt \
  --test_dataset_path /raid/R12K41024/BCCTN/Dataset/noisy_testset/snr_-3dB \
  --clean_dataset_path /raid/R12K41024/BCCTN/Dataset/clean_testset \
  --output_dir /tmp/mbstoi_parallel_smoke2_11 \
  --specific_snr -3 \
  --batch_size 1 \
  --limit_pairs 2 \
  --metric_workers 2 \
  --hydra_overrides training.qat_scope=full training.top_k_attention=4 model.attention_softmax.mode=shift
```

Diff results:

```bash
diff -u /tmp/mbstoi_serial_smoke2_11/detailed_results.csv \
        /tmp/mbstoi_parallel_smoke2_11/detailed_results.csv
diff -u /tmp/mbstoi_serial_smoke2_11/average_results.txt \
        /tmp/mbstoi_parallel_smoke2_11/average_results.txt
```

Both diffs produced no output, so the 2-pair serial and parallel outputs were byte-identical. The 2-row CSV also confirms result rows are emitted in original pair order.

## Caveats

- Full spec acceptance (`LIMIT_PAIRS=30` paper-style serial/parallel diff, plus 100-pair speed table) was not run in this handoff; only the 2-pair end-to-end sanity was run.
- The SPEC requested BLAS/OpenMP thread limits at `eval.py` import time, so `--metric_workers=1` also runs with those env vars set. This preserves numeric behavior but can change legacy wall-clock behavior versus a pre-patch process that imported numpy/librosa before the limits were set.
- If `BrokenProcessPool` fires, the remaining metric work intentionally falls back to sequential computation for correctness and completion. That can be slow on long 750-pair runs.
- Direct import with the system Python fails because that environment does not have numpy. Use the repo's `bcctn` conda env.
- In this shell, importing through `conda run` needed `NUMBA_CACHE_DIR=/tmp/numba_cache` to avoid a numba/librosa cache locator error. The eval scripts were not changed to set this automatically.
- The local shell prints an unrelated `/etc/profile.d/set_mig_gpu.sh` syntax warning before commands run.
