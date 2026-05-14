# SPEC: MBSTOI / Metric Computation Parallelization

日期：2026-05-14
Base：`09_weight_tied_mha/`（master 分支最新狀態，softmax RTL 改動之後）

## 1. Goal

把 `eval.py` 中的 per-pair metric 計算（MBSTOI、PESQ、SegSNR、ILD、IPD）從現行的單一進程序列執行，改為 CPU multi-process 平行執行。GPU 推論部分維持單進程不變。

目標：完整 750-pair × 8-SNR VCTK paper-style eval 從目前 ~3 小時降到 ~10–20 分鐘。

## 2. Motivation

目前 `safe_evaluate_file` 每個 pair 做完 GPU model inference 後，所有 metric（特別是 MBSTOI EC 階段的雙重 Python 迴圈、PESQ）都在主進程 CPU 上順序執行。這台機器 224 cores，目前只用 1 顆。MBSTOI 是純 CPU、per-pair 獨立、numpy-only，可以無數值風險地丟給 process pool。

## 3. Scope

### In-scope
- 改 `09_weight_tied_mha/eval.py`：拆分 `safe_evaluate_file` 為 GPU 階段（保留 main process）+ CPU metric 階段（移到 worker）
- 加 CLI flag `--metric_workers`（int，預設 1 = legacy 行為）
- 新增小型 unit test / acceptance script 驗證並行結果與序列結果在 tolerance 內一致

### Out-of-scope
- **不要動** `MBSTOI/` 任何檔案（含 ec.py、mbstoi.py、stft.py）。重寫 MBSTOI 是另一個工作項，本 spec 不處理。
- 不要動 `DCNN/` 任何模型程式碼。
- 不要動其他變體（`BCCTN/`、`BCCTN_sep_conv/`、…）的 `eval.py`。本 spec 只改 `09_weight_tied_mha/eval.py`。
- 不要改 default 行為。所有現有 shell scripts（含 `softmax_rtl_*/run_eval_*.sh`、`run_eval.sh`）若不傳 `--metric_workers`，輸出數值與 wallclock 行為應與 patch 前一致。
- 不要改 output file 格式（`detailed_results.csv`、`average_results.txt`、`vctk_paper_style_results_averages.txt` 的欄位與排版維持不變）。

## 4. Target file & exact change locations

`09_weight_tied_mha/eval.py`

- `safe_evaluate_file(clean_path, noisy_path, model, device)`（約 L458–L628）  
  需要拆成兩段：
  1. **GPU stage**：file I/O、shape/silence check、resample、model inference → 回傳一個 `dict`，含 `clean_data` (N,2)、`noisy_data` (N,2)、`enhanced_data` (2,N)、`sr`、`filename` 等資料。
  2. **CPU metric stage**：純 numpy / scipy / MBSTOI / pesq 計算，吃 GPU stage 的 dict，回傳目前 `safe_evaluate_file` 的 result dict。

- 主迴圈（約 L756–L778）：  
  改成 producer-consumer：主進程做 GPU stage，把 metric stage 的工作 submit 到 `concurrent.futures.ProcessPoolExecutor`。處理完成的 future 收進 `results`。`tqdm` 進度條依 future 完成數更新。

- `argparse` 區段（L1050 附近）：加 `--metric_workers` flag。

- 不需要改 `evaluate_at_paper_snrs`、`compute_..._error_with_mask`、`safe_mbstoi`、`safe_pesq` 的內部實作。

## 5. API contract

### 5.1 拆分後的兩個函式

請至少新增兩個函式，命名與簽章如下：

```python
def gpu_prepare_pair(clean_path, noisy_path, model, device, save_audio=False, output_dir=None):
    """Run file I/O + model inference. Returns a *picklable* dict with:
        clean_data    : ndarray (N, 2) float32
        noisy_data    : ndarray (N, 2) float32
        enhanced_data : ndarray (2, N) float32
        sr            : int
        filename      : str
        enhanced_path : Optional[str]  # only if save_audio
    or None if the pair should be skipped (mono / too short / too quiet / load error).
    Model and device live in the main process only; the returned dict must not
    contain torch tensors or any unpicklable object.
    """

def compute_metrics_for_pair(pair_payload):
    """Pure-CPU metric computation. Runs inside ProcessPoolExecutor workers.
    Input: dict returned by gpu_prepare_pair (or its 'metric_inputs' sub-dict).
    Returns: dict with the same keys safe_evaluate_file currently returns
        (filename, MBSTOI_noisy, MBSTOI, PESQ_noisy, PESQ,
         SegSNR_noisy_*, SegSNR_*, SegSNR_impr_*,
         ILD_error_noisy, ILD_error, IPD_error_noisy, IPD_error,
         IPD_error_noisy_deg, IPD_error_deg).
    Per-metric errors must be caught and the failing metric set to nan in the
    result, matching legacy behavior (do NOT let one bad metric kill the pair).
    """
```

注意陣列形狀：clean/noisy 是 `(N, 2)`，enhanced 是 `(2, N)`，**不要對調**。所有 metric 呼叫的索引（`[:, 0]` vs `[0]`）必須與目前 `safe_evaluate_file` 完全一致。

### 5.2 重構後的 `safe_evaluate_file`

保留這個函式（單進程路徑會用到），實作改為：
```python
def safe_evaluate_file(clean_path, noisy_path, model, device, save_audio=False, output_dir=None):
    payload = gpu_prepare_pair(clean_path, noisy_path, model, device, save_audio, output_dir)
    if payload is None:
        return None
    return compute_metrics_for_pair(payload)
```

這樣 `--metric_workers=1`（預設）走原本路徑，數值與行為完全等價。

## 6. Concurrency model

`evaluate_paper_style(...)` 主迴圈：

```python
if metric_workers <= 1:
    # legacy sequential path — must be numerically identical to today
    for clean_path, noisy_path in pairs:
        pair_results = safe_evaluate_file(clean_path, noisy_path, model, device, save_audio, output_dir)
        ...
else:
    from concurrent.futures import ProcessPoolExecutor, as_completed
    ctx = multiprocessing.get_context("spawn")
    executor = ProcessPoolExecutor(max_workers=metric_workers, mp_context=ctx,
                                   initializer=_worker_init)
    inflight = {}
    pair_iter = iter(pairs)
    # Keep at most 2 * metric_workers futures pending to bound memory
    for _ in range(min(2 * metric_workers, total_pairs)):
        ...   # submit next pair
    while inflight:
        for fut in as_completed(list(inflight)):
            ...   # collect result, update pbar, submit next
```

要求：

- **必須用 `spawn` context**（不用 fork）。理由：避免 fork 後子進程繼承 CUDA context / OMP 狀態造成 deadlock。
- **必須加 `_worker_init`** 在 worker 啟動時設定 BLAS 單執行緒：
  ```python
  def _worker_init():
      os.environ["OMP_NUM_THREADS"] = "1"
      os.environ["MKL_NUM_THREADS"] = "1"
      os.environ["OPENBLAS_NUM_THREADS"] = "1"
      os.environ["NUMEXPR_NUM_THREADS"] = "1"
  ```
  （另外 `eval.py` 開頭也加同樣四個 env 設定，避免 `import numpy` 時就被綁定。設環境變數 + `try: import threadpoolctl; threadpoolctl.threadpool_limits(1)` 雙保險。`threadpoolctl` 若 import 失敗只 warning，不要 raise。）
- **In-flight 上限 = `2 * metric_workers`**，避免主進程一口氣 submit 全部 750 個 future 把 audio 全部塞進 queue，記憶體爆掉。
- **tqdm 由 future 完成事件驅動更新**，不要在 submit 時更新。
- 主進程在主迴圈中保留 GPU inference；GPU 不會閒置（一邊跑下一個 pair 的 model，一邊 worker 在算前一個 pair 的 MBSTOI）。

### Memory budget sanity check
- 每個 pair 的 (clean+noisy+enhanced) 約 3s × 16kHz × 2ch × 3 versions × 4B = ~1.2 MB
- in-flight 上限 = 2 × metric_workers，例如 64 workers → 128 pairs × 1.2 MB ≈ 150 MB。OK。

## 7. CLI

在 `eval.py` 的 `argparse` 加：
```python
parser.add_argument("--metric_workers", type=int, default=1,
                    help="Parallel CPU workers for metric computation. "
                         "1 = legacy single-process. Recommended: 32–64 on this machine.")
```

並把這個值一路傳進 `evaluate_at_paper_snrs` 與 `evaluate_paper_style`。

**不要動現有 default**：未傳 `--metric_workers` 時走 worker=1 序列路徑。

## 8. Numerical equivalence requirement

跑下列 acceptance test（spec 第 10 節）後，**所有 metric 在所有 SNR、所有 pair 必須與序列版完全 bit-exact**：

- MBSTOI / PESQ：因為演算法本身是 deterministic，且每個 pair 獨立，平行版只是把計算分散到多進程，**結果必須完全相等**（不是「在 tolerance 內接近」）。
- SegSNR / ILD / IPD：同上，bit-exact。
- 唯一允許不同的是 `tqdm` 的進度條順序 / `detailed_results.csv` 的列順序（因為 future as_completed 是亂序）。對於 `detailed_results.csv`，**請保留原始 pair 順序排序**（用 submit 順序 index 排），這樣輸出檔 byte-by-byte 一致。

## 9. Error handling

- 單一 pair 的 metric 計算 raise exception → log + 該 pair 整筆當 None（與 legacy `safe_evaluate_file` `except` 區段行為一致）。
- Worker process 死掉（OOM、segfault）→ 主迴圈須能繼續處理其他 pair，最後打印 `Files Processed: K / 750` 反映實際成功數。
- Ctrl-C → 必須能乾淨關閉 pool（用 `with ProcessPoolExecutor(...) as ex:` 或在 finally 裡 `executor.shutdown(cancel_futures=True)`）。

## 10. Acceptance criteria & test

### 10.1 Smoke test（必跑）

```bash
cd /raid/R12K41024/09_weight_tied_mha

# Reference run (legacy serial path)
LIMIT_PAIRS=30 BATCH_SIZE=8 bash softmax_rtl_shift_based/run_eval_shift_strict.sh \
    "" softmax_rtl_shift_based/results/_accept_serial 4

# Parallel run with the new flag — needs eval.py to accept --metric_workers
# (passed via the script's CLI; codex may need to add a passthrough env var
# METRIC_WORKERS in run_eval_shift_strict.sh, or call eval.py directly.)
LIMIT_PAIRS=30 BATCH_SIZE=8 METRIC_WORKERS=32 bash softmax_rtl_shift_based/run_eval_shift_strict.sh \
    "" softmax_rtl_shift_based/results/_accept_parallel 4
```

兩個輸出資料夾的 `vctk_paper_style_results_averages.txt` 與 `detailed_results.csv` 必須 byte-by-byte 相同。請在 PR/handoff 附 `diff` 結果。

### 10.2 速度驗證

跑 `LIMIT_PAIRS=100`，比對 wall-clock：
- `--metric_workers=1`：基準
- `--metric_workers=32`：應該至少快 8×
- `--metric_workers=64`：應該至少快 12×（若沒有，代表 GPU 端變成瓶頸，請在 handoff 報告中註記）

### 10.3 既有腳本回歸

```bash
# 確認預設行為未變
LIMIT_PAIRS=10 bash softmax_rtl_shift_based/run_eval_shift_strict.sh \
    "" softmax_rtl_shift_based/results/_regress 4
# 對比改動前的 git stash / commit，輸出 averages 應 bit-exact
```

### 10.4 編譯 sanity

```bash
python -m py_compile 09_weight_tied_mha/eval.py
```

## 11. Shell script 更新（可選但建議）

在 `softmax_rtl_shift_based/run_eval_shift_strict.sh` 與 `softmax_rtl_taylor_relu/run_eval_exp_free_strict.sh` 加：

```bash
METRIC_WORKERS="${METRIC_WORKERS:-1}"
...
python eval.py \
    ... \
    --metric_workers "$METRIC_WORKERS" \
    ...
```

預設 1（不影響 legacy），可由 `METRIC_WORKERS=32 bash run_eval_*.sh` opt-in。

## 12. Out of scope / future work

- 把 MBSTOI 本身（特別是 `ec.py` 的雙重迴圈）改為 numpy vectorized 或 numba/cython。這是下一個 spec。
- 用 PyTorch GPU 重寫整個 MBSTOI。風險較高，需獨立 spec + numerical tolerance test。
- 把 GPU inference batch 化（目前 batch_size 是 file 數而非 sample 數，inference 仍然是逐檔），這也屬另一個工作項。

## 13. Deliverables

1. Patched `09_weight_tied_mha/eval.py`
2. （可選）兩個 `run_eval_*.sh` 加 `METRIC_WORKERS` 環境變數 passthrough
3. Handoff `HANDOFF_MBSTOI_PARALLEL_<YYYY-MM-DD>.md`，內容必含：
   - Smoke test diff（10.1）的結果（請貼 `diff -r` 輸出證明一致）
   - 速度比較表（10.2）
   - 主要改動程式碼位置（檔名 + 行號 range）
   - 已知 caveats（例如 worker 數過高時觀察到的瓶頸）
