# RESULTS: 參數量 vs MBSTOI — 每一步化簡各一個點（投影片「Network Restructuring」改圖用）

Date: 2026-09-17
Base candidate: 見 §1 表（各點 ckpt）；終點 = `18_.../ckpt/strict_core_pow2_blockfloat_deploy_ep3_vctk0.8357_timit0.7898.ckpt`
Handoff: [RESULTS_FP32_REFERENCE_IDENTITY_2026-09-16.md](../../18_w4a8_pow2_mixed_lmul_hw_handoff_20260530/docs/RESULTS_FP32_REFERENCE_IDENTITY_2026-09-16.md)、[RESULTS_ARCH_PROGRESSION.md](../../18_w4a8_pow2_mixed_lmul_hw_handoff_20260530/docs/RESULTS_ARCH_PROGRESSION.md)

## 0. TL;DR

圖：`paper_figures/out/{svg,emf,png}/slide_param_vs_mbstoi.*`（`make_param_vs_mbstoi_fig.py`）。

| # | 步驟（照實際訓練順序） | 參數 | VCTK | TIMIT | 來源 |
|---|---|---:|---:|---:|---|
| 0 | Baseline：fft 512 / win 400 / hop 100，E 512 / 32 heads，full conv (5,1) causal | 3,973,288 | **0.8451** | **0.8032** | 新跑：`BCCTN/outputs/2025-05-11/16-34-17` ep94 |
| 1 | + depthwise-separable conv，fft 256 / hop 80 | 2,798,940 | **0.8398** | **0.7956** | 新跑：`BCCTN_sep_conv_causal/outputs/2025-06-12/16-10-22` ep159 |
| 2 | + fft 128（5 ms frame） | 2,536,540 | **0.8373** | **0.7955** | 新跑：`BCCTN_sep_conv_causal/outputs/2025-06-16/00-32-24` ep142 |
| 3 | + 32-frame causal window（參數不變） | 2,536,540 | **0.8327** | **0.7871** | 新跑：`BCCTN_sep_conv_causal_slide_win/outputs/2026-01-06/17-03-19` ep142 |
| 4 | + attention 256-d / 16 heads（+ FP8 QAT）= hardware-sized | 830,300 | 0.8399 | 0.7966 | `main_model/results_May21` |
| 5 | + tied MHA（09） | 567,132 | 0.8411 | 0.7966 | RESULTS_ARCH_PROGRESSION |
| 6 | + grouped decoder + top-50% skip（12） | 485,332 | 0.8352 | 0.7919 | 同上 |
| 7 | + grouped QKV = production（14） | 337,876 | 0.8354 | 0.7880 | 同上 |
| 8 | zero-multiplier W4A8 deploy（18） | 337,876 | 0.8357 | 0.7898 | 同上 |

投影片那頁的數字改成：**3.97 M → 830 K（−79%），MBSTOI 0.8451 → 0.8399（VCTK −0.62%）/ 0.8032 → 0.7966（TIMIT −0.82%）**。原本寫的「0.75%」就在這個量級。整條鏈 0 → 8：參數 −91.5%，MBSTOI −0.0094 / −0.0134。

## 1. 設定

- 四個新點用 `24_param_mbstoi_progression/code/eval_legacy_ckpt.py`：以各 run 自己的 Hydra config + 該 variant dir 的模型程式碼重建、`strict=True` 載入，再交給 dir-18 現行 `eval.py` 的 `evaluate_model_object`（同一個 750-pair × 8 SNR VCTK subset、160-pair TIMIT、32 metric worker、同一套 metric）。四個點的 `MBSTOI_noisy` 逐 SNR 與 deploy run 完全相同；MBSTOI 計算錯誤數皆為 32（資料集固有，與所有 run 相同）。
- 管線驗證：用同一 wrapper 重跑已知的 0.8519 模型（`BCCTN/outputs/2025-06-02` ep137），每檔 MBSTOI 與 `results_May21` 一致到 1e-7。
- 依使用者要求，full conv 一律 (5,1)（causal）；(5,2) 的 ckpt（4.79 M / 4.27 M）不是 causal，不列入。
- 投影片畫的順序（FFT → attention 維度 → depthwise）沒有對應的訓練點：fft 128 + full conv、fft 128 + E 256 + full conv 兩個模型從未訓練。這裡照實際順序標點。

## 2. 觀察

- 掉分最大的一步是 **32-frame window**（−0.0046 / −0.0084，參數不變），其次是 depthwise + fft 256（−0.0053 / −0.0076）；fft 256 → 128 幾乎不掉（−0.0025 / −0.0001）。
- attention 512 → 256 / 16 heads（步驟 4）反而 **+0.0072 / +0.0095**：830 K 那顆是 FP8 QAT、訓練到 ep134 的完整版，E 512 win31 的 FP32 點沒有同等訓練預算；這一步的「架構代價」實際上被訓練差異蓋過，投影片不要把它講成掉分。
- 830 K 之後（09 → 18）的點都在 0.835–0.841 之間，量化 + 結構壓縮的總代價 −0.0042 / −0.0068。

## 3. Reproduction

```bash
cd /raid/R12K41024/24_param_mbstoi_progression
GPU_ID=<free> bash run_missing_points.sh          # 4 points, ~15 min each, one GPU, sequential
cd /raid/R12K41024/paper_figures && CUDA_VISIBLE_DEVICES= ~/miniforge3/envs/bcctn/bin/python make_param_vs_mbstoi_fig.py
```
結果表副本：`docs/P*_{vctk,timit}_paper_style_results*.{csv,txt}`。

## 4. 追加（2026-09-17）：起點改為官方 11.09 M

圖的起點往前加兩個點，讓參數量與投影片引用的 11.09 M 一致：

| # | 步驟 | 參數 | Δ 參數 | VCTK | TIMIT | 來源 |
|---|---|---:|---:|---:|---:|---|
| 0 | 官方 BCCTN（complex MHA 直接在 1024 維，無 projection；+ Linear 1024²） | 11,086,488 | — | 0.84（引用） | — | 參數：官方 ckpt；MBSTOI：GAF-Net（arXiv 2509.14076）Table 1 的 BCCTN 列，**HUTUBS / NOISEX-92 測試集，與本 repo 不同**，圖上以灰色菱形獨立標示、不與我方曲線相連 |
| 1 | + attention input/output projection（1024→512→1024），非 causal (5,2) | 4,791,448 | −6,295,040（−56.8%） | 0.8535 | 0.8134 | `20_pesq_finetune/results/pesq_eval/fp32_512_ref_full` |
| 2 | + causal conv (5,2)→(5,1)（原 §0 的點 0） | 3,973,288 | −818,160 | 0.8451 | 0.8032 | 本文 §0 |

之後的點編號順延（原 1–8 → 3–10）。官方釋出權重在本 repo 測試集上為 0.7958 / 0.7247（out-of-domain，見 `23_official_bcctn_ckpt_eval`），不用於此圖。掉分計算的 FP32 起點 = 點 1（0.8535 / 0.8134）：到 deploy 共 −0.0178 / −0.0236，其中 causal 化一步 −0.0084 / −0.0102。
