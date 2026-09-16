# RESULTS: FP32「原始 BCCTN」參考分數到底是哪個模型 — 身分確認與勘誤

Date: 2026-09-16（supersedes the "Original BCCTN reference (fft_len=512, embed_dim=512, 3,973,288)" label in [RESULTS_ARCH_PROGRESSION.md](RESULTS_ARCH_PROGRESSION.md) / `.zh.md`、`19_algo_final/docs/` 同名檔、`journal_sync_2026-07-07/PAPER_FEATURE_SUMMARY_2026-07-07*.md`、`paper_figures/CAPTIONS.md` fig13、`paper_figures/make_fp32_vs_deploy_fig.py`）
Base candidate: 見 §1 表
Handoff: [RESULTS_PARAM_REDUCTION_OFFICIAL_BASIS_2026-09-16.md](RESULTS_PARAM_REDUCTION_OFFICIAL_BASIS_2026-09-16.md)、[RESULTS_MBSTOI_STAGE_DROP_PER_SNR_2026-09-16.md](RESULTS_MBSTOI_STAGE_DROP_PER_SNR_2026-09-16.md)

## 0. TL;DR

repo 裡有兩個被當成「原始 BCCTN FP32」的分數，來自**兩個不同的自製架構**，而且都**不是**文件標的 `fft 512 / embed 512 / 3,973,288`，更不是官方 11.09 M ckpt。

| 分數（VCTK / TIMIT） | 用在哪 | 真正的 ckpt | 架構 | 參數 |
|---|---|---|---|---|
| **0.8519 / 0.8114** | `BCCTN/results_May21/`、fig13、slide 的 FP32 baseline、RESULTS_ARCH_PROGRESSION、今天的兩份 RESULTS | `BCCTN/outputs/2025-06-02/00-25-51/.../epoch=137-step=133584.ckpt` | fft **256** / win 256 / hop 80，**5 個 stage** `[16,32,64,128,256]`，kernel (5,2)，Linear 1024→512 + MHA@512 + Linear 512→1024，32 heads | **3,806,868** |
| **0.8535 / 0.8134**（含 PESQ / STOI 欄） | `20_pesq_finetune/results/pesq_eval/fp32_512_ref_full/`、chip-paper 比較 | `BCCTN/outputs/2025-05-10/19-39-07/.../epoch=98-step=95832.ckpt` | fft **512** / win 400 / hop 100，6 stage，kernel (5,2)，同上 transformer | **4,791,448**（= 官方鏈的 T1 點） |
| （文件標籤）3,973,288 | — | `BCCTN/outputs/2025-05-11/16-34-17/`（(5,1) kernel） | fft 512，6 stage，kernel (5,1) | 3,973,288 — **有訓練、從未當參考分數評估過** |
| 官方 11.09 M | 只量過 64 檔 ILD/IPD | `BCCTN/DCNN/Checkpoints/Trained_model.ckpt` | MHA@1024 無 projection + Linear 1024²，(5,2) | 11,086,488 |

⭐⭐ 建議：投影片 / 論文的 FP32 參考改用 **0.8535 / 0.8134（4,791,448）**。理由：(1) 它是真正的 fft-512、6-stage 模型，和硬體版差的就是文件列的三步（FFT、attention、depthwise）+ kernel；(2) 它正好是官方鏈的 T1 點（官方 → 只換 transformer block），11.09 M → 4.79 M → 0.83 M 的參數故事和品質數字對得上同一個模型；(3) 它有 PESQ / STOI 欄，和 chip-paper 表一致。若要保留 0.8519，則參數標籤要改成 3.81 M、fft 256、5 stage。

## 1. 證據

### 1.1 `BCCTN/results_May21` = 2025-06-02 那次 eval

- `BCCTN/run_eval.sh` 的 `MODEL_CHECKPOINT` 指向 `outputs/2025-06-02/00-25-51/.../epoch=137-step=133584.ckpt`（註解 `en/decoder = 5 num_head = 32`），`OUTPUT_DIR=./results_May21`。
- 該 ckpt 寫入時間 2025-06-02 16:40；`results_May21/vctk_paper_style/` 下 VCTK `snr_9/12/15dB` 目錄 20:40–21:29（各 750 pairs）、TIMIT `snr_-6…15dB` 21:35–22:19（各 160 pairs）、`vctk_paper_style_results.csv` 22:19 — 同一次 run。
- VCTK `snr_-6…6dB` 目錄在 2025-06-03 03:09 被一次 `--limit_pairs 5` 的 smoke 覆寫（各 5 pairs），csv 未被覆寫；`run_eval.sh` 現在殘留的 `--limit_pairs 5` 就是那次。
- 該 run 的 `.hydra/config.yaml`：`fft_len 256, win_len 256, win_inc 80, kernel_num [16,32,64,128,256], embed_dim 512, num_heads 32`。ckpt 逐層：encoder 5 層 conv (5,2)、`input_proj (512,1024)`、MHA `in_proj (1536,512)`、`output_proj (1024,512)`；可訓練參數 3,806,868。
- 三組參考分數（0.8519、0.8535、deploy）的 `MBSTOI_noisy` 逐 SNR 相同（−6 dB = 0.5366），確認同一個 750-pair subset。

### 1.2 `fp32_512_ref_full` = 2025-05-10 那次訓練的 epoch 98

- `20_pesq_finetune/fp32_512_pesq_ft/run_queue_eval.sh` / `run_queue_phase2.sh`：`--model_checkpoint .../2025-05-10/19-39-07/.../epoch=98-step=95832.ckpt --limit_pairs 750`；結果 2026-07-09。
- 同一個 ckpt 也被複製成 `20_pesq_finetune/fp32_512_pesq_ft/warmstart_fp32_512_ep98.ckpt`（PESQ finetune 的 warm start）。
- 參數 4,791,448：fft 512、6 stage、kernel (5,2)、projected transformer — 與 [RESULTS_PARAM_REDUCTION_OFFICIAL_BASIS_2026-09-16.md](RESULTS_PARAM_REDUCTION_OFFICIAL_BASIS_2026-09-16.md) §2 的 T1 列**逐 bit 相同的架構與參數量**。

### 1.3 3,973,288 從哪來

`BCCTN/config/model.yaml`（fft 512 / embed 512）配上**現在**的 `BCCTN/DCNN/models/model.py`（kernel 已改成 (5,1)）實例化出來的數字。對應的訓練 run 是 `outputs/2025-05-11/16-34-17/`（ep94 / ep99，3,973,288），但沒有任何參考分數是用它評的。

## 2. 影響範圍（需要更正的地方）

| 位置 | 目前寫法 | 應改為 |
|---|---|---|
| `RESULTS_ARCH_PROGRESSION.md` / `.zh.md`（18、19 兩份）Original BCCTN reference 列、Headline、參數曲線第一點 | fft 512 / embed 512，3,973,288，quality 0.8519 / 0.8114 | 品質 0.8519 對應 3,806,868（fft 256、5 stage）；或改用 0.8535 / 0.8134 對應 4,791,448 |
| `paper_figures/CAPTIONS.md` fig13、`make_fp32_vs_deploy_fig.py` docstring 與圖例 "Original BCCTN FP32 (3.97 M)"、`make_perf_loss_slide_fig.py` | 同上 | 同上（圖要重產） |
| `journal_sync_2026-07-07/PAPER_FEATURE_SUMMARY_2026-07-07*.md`、`FEATURES_en.md` "3,973,288 → 337,876 (−91.5%)" | 3.97 M 起點 | 依選定的參考改：4.79 M → 338 K（−92.9%）或 3.81 M → 338 K（−91.1%）；官方基準 11.09 M → 338 K（−97.0%） |
| 今天的 [RESULTS_MBSTOI_STAGE_DROP_PER_SNR_2026-09-16.md](RESULTS_MBSTOI_STAGE_DROP_PER_SNR_2026-09-16.md) row 0、§2.2 欄名；[RESULTS_PARAM_REDUCTION_OFFICIAL_BASIS_2026-09-16.md](RESULTS_PARAM_REDUCTION_OFFICIAL_BASIS_2026-09-16.md) §5 | 「0.8519 是 3.97 M 重現版」 | 見各檔的 Erratum 節 |

A 區段（downsizing）的 MBSTOI 掉分若改用 0.8535 / 0.8134 當起點：VCTK −0.0136、TIMIT −0.0168（原本 −0.0120 / −0.0148）；總掉分 −0.0178 / −0.0236。

## 3. Reproduction

```bash
cd /raid/R12K41024/18_w4a8_pow2_mixed_lmul_hw_handoff_20260530
CUDA_VISIBLE_DEVICES= ~/miniforge3/envs/bcctn/bin/python scripts/identify_fp32_reference_ckpts.py
# eval provenance
grep -E '^MODEL_CHECKPOINT=|OUTPUT_DIR=' /raid/R12K41024/BCCTN/run_eval.sh
grep -nE 'model_checkpoint|output_dir' /raid/R12K41024/20_pesq_finetune/fp32_512_pesq_ft/run_queue_eval.sh
find /raid/R12K41024/BCCTN/results_May21 -name detailed_results.csv -printf '%TY-%Tm-%Td %TH:%TM %p\n' | sort
```
