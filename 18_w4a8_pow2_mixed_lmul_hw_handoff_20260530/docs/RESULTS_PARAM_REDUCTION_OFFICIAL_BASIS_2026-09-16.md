# RESULTS: 參數量逐步縮減 — 官方 BCCTN 11.09 M 基準 vs repo 重現版 3.97 M 基準

Date: 2026-09-16
Base candidate: `/raid/R12K41024/BCCTN/DCNN/Checkpoints/Trained_model.ckpt`（VikasTokala/BCCTN 官方 paper checkpoint）→ hardware-sized working point 830,300
Handoff: [RESULTS_ARCH_PROGRESSION.md](RESULTS_ARCH_PROGRESSION.md)（3.97 M 基準的階段表）、[RESULTS_MBSTOI_STAGE_DROP_PER_SNR_2026-09-16.md](RESULTS_MBSTOI_STAGE_DROP_PER_SNR_2026-09-16.md)

## 0. TL;DR

| 問題 | 答案 |
|---|---|
| 11.09 M 是 MAC 還是參數？ | **參數**。官方 ckpt 可訓練參數 **11,086,488**（ckpt 內 tensor 合計 11,093,257 = 參數 + 2,518 個 BN running-stat buffer + 4,251 個 loss 模組 STOI buffer）。LBCCN 論文（ICASSP'25）引用 BCCTN 為 **11.1 M params、12.7 G MACs**，兩者分開列。 |
| 官方拆解算不出來？ | 算得出來。官方 ckpt 就在 repo（`BCCTN/DCNN/Checkpoints/Trained_model.ckpt`），逐層 shape 直接數；中間點用 repo 程式碼實例化。腳本：`scripts/count_params_official_to_hw.py`。 |
| 11.09 M 和 3.97 M 差在哪？ | 兩個**架構上的差異**，不是設定值：(1) 官方 transformer 是 complex MHA 直接跑在 **embed 1024**（無 in/out projection）+ 一個實數 Linear 1024→1024；repo 重現版改成 Linear 1024→512 + complex MHA@512 + Linear 512→1024。(2) 官方 conv kernel 是 (5,2)；repo 是 (5,1)。這兩步合計 **−7.11 M**，佔 11.09 M→0.83 M 總省下量的 69%，而投影片/顧問表都沒列。 |
| ⭐⭐ 建議 | 投影片若要用 11.09 M，就用 §2 的 5 步表（或把 T1+C1 合併成「官方 → 重現版」一列）。若維持 3 步表，基準必須改成 3.97 M（§3），且顧問給的每步數字要換成 §3 的實算值。 |

## 1. 官方 ckpt 逐模組

| 模組 | 官方（可訓練參數） | 內容 |
|---|---:|---|
| Transformer：MHA_re + MHA_im | 8,396,800 | 各 `in_proj (3072×1024)` + `out_proj (1024×1024)`，embed 1024，無 input/output projection |
| Transformer：`transform` Linear | 1,049,600 | 實數 Linear 1024→1024 |
| Encoder | 547,708 | 6 層 complex conv，kernel (5,2)，channel 8/16/32/64/128/128（re、im 各一） |
| Decoder | 1,092,380 | 6 層 complex conv-transpose，kernel (5,2) |
| **合計** | **11,086,488** | |

（`main_model` 程式碼以 `use_depthwise=false` 實例化的 encoder = 547,708，與官方逐層 shape 完全一致，證明 (5,2) 路徑就是官方 conv。）

## 2. 官方基準：11,086,488 → 830,300（−92.5%）

| 步驟 | Transformer | Encoder | Decoder | 合計 | Δ 本步 | Δ / 原始 | 佔總省下量 |
|---|---:|---:|---:|---:|---:|---:|---:|
| S0 官方 ckpt：MHA@1024（無 proj）+ Linear 1024²，conv (5,2)，fft 512 | 9,446,400 | 547,708 | 1,092,380 | 11,086,488 | | | |
| T1 Transformer block 改設計：Linear 1024→512 + complex MHA@512 + Linear 512→1024 | 3,151,360 | 547,708 | 1,092,380 | 4,791,448 | −6,295,040 | −56.8% | 61.4% |
| C1 conv kernel (5,2) → (5,1)　［= repo 重現版 3,973,288］ | 3,151,360 | 274,988 | 546,940 | 3,973,288 | −818,160 | −7.4% | 8.0% |
| F　fft 512 → 128（win 400→128、hop 100→80）：transformer 輸入 1024 → 256 | 2,364,160 | 274,988 | 546,940 | 3,186,088 | −787,200 | −7.1% | 7.7% |
| A　embed 512 → 256、heads 32 → 16 | 657,920 | 274,988 | 546,940 | 1,479,848 | −1,706,240 | −15.4% | 16.6% |
| D　full conv → depthwise-separable (5,1) + pointwise　［= hardware-sized 830,300］ | 657,920 | 59,302 | 113,078 | 830,300 | −649,548 | −5.9% | 6.3% |

總省下量 10,256,188。三個「投影片步驟」F + A + D 只佔 30.6%（−3,142,988）；其餘 69.4% 來自 T1 + C1。

## 3. 重現版基準：3,973,288 → 830,300（−79.1%）— 顧問數字 vs 實算

顧問表把 FFT 記為 0、把 FFT 的效果併進 attention 那列、conv 的終點寫成 0.04 M。實算如下（順序同投影片：FFT → attention → conv）：

| 步驟 | 顧問給的 Δ | 實算 Δ | 實算後合計 | 說明 |
|---|---:|---:|---:|---|
| FFT 512 → 128（win/hop 跟著換） | 0 | **−787,200** | 3,186,088 | `flatten = (kernel_num[-1]/2) × (fft_len/128) × 2` → 1024 → 256，input_proj / output_proj 各縮 4×。**FFT 會影響參數**。 |
| Attention embed 512 → 256、heads 32 → 16 | −2.36 M | **−1,706,240** | 1,479,848 | MHA 2,101,248 → 526,336（÷4）；in/out proj 各再減半。顧問的 3.15 M → 0.79 M 是把 FFT 那段併進來，且 0.79 M 是錯的（working point transformer = 657,920）。 |
| full conv → depthwise-separable | −0.78 M | **−649,548** | 830,300 | encoder 274,988 → 59,302、decoder 546,940 → 113,078；conv 合計 821,928 → 172,380（不是 0.04 M）。 |
| 合計 | −3.14 M | −3,142,988 | | −79.1%，總數相同，但分配不同。 |

Heads 數不影響參數（只切分 embed）；sliding window、causal padding、hop 都不影響參數；`win_len` 也不影響（只有 `fft_len` 透過 flatten size 影響）。

## 4. 順序敏感度

F 與 A 透過 in/out projection 交互（projection 大小 = flatten × embed），所以先後順序會改變各自的 Δ：

| 順序 | 第一步 Δ | 第二步 Δ | 合計 |
|---|---:|---:|---:|
| F 再 A（投影片順序） | −787,200 | −1,706,240 | −2,493,440 |
| A 再 F | −2,099,456 | −393,984 | −2,493,440 |

T1、C1、D 與其他步驟無交互，Δ 不隨順序改變。投影片要標明採用的順序。

## 5. 未決事項

- repo 裡所有「原始 BCCTN FP32」的品質數字（VCTK 0.8519 / TIMIT 0.8114）都是 **3.97 M 重現版**的分數，不是 11.09 M 官方 ckpt 的分數。官方 ckpt 目前只在 64 檔 VCTK 上量過 ILD/IPD（[STATUS_CHIP_PAPER_COMPARISON_POSITIONING_2026-06-28.md](../../20_pesq_finetune/docs/STATUS_CHIP_PAPER_COMPARISON_POSITIONING_2026-06-28.md)）。若投影片同時放 11.09 M 與 0.8519，等於混用兩個模型；要嘛參數用 3.97 M，要嘛補跑官方 ckpt 的 full MBSTOI eval（需要能載入 embed-1024 無 projection 架構的程式碼，目前 repo 的 `DCNN.models.model` 不是這個架構）。

## 6. Reproduction

```bash
cd /raid/R12K41024/18_w4a8_pow2_mixed_lmul_hw_handoff_20260530
CUDA_VISIBLE_DEVICES= ~/miniforge3/envs/bcctn/bin/python scripts/count_params_official_to_hw.py
```
