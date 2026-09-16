# RESULTS: 官方 BCCTN checkpoint（11.09 M）在本 repo 測試協定下的分數 + 投影片 MAC 數字查核

Date: 2026-09-16
Base candidate: `official_bcctn/DCNN/Checkpoints/Trained_model.ckpt`（VikasTokala/BCCTN @ 53da450，md5 `991c7096f84d8b8dbf7d591277ba66ea`，epoch 20，11,086,488 params）
Handoff: [RESULTS_FP32_REFERENCE_IDENTITY_2026-09-16.md](../../18_w4a8_pow2_mixed_lmul_hw_handoff_20260530/docs/RESULTS_FP32_REFERENCE_IDENTITY_2026-09-16.md)、[RESULTS_PARAM_REDUCTION_OFFICIAL_BASIS_2026-09-16.md](../../18_w4a8_pow2_mixed_lmul_hw_handoff_20260530/docs/RESULTS_PARAM_REDUCTION_OFFICIAL_BASIS_2026-09-16.md)

## 0. TL;DR

| | VCTK MBSTOI | TIMIT MBSTOI | 備註 |
|---|---:|---:|---|
| Noisy | 0.7861 | 0.7311 | |
| **官方 ckpt（11.09 M，as released）** | **0.7958** | **0.7247** | VCTK 只比 noisy +0.0097；TIMIT 比 noisy 還低 −0.0064 |
| 自製 FP32 fft-512 重現版（4.79 M，`fp32_512_ref_full`） | 0.8535 | 0.8134 | |
| 自製 FP32 fft-256 / 5-stage（3.81 M，`BCCTN/results_May21`） | 0.8519 | 0.8114 | 投影片目前的 FP32 baseline |
| ⭐⭐ Deploy（W4A8 zero-mult，pow2 block-float） | 0.8357 | 0.7898 | **比官方 ckpt 高 +0.0399 / +0.0651** |

- 同一個 eval：750-pair × 8 SNR VCTK + 160-pair TIMIT、dir-18 metric 程式碼、32 CPU metric worker、1 GPU；逐 SNR `MBSTOI_noisy` 與 deploy run 相同（< 1e-4）。模型用**官方程式碼**建、`load_state_dict(strict=True)`。
- 官方 ckpt 在本測試集是 **out-of-domain**（作者自己的 HRIR / 方位渲染），不是 metric 問題（§2）。它的 ILD/IPD error 也比 noisy 差（VCTK ILD 3.71→4.28 dB、IPD 15.4→16.6°；TIMIT 3.29→4.46 dB、21.1→23.9°），與 2026-06-28 STATUS 用原始 metric 量到的結論一致。
- 官方 ckpt 的輸出**音量固定大 ~15–24×（+24…+28 dB）**（§2），MBSTOI/STOI/PESQ/ILD/IPD 不受影響，**SegSNR 對它無意義**（被夾在 −10 dB 下限），本文不列其 SegSNR。
- 投影片「93% / 35% / 96%、~1% MBSTOI loss」：四個數字都對不上任何單一基準（§3）。

## 1. Per-SNR MBSTOI

| Dataset | SNR | Noisy | 官方 ckpt (11.09 M) | 自製 FP32 fft-512 (4.79 M) | Deploy (830 K→338 K RTL) | Deploy − 官方 |
|---|---:|---:|---:|---:|---:|---:|
| VCTK | -6 | 0.5366 | 0.5408 | 0.7023 | 0.6649 | +0.1241 |
| VCTK | -3 | 0.6376 | 0.6448 | 0.7613 | 0.7310 | +0.0862 |
| VCTK | 0 | 0.7264 | 0.7433 | 0.8197 | 0.7962 | +0.0529 |
| VCTK | 3 | 0.7949 | 0.8131 | 0.8585 | 0.8403 | +0.0272 |
| VCTK | 6 | 0.8474 | 0.8630 | 0.8895 | 0.8767 | +0.0137 |
| VCTK | 9 | 0.8897 | 0.8990 | 0.9152 | 0.9073 | +0.0083 |
| VCTK | 12 | 0.9171 | 0.9224 | 0.9333 | 0.9260 | +0.0036 |
| VCTK | 15 | 0.9395 | 0.9401 | 0.9485 | 0.9431 | +0.0030 |
| **VCTK** | **Avg** | 0.7861 | **0.7958** | **0.8535** | **0.8357** | **+0.0399** |
| TIMIT | -6 | 0.4576 | 0.4482 | 0.6273 | 0.5792 | +0.1310 |
| TIMIT | -3 | 0.5614 | 0.5383 | 0.7067 | 0.6716 | +0.1333 |
| TIMIT | 0 | 0.6548 | 0.6324 | 0.7667 | 0.7371 | +0.1047 |
| TIMIT | 3 | 0.7227 | 0.7186 | 0.8085 | 0.7840 | +0.0654 |
| TIMIT | 6 | 0.8034 | 0.8054 | 0.8596 | 0.8414 | +0.0360 |
| TIMIT | 9 | 0.8463 | 0.8526 | 0.8894 | 0.8744 | +0.0218 |
| TIMIT | 12 | 0.8842 | 0.8871 | 0.9128 | 0.9021 | +0.0150 |
| TIMIT | 15 | 0.9181 | 0.9149 | 0.9359 | 0.9282 | +0.0133 |
| **TIMIT** | **Avg** | 0.7311 | **0.7247** | **0.8134** | **0.7898** | **+0.0651** |

其他指標（官方 ckpt，VCTK / TIMIT 平均）：STOI 0.8241→0.8473 / 0.7666→0.7820；PESQ wb 1.24→1.50 / 1.27→1.52；PESQ nb 1.86→2.26 / 1.80→2.15。完整檔：`docs/official_ckpt_full_{vctk,timit}_paper_style_results{,_averages}.{csv,txt}`。

## 2. 為什麼這麼低 — 已排除的解釋

| 假設 | 檢查 | 結果 |
|---|---|---|
| 架構 / 權重載錯 | 官方程式碼建模、strict load、md5 對 GitHub | 0 missing / 0 unexpected，11,086,488 params，md5 相同 |
| Eval subset / metric 不同 | 同檔案的 noisy MBSTOI/SegSNR/ILD 逐 bit 比對 deploy run | 相同 |
| 官方程式碼版本 ≠ 訓練當時 | `git log` 全歷史：`feature_extractors.py` 兩版只差一行 `import nnAudio`；`apply_mask.py` 與本 repo 相同；ckpt 2023-12-14 commit | 無版本錯配 |
| STFT / iSTFT normalization 不對稱（√512 = +27.1 dB） | 官方 `Stft`/`IStft` 都是 `normalized=True`、hann | 對稱，不是原因 |
| Loss 是 scale-invariant 所以增益任意 | 官方 `snr_loss` = 一般 SNR（scale-sensitive，weight 1） | 不是原因；增益是訓練資料慣例學進去的 |
| 輸入音量 off-distribution（BN running stats） | 輸入 ×0.03…×30 掃描：輸出增益 14×→236×、best-gain SNR 與 STOI 在 **×0.3–×1 最好**（15 dB：+16.6 dB、STOI 0.990；0 dB：+8.8 dB、STOI 0.852） | 我們的測試集音量就在它的最佳區，正規化救不了 |

結論：釋出的 ckpt 對本 repo 的 HRIR / 隨機方位 −90…90° 測試條件泛化很差（TIMIT 甚至變差），並帶一個固定的輸出增益。用它當投影片的 FP32 參考會讓「量化掉多少」變成負數（deploy 比它高），反而不能說明量化代價；**FP32 參考仍建議用自製 fft-512 重現版 0.8535 / 0.8134（4.79 M）**，官方 ckpt 另列一行「official checkpoint, out-of-domain on our test set」。

## 3. 投影片「Algorithm Optimization」查核

實際執行 MAC（forward hook，2 s 立體聲，複數層以實際執行的 4 個實數 conv 計，`code/mac_profile_official.py`）與 `19_algo_final/scripts/mac_profiler.py`（解析式，含 sliding window / top-4；兩者在 HW-sized conv 上完全一致：1.19 M/frame）：

| 模型 | MAC/s | 每 frame |
|---|---:|---:|
| 官方 11.09 M（fft 512 / hop 100，MHA@1024） | **11.27 G**（LBCCN 論文引 12.7 G） | 70.5 M / 100-sample frame |
| 自製 fft-512 / embed-512 重現版（3.97 M） | 5.21 G（文件寫 ≈29 M/frame = 4.7 G/s） | 32.5 M / 100-sample frame |
| HW-sized dense（fft 128 / E 256 / tied MHA） | 0.454 G（×4）/ 0.369 G（×3） | 2.27 M / 1.84 M per 5 ms |
| Deploy（grouped QKV + grouped decoder + top-4） | 0.279 G（×4）/ 0.222 G（×3） | 1.40 M / **1.10 M** per 5 ms |

| 投影片說法 | 投影片數字 | 重現版基準（5.21 G/s，×4） | 官方基準（11.27 G/s） |
|---|---:|---:|---:|
| (1) FFT 512→128 + embed 512→256 + tied MHA | −93% | **−91.3%** | **−96.0%** |
| (2) block-diagonal QKV + grouped decoder | −35% | **−38.5%**（×3：−39.8%） | 同左 |
| 合計 → 1.10 M/frame | −96% | **−94.6%** | **−97.5%** |
| 「about 1% MBSTOI loss」 | 1% | **−0.016 / −1.9%**（VCTK vs 0.8519）；−0.022 / −2.7%（TIMIT）；改用 0.8535 參考：−2.1% / −2.9% | 對官方 ckpt 反而 **+0.040 / +0.065** |

備註：(a) 文件的「29 M/frame → 1.10 M/frame」是 hop-100 frame 對 hop-80 frame、×4 對 ×3 的混合基準，總降幅 96% 只在這個混合下成立；同一基準是 94.6%（重現版）或 97.5%（官方）。(b) 「attention embedding 512→256」是相對重現版；官方 embed 是 1024。(c) tied MHA 在 mac_profiler 的 dense 基準已含（passes=2），所以 (1) 的 91%/96% 已包含 weight sharing。

建議寫法：明講基準。例如「Baseline = official BCCTN (11.09 M params, 11.3 GMAC/s): −96% → −38% → total −97.5%, 1.10 M MAC/frame (Gauss ×3)」或重現版「−91% → −38% → −95%」；MBSTOI 寫「−0.016 (−1.9%) vs. our FP32 reference」。

## 4. Reproduction

```bash
cd /raid/R12K41024/23_official_bcctn_ckpt_eval
git submodule update --init official_bcctn
GPU_ID=<free> LIMIT=750 METRIC_WORKERS=32 VCTK_ONLY=0 TAG=official_ckpt_full bash run_eval.sh   # ~15 min
CUDA_VISIBLE_DEVICES= ~/miniforge3/envs/bcctn/bin/python code/mac_profile_official.py          # MAC table
cd /raid/R12K41024/19_algo_final/scripts && CUDA_VISIBLE_DEVICES= ~/miniforge3/envs/bcctn/bin/python mac_profiler.py \
  --config /raid/R12K41024/18_w4a8_pow2_mixed_lmul_hw_handoff_20260530/code/config/config.yaml --profile both
```
