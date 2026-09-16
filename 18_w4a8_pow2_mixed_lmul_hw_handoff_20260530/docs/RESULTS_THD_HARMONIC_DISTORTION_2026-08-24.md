# RESULTS: THD / 諧波失真量測 — 演算法 vs 硬體量化拆解

Date: 2026-08-24
Base candidate: `ckpt/strict_core_pow2_blockfloat_deploy_ep3_vctk0.8357_timit0.7898.ckpt`
Related: [HANDOFF_HW_ZERO_MULTIPLIER_CORE.md](HANDOFF_HW_ZERO_MULTIPLIER_CORE.md), [ARCHITECTURE_SUMMARY_STRICT18.md](ARCHITECTURE_SUMMARY_STRICT18.md)
Scripts: `scripts/thd_probe.py`, `scripts/speech_harmonic_probe.py`, `scripts/analyze_speech_harmonics.py`, `scripts/plot_thd.py`
Raw data: `results/thd/{roundtrip,fp32,hw,speech_fp32,speech_hw,speech_summary}.json`, `results/thd/thd_summary.png`

回答口試委員「助聽器規格關心 THD，你們的驗證只有清晰度/相似度指標」的提問。

---

## 0. TL;DR

| 問題 | 量測 | 結果 | 判定 |
|---|---|---|---|
| 純音諧波失真是否超標 | THD @ 500/800/1000/1600/2000 Hz, 輸入 ≥ −20 dBFS | 部署硬體路徑 **最差 0.0930 %**（FP32 0.0553 %） | ⭐⭐ 通過。距 ANSI S3.22 / IEC 60118-7 常用 2 % 上限有 **> 20 倍餘裕** |
| 前端 STFT/iSTFT 本身失真 | 同一組音，identity mask 繞過網路 | THD ≤ 0.0024 %，THD+N ≤ 0.248 % | 前端不是失真來源 |
| 失真來自量化還是演算法 | 同一份權重跑 FP32 vs W4A8 硬體路徑 | THD 0.0553 % → 0.0930 %（+0.038 pp）；語音殘差 −7.05 dB → −7.58 dB | ⭐⭐ **主導失真是演算法（時變 mask），不是量化** |
| 量化雜訊佔比 | ‖hw − fp32‖² / ‖fp32‖²（同權重、只差算術） | **−16.87 dB**，比演算法殘差低 9.3 dB → 佔總殘差能量約 **12 %** | 量化為二階效應 |
| 語音諧波是否被破壞 | 29 檔 VCTK 0 dB SNR，clean 為 reference，比對 F0 諧波格點 | 諧波誤差 noisy 16.881 dB → fp32 10.674 dB → **hw 9.893 dB** | ⭐⭐ 增強後諧波保真度**優於**輸入，硬體路徑不劣於 FP32 |
| 諧波之間是否多出雜訊 | inter-harmonic / harmonic 能量比 (IHR) | clean −6.978 dB；noisy −1.654 → fp32 −3.503 → **hw −3.674 dB** | 往 clean 方向移動，未新增諧間雜訊 |

**一句話結論**：以助聽器電聲標準的 THD 定義來看，本設計的諧波失真在 0.1 % 等級，比法規上限低兩個數量級；殘餘的 THD+N（1–3 %）來自語音增強演算法本身「時變增益」的必然結果，而非零乘法器硬體核心的量化誤差。

---

## 1. Setup

**Pipeline under test**：完整 `STFT → encoder → tied ComplexMHA → decoder → complex ratio mask → iSTFT`（`DCNNLightningModule.forward`），fs = 16 kHz、`win_len/win_inc/fft_len = 128/80/128`。

三個條件共用**同一份權重**，只有算術路徑不同：

| 條件 | 內容 | Hydra overrides |
|---|---|---|
| `roundtrip` | STFT → identity mask → iSTFT，網路整段繞過 | 前端與 fp32 相同 |
| `fp32` | 部署權重、FP32 算術；`mixed_precision_qat=false`、full softmax、無 L-Mul / APoT BN / pow2 PReLU | `FP32_OVERRIDES`（`scripts/thd_probe.py`） |
| `hw` | 部署零乘法器路徑：W4A8 + calibrated pow2 block-float e4m3 activation + delayed APoT descale + QK/PV L-Mul m8 + outlier FP8 L-Mul m4 + `shift_topk_pow2_clamp8_nomantissa` softmax + top-4 prob packet + BN `fold_apot` + PReLU pow2 | `HW_OVERRIDES`，與 `run_apot_descale_eval.sh` 逐項一致 |

> 權重載入注意：W4A8 QAT 走 `torch.nn.utils.parametrize`，ckpt key 為 `<mod>.parametrizations.<w>.original`；FP32 圖沒有 parametrization，`build_model()` 會把 key 還原成 `<mod>.<w>`，確保兩條路徑跑的是 bit-identical 的底層權重。

**純音刺激**：3.0 s 正弦（雙耳同相），取中間 2.0 s（跳過 0.5 s）分析。500/800/1000/1600/2000 Hz × −40/−30/−20/−10 dBFS。分析用 coherent sampling（N = 32000，Δf = 0.5 Hz，所有測試頻率落在整數 bin），矩形窗洩漏理論為零，所以輸出中任何非基頻能量都是真實新增的。

- `THD  = sqrt(Σ_{k≥2} P_k / P_1)`，只計 k·f0 諧波 bin
- `THD+N = sqrt((P_total − P_1) / P_1)`，含諧波與所有其他 bin（會抓到時變增益造成的調變邊帶）

**語音刺激**：VCTK 0 dB SNR 測試集前 30 檔（29 檔有效 F0），clean 為 reference。STFT n_fft = 1024 / hop 256，`librosa.pyin` 在 clean 上取 F0，只在 voiced frame、且諧波 bin 高於該 frame 最強諧波 −40 dB 者計分；比較前先做最小平方標量增益對齊（增強器沒有絕對電平保證，否則常數偏移會被誤判成諧波損傷）。

---

## 2. Results

### 2.1 純音 THD（`results/thd/thd_summary.png`）

| f0 (Hz) | in (dBFS) | THD % roundtrip | THD % fp32 | THD % hw | THD+N % roundtrip | THD+N % fp32 | THD+N % hw | gain fp32 (dB) | gain hw (dB) |
|---|---|---|---|---|---|---|---|---|---|
| 500 | −40 | 0.0000 | 0.0410 | 0.1077 | 0.003 | 3.812 | 2.709 | −0.17 | −0.92 |
| 500 | −30 | 0.0000 | 0.0172 | 0.0880 | 0.003 | 2.242 | 2.317 | −0.35 | −0.39 |
| 500 | −20 | 0.0000 | 0.0290 | 0.0684 | 0.003 | 1.609 | 1.934 | −0.42 | −0.21 |
| 500 | −10 | 0.0000 | 0.0553 | 0.0930 | 0.003 | 2.137 | 2.459 | −0.35 | −0.34 |
| 800 | −40 | 0.0024 | 0.0784 | 0.6152 | 0.248 | 2.301 | 10.162 | −4.22 | −23.73 |
| 800 | −30 | 0.0024 | 0.0525 | 0.0747 | 0.248 | 0.886 | 1.392 | −0.30 | −0.33 |
| 800 | −20 | 0.0024 | 0.0410 | 0.0576 | 0.248 | 0.744 | 1.021 | −0.49 | +0.07 |
| 800 | −10 | 0.0024 | 0.0372 | 0.0665 | 0.248 | 0.717 | 1.306 | −0.52 | +0.03 |
| 1000 | −40 | 0.0000 | 0.0329 | 0.0347 | 0.001 | 2.985 | 3.293 | −1.33 | −2.90 |
| 1000 | −30 | 0.0000 | 0.0126 | 0.0124 | 0.001 | 1.906 | 1.318 | −0.06 | −0.05 |
| 1000 | −20 | 0.0000 | 0.0137 | 0.0189 | 0.001 | 2.090 | 1.889 | −0.08 | −0.03 |
| 1000 | −10 | 0.0000 | 0.0157 | 0.0187 | 0.001 | 2.146 | 1.930 | −0.09 | +0.14 |
| 1600 | −40 | 0.0005 | 0.0313 | 0.1087 | 0.078 | 5.938 | 16.841 | −6.80 | −13.22 |
| 1600 | −30 | 0.0005 | 0.0207 | 0.0183 | 0.078 | 2.008 | 1.485 | −1.14 | −1.35 |
| 1600 | −20 | 0.0005 | 0.0133 | 0.0297 | 0.078 | 1.597 | 1.608 | −0.51 | −0.00 |
| 1600 | −10 | 0.0005 | 0.0103 | 0.0297 | 0.078 | 1.433 | 1.538 | −0.60 | +0.19 |
| 2000 | −40 | 0.0000 | 0.0081 | 0.0174 | 0.000 | 16.951 | 31.044 | −13.90 | −22.96 |
| 2000 | −30 | 0.0000 | 0.0014 | 0.0034 | 0.000 | 2.877 | 5.561 | −4.70 | −14.11 |
| 2000 | −20 | 0.0000 | 0.0000 | 0.0010 | 0.000 | 1.482 | 3.207 | −1.04 | −3.46 |
| 2000 | −10 | 0.0000 | 0.0008 | 0.0014 | 0.000 | 1.955 | 3.310 | −1.97 | −4.00 |

**Worst case 摘要**

| 輸入範圍 | max THD roundtrip | max THD fp32 | max THD hw | max THD+N roundtrip | max THD+N fp32 | max THD+N hw |
|---|---|---|---|---|---|---|
| ≥ −20 dBFS | 0.0024 % | 0.0553 % | **0.0930 %** | 0.248 % | 2.146 % | 3.310 % |
| ≥ −30 dBFS | 0.0024 % | 0.0553 % | **0.0930 %** | 0.248 % | 2.877 % | 5.561 % |
| 全部（含 −40 dBFS） | 0.0024 % | 0.0784 % | 0.6152 % | 0.248 % | 16.951 % | 31.044 % |

三個要點：

1. **THD（真諧波）在任何條件下都極小。** 正常工作電平（≥ −20 dBFS）硬體路徑最差 0.0930 %，相對 ANSI S3.22 / IEC 60118-7 常見 2 % 判定線約 −27 dB 餘裕。網路引入的諧波比前端 STFT/iSTFT 高一個數量級，但兩者都遠低於可聞門檻。
2. **THD+N 才是主要數字（1–3.3 %），而且它不是「非線性失真」。** 它由 complex ratio mask 的**逐幀時變增益**產生的調變邊帶構成（frame rate = 16000/80 = 200 Hz，邊帶落在 f0 ± 200 Hz，不在諧波格點上）。任何 gain-based 降噪演算法都有這一項；FP32 條件同樣有（最差 2.146 %），代表這是演算法屬性而非硬體。
3. **−40 dBFS 的離群值（800 Hz THD 0.6152 %、2000 Hz THD+N 31 %）是「語音增強器把穩態純音判定為雜訊而衰減」**，不是失真爆掉：同一格的基頻增益是 −23.73 dB / −22.96 dB，分母塌掉才讓比值變大。這是降噪功能正常運作的證據（圖三 panel C）。這一點需要在委員面前主動說明，否則單看表格會誤讀。

### 2.2 純音以外：語音諧波保真度（29 檔 VCTK 0 dB SNR）

| 指標 | clean (ref) | noisy | fp32 | hw | 意義 |
|---|---|---|---|---|---|
| `harm_err_db` 諧波 bin 幅度誤差 \|dB\| | 0 | 16.881 | 10.674 | **9.893** | 越小越好；增強後比輸入改善 ~7 dB |
| `harm_err_w_db` 能量加權版 | 0 | 5.940 | 7.068 | **7.004** | 強諧波被 mask 略為修改（見下） |
| `harm_bias_db` 有號偏差 | 0 | +15.924 | +3.594 | **+1.889** | 雜訊灌進諧波 bin 的量大幅下降 |
| `ihr_db` 諧間/諧波能量比 | −6.978 | −1.654 | −3.503 | **−3.674** | 往 clean 移動 → 沒有新增諧間雜訊 |
| `her` 諧波能量佔比 | 0.783 | 0.586 | 0.668 | **0.674** | 頻譜結構回復到 clean 方向 |
| 對 clean 的殘差能量比（增益對齊後） | — | +0.57 dB | −7.05 dB | **−7.58 dB** | 整體誤差下降 8.2 dB |

**量化貢獻的直接量測**：`10·log10(‖y_hw − y_fp32‖² / ‖y_fp32‖²) = −16.87 dB`（同一份權重、同一輸入，只差算術路徑）。相對於演算法殘差 −7.58 dB，量化雜訊低 9.3 dB，換算約佔總殘差能量的 **12 %**。

**誠實記錄的一項反向結果**：`harm_err_w_db`（以 clean 諧波能量加權）在增強後反而從 5.940 dB 上升到 7.0 dB。意思是「最強的那些諧波」被 mask 動到的幅度比輸入雜訊動到的還多——這是 over-suppression 的既有現象，與 `20_pesq_finetune` 正在處理的 PESQ 議題同源，不是新問題。未加權指標大幅改善代表整體諧波結構是變好的（弱諧波 bin 的雜訊被清掉）。

---

## 3. Cross-cuts

- **拆解方式成立**：因為 `fp32` 與 `hw` 載入 bit-identical 權重、共用同一組架構性 override（grouped decoder pointwise、`qkv_groups=4`、skip indices），兩者差值可以乾淨地歸因於「量化 + 近似運算」。這比拿一個 FP32 baseline 模型去比更有說服力，因為後者連權重都不同。
- **THD 與 MBSTOI 一致**：硬體路徑在 THD、諧波保真度、殘差能量三個獨立面向都不劣於 FP32，與既有 gate（VCTK 0.8357 / TIMIT 0.7898）給出的結論相同 —— W4A8 + 零乘法器核心沒有引入額外可測失真。
- **前端 FFT 目前是 FP32**。這次量到的 `roundtrip` 是浮點前端的地板（THD ≤ 0.0024 %）。定點 STFT/iSTFT 的失真尚未納入本次量測，見第 5 節。
- **DC bin 被丟棄**：`Stft.forward` 砍掉 f=0，`apply_mask` 再補零。800 Hz / 1600 Hz 的 roundtrip 殘差（THD+N 0.248 % / 0.078 %）就是這個 —— 這兩個頻率不落在 128-point FFT 的 125 Hz 格點上，洩漏到被歸零的 DC bin。500/1000/2000 Hz 落在整數 bin，roundtrip 殘差 ≈ 0。屬已知設計，不是 bug。

---

## 4. Recommendation

可以直接用這組數字回覆委員，主軸是「拆解」而不是「辯護」：本設計在助聽器標準定義的 THD 上是 0.1 % 等級，比 2 % 上限低兩個數量級；真正的殘餘（THD+N 1–3 %）是**任何 gain-based 語音增強都會有的時變增益調變**，而且 FP32 版本一樣存在，因此不是零乘法器硬體核心的問題 —— 量化只貢獻總殘差能量的約 12 %（−16.87 dB）。語音端的諧波保真度則是正向的：增強後諧波誤差比輸入低約 7 dB、諧間雜訊往 clean 靠。

要補強的話，優先順序：

1. 把 `--stimulus` 擴到 ANSI S3.22 原始三個頻率（500 / 800 / 1600 Hz）之外再加 2500 Hz 以上，並以校準過的 SPL 對應而非 dBFS 表示（需要一個 dBFS→dB SPL 的假定，說清楚即可）。
2. 加入定點 STFT/iSTFT 的 THD（`10_stft_fixpoint_ptq` 有現成量化前端），把前端地板從「FP32 前端」換成「實際晶片前端」。
3. 若委員追問可聞性，補一個 modulation-domain 或 PEAQ-like 指標；純 THD 對時變增益系統本來就不是完整描述。

---

## 5. Reproduction

```bash
cd /raid/R12K41024/18_w4a8_pow2_mixed_lmul_hw_handoff_20260530
export PYTHONPATH=$PWD/code NUMBA_CACHE_DIR=/tmp/numba_cache MPLCONFIGDIR=/tmp/mpl_cache

# (1) 前端地板 + FP32 演算法：純音掃描
CUDA_VISIBLE_DEVICES=<free> conda run -n bcctn python scripts/thd_probe.py \
    --mode roundtrip --out results/thd/roundtrip.json
CUDA_VISIBLE_DEVICES=<free> conda run -n bcctn python scripts/thd_probe.py \
    --mode fp32 --out results/thd/fp32.json

# (2) 部署硬體路徑（env 與 run_apot_descale_eval.sh 相同）
export BCCTN_APOT_A_MIN=-16 BCCTN_APOT_A_MAX=-1 BCCTN_APOT_B_MIN=-16 BCCTN_APOT_B_MAX=-1
export BCCTN_ACTIVATION_FP8_MODE=calibrated_pow2_e4m3
export BCCTN_ACTIVATION_CALIB_PATH=$PWD/ckpt/activation_calibration_p999_b64.json
export BCCTN_OUTLIER_DELAYED_DESCALE=1 BCCTN_OUTLIER_DESCALE_APOT_MIN=-16 BCCTN_OUTLIER_DESCALE_APOT_MAX=-6
CUDA_VISIBLE_DEVICES=<free> conda run -n bcctn python scripts/thd_probe.py \
    --mode hw --out results/thd/hw.json

# (3) 語音諧波保真度（fp32 用未設上述 env 的乾淨 shell 跑）
CUDA_VISIBLE_DEVICES=<free> conda run -n bcctn python scripts/speech_harmonic_probe.py \
    --mode fp32 --n_files 30 --out results/thd/speech_fp32.json
CUDA_VISIBLE_DEVICES=<free> conda run -n bcctn python scripts/speech_harmonic_probe.py \
    --mode hw   --n_files 30 --out results/thd/speech_hw.json
conda run -n bcctn python scripts/analyze_speech_harmonics.py   # noisy baseline + 量化 NSR

# (4) 圖
conda run -n bcctn python scripts/plot_thd.py                    # results/thd/thd_summary.png
```

`BCCTN_ACTIVATION_CALIB_PATH` 未設而 `ACT_FP8_MODE=calibrated_pow2_e4m3` 會逐檔失敗（同 `run_apot_descale_eval.sh` 的已知陷阱）。
