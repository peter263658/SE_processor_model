# RESULTS: MBSTOI 逐階段掉分 + per-SNR deploy vs 原始 BCCTN

Date: 2026-09-16
Base candidate: `ckpt/strict_core_pow2_blockfloat_deploy_ep3_vctk0.8357_timit0.7898.ckpt`（deploy）
Handoff: [RESULTS_ARCH_PROGRESSION.md](RESULTS_ARCH_PROGRESSION.md)（階段定義與 params/MAC；本檔只補 MBSTOI 掉分拆解）

## 0. TL;DR

原始 FP32 BCCTN → deploy 共掉 **VCTK −0.0162 / TIMIT −0.0216**。掉分主要來自第一步的 HW downsizing，不是 calibration；static activation calibration 本身 zero-shot 不掉分。

| 區段 | 內容 | ΔVCTK | ΔTIMIT | 佔總掉分（VCTK / TIMIT） |
|---|---|---:|---:|---:|
| A. HW downsizing + FP8 QAT | fft 512→128、embed 512→256、heads 32→16、sliding attn、depthwise-sep；3.97 M → 830 K | **−0.0120** | **−0.0148** | 74% / 69% |
| B. 結構壓縮 + W4A8 + attention 近似（09 → 14 composite） | tied MHA、full-scope W4A8 + top-4、shift-topK pow2 softmax、grouped decoder/skip、FP16 acc、L-Mul、grouped QKV、sparse packet | −0.0045 | −0.0086 | 28% / 40% |
| C. Zero-multiplier 算術 + activation calibration（14 → 18 deploy） | APoT scale、shift-add conv、L-Mul m8/m4、BN/PReLU pow2、delayed APoT descale、static calib、pow2 block-float（皆含 recovery FT） | +0.0003 | +0.0018 | −2% / −8% |
| **⭐⭐ 合計** | FP32 BCCTN 0.8519 / 0.8114 → deploy 0.8357 / 0.7898 | **−0.0162** | **−0.0216** | 100% |

Calibration 本身（dynamic activation deploy 0.8365 / 0.7912 → pow2 block-float deploy 0.8357 / 0.7898）：

| 步驟 | zero-shot Δ（無 FT） | FT 後 Δ | 註 |
|---|---:|---:|---|
| Static calibrated e4m3 activation（取代 runtime amax） | **+0.0005 / −0.0002** | −0.0011 / −0.0015 | vs 0.8365 / 0.7912；recovery FT（best ep2）反而低於 zero-shot |
| pow2 block-float shared exponent | −0.0015 / 未跑 | +0.0003 / +0.0001 | vs calib FT 0.8354 / 0.7897 |
| **Calibration 合計（FT 後）** | — | **−0.0008 / −0.0014** | 0.8365 / 0.7912 → 0.8357 / 0.7898 |

投影片建議：不要寫「calibration 會掉分」。Static calibration zero-shot 為 +0.0005 / −0.0002（noise 內）；calibration 鏈淨掉的 −0.0008 / −0.0014 來自 calib recovery FT（相對 zero-shot 再掉 −0.0016 / −0.0013）與 pow2 rounding（zero-shot −0.0015 / 未跑，FT 後收回）。最大掉分是 A 區段（downsizing + FP8 QAT）。

## 1. Setup

- Eval：paper-style VCTK（matched）+ TIMIT（unmatched），750 pairs × 8 SNR {−6,−3,0,3,6,9,12,15} dB。三個 per-SNR 來源的 `MBSTOI_noisy` 逐 SNR 一致（< 1e-4），確認是同一個 eval subset。
- Gates（deploy）：full VCTK ≥ 0.835、full TIMIT ≥ 0.7880。
- 平均值一律取 `*_averages.txt`（canonical），不是 per-SNR 等權平均（見 [RESPONSE_OPEN_ITEMS_PAPER_NUMBERS_2026-07-07.md](../../19_algo_final/docs/RESPONSE_OPEN_ITEMS_PAPER_NUMBERS_2026-07-07.md) §3）。
- 每列分數 = 該階段被報告/採用的分數（多數為 QAT/FT 後）；zero-shot 列另外標示。
- 相鄰兩列差 ≤ ~0.001 在 cross-run noise 範圍內，不要逐列過度解讀。

## 2. Results

### 2.1 逐階段 MBSTOI

Δstep = 相對進入該步驟前的 checkpoint 分數（主列即上一列；`a` zero-shot 與 `b` FT 都相對同一個起點：13 相對 row 12、14 相對 13b、15 相對 14b）；Δcum = 相對原始 FP32 BCCTN。

| # | 階段 | 模式 | VCTK | TIMIT | ΔstepV | ΔstepT | ΔcumV | ΔcumT | 來源 |
|---|---|---|---:|---:|---:|---:|---:|---:|---|
| 0 | 原始 BCCTN（自製重現版；實為 fft 256 / 5 stage / 3.81 M，見 §7 Erratum） | FP32 train | 0.8519 | 0.8114 | — | — | — | — | `BCCTN/results_May21/vctk_paper_style/` |
| 1 | HW-sized + FP8 QAT（fft 128 / embed 256 / 16 heads / attn_win 31 / depthwise-sep，830 K） | QAT | 0.8399 | 0.7966 | −0.0120 | −0.0148 | −0.0120 | −0.0148 | `main_model/results_May21/vctk_paper_style/` |
| 2 | 09 Tied MHA（qat_scope=mha_only：MHA W4A8，full softmax） | FT 80 ep | 0.8411 | 0.7966 | +0.0012 | +0.0000 | −0.0108 | −0.0148 | `09_weight_tied_mha/results/finetune/` |
| 3 | 09 strict：full-scope W4A8（conv 也 INT4）+ top-4 attention | QAT FT | 0.8383 | n/a | −0.0028 | n/a | −0.0136 | n/a | `09_weight_tied_mha/results/finetune_strict/` |
| 4 | 11 Shift-topK pow2 softmax（無 exp/div LUT） | eval-time 替換（無 FT） | 0.8360 | 0.7927 | −0.0023 | −0.0039 † | −0.0159 | −0.0187 | `11_.../softmax_rtl_shift_based/results/shift_topk_pow2_vctk_timit_w64/` |
| 5 | 12 Decoder grouped pointwise（g42） | QAT FT | 0.8361 | 0.7923 | +0.0001 | −0.0004 | −0.0158 | −0.0191 | `12_.../results/p1_grouped_timit/g42_vctk_timit_w750/` |
| 6 | 12 Decoder skip subset（top-50% L1） | QAT FT | 0.8352 | 0.7919 | −0.0009 | −0.0004 | −0.0167 | −0.0195 | `12_.../results/p2_skip_timit/s050_topk_vctk_timit_w750/` |
| 7 | 13 FP16 attention accumulator | QAT FT | 0.8352 | 0.7914 | +0.0000 | −0.0005 | −0.0167 | −0.0200 | [RESULTS_ARCH_PROGRESSION.md](RESULTS_ARCH_PROGRESSION.md) |
| 8 | 13 L-Mul M4 on QK + Score×V | QAT FT | 0.8364 | 0.7925 | +0.0012 | +0.0011 | −0.0155 | −0.0189 | 同上 |
| 9 | 14 Grouped QKV（qkv_groups=4） | QAT 30 ep | 0.8353 | 0.7881 | −0.0011 | −0.0044 | −0.0166 | −0.0233 | 同上 |
| 10 | 14 composite：+ sparse top-4 q0.8 packet + bit-exact L-Mul（production） | QAT FT | 0.8354 | 0.7880 | +0.0001 | −0.0001 | −0.0165 | −0.0234 | 同上 |
| 11 | APoT weight scale（apot_clip [−16,−1]） | KD FT | 0.8350 | 0.7892 | −0.0004 | +0.0012 | −0.0169 | −0.0222 | `18_w4a8_pow2_mixed_lmul/results/mixed_lmul_eval/B1b_KD_attn8_apotclip16_ftEp2_best/` |
| 12 | Strict zero-mult core，exact descale（shift-add conv、L-Mul m8/m4、BN APoT、PReLU pow2） | FT | 0.8355 | 0.7902 | +0.0005 | +0.0010 | −0.0164 | −0.0212 | `results/apot_descale_eval/control/` |
| 13a | Delayed 2-term APoT outlier descale [−16,−6] | **zero-shot** | 0.8335 | 0.7888 | −0.0020 | −0.0014 | −0.0184 | −0.0226 | `results/apot_descale_eval/bake/` |
| 13b | 同上 | FT 5 ep（取 ep2） | 0.8365 | 0.7912 | +0.0010 | +0.0010 | −0.0154 | −0.0202 | `results/apot_descale_eval/ft_ep2_full/` |
| 14a | Static calibrated e4m3 activation（p999，64 batch） | **zero-shot** | 0.8370 | 0.7910 | +0.0005 | −0.0002 | −0.0149 | −0.0204 | `results/apot_descale_eval/ft_calib_zeroshot_from_apot_ep2_full/`（本次新跑） |
| 14b | 同上 | FT 5 ep（取 ep2） | 0.8354 | 0.7897 | −0.0011 | −0.0015 | −0.0165 | −0.0217 | `results/calibrated_e4m3_activation_eval/calib_p999_b64_ep2_full/` |
| 15a | pow2 block-float activation（per-tensor pow2 shared exponent） | **zero-shot** | 0.8339 | 未跑 | −0.0015 | — | −0.0180 | — | `results/apot_descale_eval/ft_pow2_zeroshot_ep2_vctk/`（2026-05-31，VCTK only） |
| **15b** | **⭐⭐ Deploy：pow2 block-float** | FT 5 ep（取 ep3） | **0.8357** | **0.7898** | +0.0003 | +0.0001 | **−0.0162** | **−0.0216** | `results/apot_descale_eval/ft_pow2_ep3_full/` |

† row 4 TIMIT 相對 row 2（row 3 沒有 TIMIT）。row 4 的 TIMIT 是 `shift_topk_pow2`（含 mantissa、clamp −16）；最終 RTL mode `shift_topk_pow2_clamp8_nomantissa` 的 VCTK 同為 0.8360，TIMIT 未跑。

### 2.2 Per-SNR MBSTOI：deploy vs 原始 BCCTN

保留增益 = (deploy − noisy) / (FP32 − noisy)，也就是 deploy 保住了原始模型 MBSTOI 提升量的幾成。

| Dataset | SNR (dB) | Noisy | 原始 BCCTN FP32 | HW-sized FP8 | Deploy | Deploy − FP32 | 相對掉分 | Deploy − HW-sized | 保留增益 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| VCTK | −6 | 0.5366 | 0.6981 | 0.6720 | 0.6649 | −0.0332 | −4.76% | −0.0071 | 79.4% |
| VCTK | −3 | 0.6376 | 0.7580 | 0.7377 | 0.7310 | −0.0270 | −3.56% | −0.0067 | 77.6% |
| VCTK | 0 | 0.7264 | 0.8178 | 0.8029 | 0.7962 | −0.0216 | −2.64% | −0.0067 | 76.4% |
| VCTK | 3 | 0.7949 | 0.8586 | 0.8458 | 0.8403 | −0.0183 | −2.13% | −0.0055 | 71.3% |
| VCTK | 6 | 0.8474 | 0.8881 | 0.8807 | 0.8767 | −0.0114 | −1.28% | −0.0040 | 72.0% |
| VCTK | 9 | 0.8897 | 0.9152 | 0.9091 | 0.9073 | −0.0079 | −0.86% | −0.0018 | 69.0% |
| VCTK | 12 | 0.9171 | 0.9320 | 0.9273 | 0.9260 | −0.0060 | −0.64% | −0.0013 | 59.7% |
| VCTK | 15 | 0.9395 | 0.9473 | 0.9439 | 0.9431 | −0.0042 | −0.44% | −0.0008 | 46.2% |
| **VCTK** | **Avg** | 0.7861 | **0.8519** | 0.8399 | **0.8357** | **−0.0162** | −1.90% | −0.0042 | 75.4% |
| TIMIT | −6 | 0.4576 | 0.6213 | 0.5902 | 0.5792 | −0.0421 | −6.78% | −0.0110 | 74.3% |
| TIMIT | −3 | 0.5614 | 0.7034 | 0.6797 | 0.6716 | −0.0318 | −4.52% | −0.0081 | 77.6% |
| TIMIT | 0 | 0.6548 | 0.7644 | 0.7470 | 0.7371 | −0.0273 | −3.57% | −0.0099 | 75.1% |
| TIMIT | 3 | 0.7227 | 0.8083 | 0.7924 | 0.7840 | −0.0243 | −3.01% | −0.0084 | 71.6% |
| TIMIT | 6 | 0.8034 | 0.8581 | 0.8477 | 0.8414 | −0.0167 | −1.95% | −0.0063 | 69.5% |
| TIMIT | 9 | 0.8463 | 0.8889 | 0.8795 | 0.8744 | −0.0145 | −1.63% | −0.0051 | 66.0% |
| TIMIT | 12 | 0.8842 | 0.9128 | 0.9054 | 0.9021 | −0.0107 | −1.17% | −0.0033 | 62.6% |
| TIMIT | 15 | 0.9181 | 0.9342 | 0.9310 | 0.9282 | −0.0060 | −0.64% | −0.0028 | 62.7% |
| **TIMIT** | **Avg** | 0.7311 | **0.8114** | 0.7966 | **0.7898** | **−0.0216** | −2.66% | −0.0068 | 73.1% |

### 2.3 Per-SNR 其他指標：deploy − 原始 BCCTN

ILD/IPD 是 error（越低越好），所以 Δ 為正代表 deploy 較差。

| Dataset | SNR (dB) | SegSNR FP32 → Deploy (dB) | ΔSegSNR | ILD err FP32 → Deploy (dB) | ΔILD | IPD err FP32 → Deploy (°) | ΔIPD |
|---|---:|---|---:|---|---:|---|---:|
| VCTK | −6 | −1.97 → −3.24 | −1.27 | 4.972 → 4.969 | −0.003 | 29.40 → 30.23 | +0.83 |
| VCTK | −3 | −0.72 → −1.97 | −1.25 | 4.429 → 4.412 | −0.017 | 23.98 → 24.99 | +1.01 |
| VCTK | 0 | 0.60 → −0.66 | −1.26 | 3.894 → 3.916 | +0.022 | 18.37 → 19.60 | +1.23 |
| VCTK | 3 | 1.78 → 0.57 | −1.21 | 3.420 → 3.466 | +0.046 | 14.42 → 15.85 | +1.43 |
| VCTK | 6 | 3.11 → 1.91 | −1.20 | 2.951 → 3.010 | +0.059 | 11.19 → 12.70 | +1.51 |
| VCTK | 9 | 4.72 → 3.53 | −1.19 | 2.487 → 2.562 | +0.075 | 8.30 → 9.94 | +1.64 |
| VCTK | 12 | 6.17 → 4.94 | −1.23 | 2.136 → 2.215 | +0.079 | 6.23 → 7.89 | +1.66 |
| VCTK | 15 | 7.85 → 6.57 | −1.28 | 1.830 → 1.913 | +0.082 | 4.65 → 6.35 | +1.70 |
| **VCTK** | **Avg** | 2.69 → 1.46 | **−1.23** | 3.265 → 3.308 | **+0.043** | 14.57 → 15.94 | **+1.37** |
| TIMIT | −6 | −1.27 → −2.76 | −1.49 | 4.982 → 4.779 | −0.203 | 37.82 → 37.02 | −0.80 |
| TIMIT | −3 | 0.15 → −1.08 | −1.23 | 4.600 → 4.367 | −0.234 | 30.44 → 29.93 | −0.51 |
| TIMIT | 0 | 1.52 → 0.36 | −1.16 | 3.777 → 3.648 | −0.129 | 25.34 → 24.91 | −0.43 |
| TIMIT | 3 | 2.70 → 1.65 | −1.05 | 3.397 → 3.314 | −0.083 | 21.12 → 21.04 | −0.08 |
| TIMIT | 6 | 4.26 → 3.37 | −0.89 | 2.852 → 2.825 | −0.027 | 16.06 → 16.29 | +0.23 |
| TIMIT | 9 | 5.85 → 4.99 | −0.86 | 2.314 → 2.325 | +0.011 | 13.08 → 13.69 | +0.61 |
| TIMIT | 12 | 7.35 → 6.52 | −0.83 | 1.902 → 1.903 | +0.001 | 10.55 → 11.20 | +0.65 |
| TIMIT | 15 | 8.95 → 8.21 | −0.74 | 1.526 → 1.516 | −0.010 | 8.32 → 9.03 | +0.71 |
| **TIMIT** | **Avg** | 3.69 → 2.66 | **−1.03** | 3.169 → 3.085 | **−0.084** | 20.34 → 20.39 | **+0.05** |

完整數值（含 HW-sized 欄）：[RESULTS_MBSTOI_STAGE_DROP_PER_SNR_2026-09-16_per_snr.csv](RESULTS_MBSTOI_STAGE_DROP_PER_SNR_2026-09-16_per_snr.csv)

## 3. Cross-cuts

- **低 SNR 掉最多**：VCTK −6 dB −0.0332 vs +15 dB −0.0042；TIMIT −6 dB −0.0421 vs +15 dB −0.0060。但以保留增益看，低 SNR 反而保得最好（VCTK −6 dB 79.4%、TIMIT −3 dB 77.6%）；高 SNR 的保留比例低是因為 noisy 本來就接近上限，分母小。
- **per-SNR 掉分大部分也來自 downsizing**：deploy − HW-sized 在每個 SNR 都只有 −0.0008 ~ −0.0110，約為 deploy − FP32 的 19–47%（Avg：VCTK 26%、TIMIT 31%）。
- **近似步驟的 zero-shot 掉分可被 FT 收回**：delayed APoT descale（−0.0020 → FT 後 +0.0010）、pow2（−0.0015 → FT 後 +0.0003）。
- **Static calib 是反例**：zero-shot 幾乎不變（0.8370 / 0.7910，逐 SNR 都 ≥ dynamic activation 的 `ft_ep2_full`），但在它之下做 recovery FT 後 MBSTOI 反而變低（ep1 / ep2 / ep3 = VCTK 0.8349 / 0.8354 / 0.8334、TIMIT 0.7903 / 0.7897 / 0.7878，全部低於 zero-shot）。FT 是依 val loss 收斂，不保證 MBSTOI；這一步的掉分屬於 FT 路徑，不是 calibration 本身。
- **空間線索保住**：ILD error VCTK 平均只多 +0.043 dB，TIMIT 反而少 0.084 dB；IPD error VCTK +1.37°，TIMIT +0.05°。SegSNR 在各 SNR 幾乎等量下降（VCTK ≈ −1.2 dB、TIMIT −0.7 ~ −1.5 dB）。

## 4. Recommendation

投影片的 calibration 那頁改成「逐階段掉分」：

- 主圖/主表用 §0 的 A/B/C 三段：A（downsizing + FP8 QAT）−0.0120 / −0.0148 是主因；B −0.0045 / −0.0086；C（zero-mult + calibration）FT 後淨 +0.0003 / +0.0018。
- Calibration 單獨一列寫 zero-shot 與 FT 後兩個數，不要只寫「會掉分」：static calib zero-shot +0.0005 / −0.0002、pow2 zero-shot −0.0015 / 未跑，整條 calibration 鏈 FT 後淨 −0.0008 / −0.0014，deploy 仍過 gate（0.8357 ≥ 0.835、0.7898 ≥ 0.7880）。
- Per-SNR 表用 §2.2（MBSTOI）；想強調 binaural 賣點時補 §2.3 的 ILD/IPD 兩欄。

## 5. Reproduction

```bash
cd /raid/R12K41024/18_w4a8_pow2_mixed_lmul_hw_handoff_20260530
PKG=$PWD
export BCCTN_ACTIVATION_CALIB_PATH=$PKG/ckpt/activation_calibration_p999_b64.json

# 14a static calibrated e4m3 activation, zero-shot on the delayed-APoT deploy (0.8365)
CKPT=$PKG/ckpt/strict_core_apot_descale_deploy_ep2_vctk0.8365_timit0.7912.ckpt \
ACT_FP8_MODE=calibrated_e4m3 TAG=calib_zeroshot_from_apot_ep2_full VCTK_ONLY=0 GPU_ID=<free> \
  bash run_apot_descale_eval.sh

# 15a pow2 block-float, zero-shot on the calibrated-e4m3 FT ckpt (0.8354) — VCTK already at
# results/apot_descale_eval/ft_pow2_zeroshot_ep2_vctk/; set VCTK_ONLY=0 only if the TIMIT cell is wanted
CKPT=$PKG/ckpt/strict_core_calibrated_e4m3_deploy_ep2_vctk0.8354_timit0.7897.ckpt \
ACT_FP8_MODE=calibrated_pow2_e4m3 TAG=pow2_zeroshot_from_calibep2_full VCTK_ONLY=0 GPU_ID=<free> \
  bash run_apot_descale_eval.sh

# §2.2 / §2.3 per-SNR tables
~/miniforge3/envs/bcctn/bin/python scripts/make_per_snr_deploy_vs_bcctn.py \
  docs/RESULTS_MBSTOI_STAGE_DROP_PER_SNR_2026-09-16_per_snr.csv
```

## 7. Erratum（2026-09-16，同日追加）

row 0 與 §2.2 的「原始 BCCTN FP32」= `BCCTN/results_May21/` 的 0.8519 / 0.8114，經查證來自 `BCCTN/outputs/2025-06-02/00-25-51/.../epoch=137` ckpt：**fft 256 / 5 stage / kernel (5,2) / projected transformer，3,806,868 參數**，不是 fft 512 / embed 512 / 3,973,288，也不是官方 11.09 M。§0 A 區段的說明文字「3.97 M → 830 K」應讀作「3.81 M → 830 K」；MBSTOI 數字本身不變。另一個 FP32 參考 `fp32_512_ref_full`（0.8535 / 0.8134，4,791,448，真正的 fft-512 模型）若改為起點，A 區段變成 −0.0136 / −0.0168、總掉分 −0.0178 / −0.0236。證據與建議見 [RESULTS_FP32_REFERENCE_IDENTITY_2026-09-16.md](RESULTS_FP32_REFERENCE_IDENTITY_2026-09-16.md)。
