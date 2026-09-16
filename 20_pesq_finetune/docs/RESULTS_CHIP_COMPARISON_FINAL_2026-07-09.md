# RESULTS: Chip-Paper Comparison Final — vs P1/P4/P3, PESQ Ceiling Chain, Low-SNR Extension

Date: 2026-07-09
Base candidate: `18_w4a8_pow2_mixed_lmul_hw_handoff_20260530/ckpt/strict_core_pow2_blockfloat_deploy_ep3_vctk0.8357_timit0.7898.ckpt`
Handoff: [STATUS_CHIP_PAPER_COMPARISON_POSITIONING_2026-06-28.md](STATUS_CHIP_PAPER_COMPARISON_POSITIONING_2026-06-28.md)

P2 (CICC'26, VB+DEMAND) 依使用者指示暫不列入;本輪對象為 P1 (Lee JSSC'20)、P4 (Park ISSCC'23/JSSC'24)、P3 (Wu A-SSCC'25)。

## 0. TL;DR

| Variant | 條件 | ΔPESQ wb (−6..9) | MBSTOI (full) | STOI (full) | Verdict |
|---|---|---|---|---|---|
| Deploy (W4A8 零乘法器) | VCTK binaural | +0.2628 | 0.8357 (gate 0.835 pass) | 0.8683 | 晶片本體數字;PESQ 不與 P4 正面比,報 Δ+雙 band |
| FP32-512 baseline | 同上 | +0.4847 | 0.8535 | 0.8852 | 縮放比的分母;演算法譜系參考 |
| **FP32-512 + torch_pesq w12 (12ep)** | 同上 | **+0.8857** | 0.8454 | 0.8840 | ⭐⭐ **演算法層 ΔPESQ 超過 P4 的 +0.78**;PESQ 上限路線 |
| FP32-512 + MetricGAN (11 GAN ep) | 同上 | +0.5828 | **0.8600** | **0.8902** | 對 baseline 全指標 Pareto 改善;PESQ 未及 w12 路線 |
| FP32-512 + torch_pesq w12 飽和(+早停續訓) | 同上 | **+0.9327** | 0.8465 | 0.8850 | ceiling 定稿;續訓僅 +0.047,w12 已收斂 |
| **Deploy-arch + torch_pesq w12 QAT(11ep, 實測)** | 同上 | **+0.5225** | 0.8209 (gate fail) | 0.8599 | 縮放律驗證:落在預測帶 [+0.48, +0.65] 內;ablation 點,非 deploy 替代 |
| Deploy @ −8/−10 dB zero-shot | 訓練分布外 (下限 −7) | wb +0.005~+0.023 | −10dB: 0.3883→0.5357 | −10dB: 0.5922→0.6567 | 打 P3 的列:純音訊、OOD 仍有大幅可懂度增益 |

## 1. Setup

- Eval sets: VCTK (matched, 750 pairs/SNR) + TIMIT (unmatched, 160 pairs/SNR), binaural HRIR render, SNR −6..+15 (8 點, 3 dB step);−6..+9 子集對齊 P4 的 CHiME-2 軸。
- Gates: full VCTK MBSTOI ≥ 0.835、full TIMIT ≥ 0.788。Deploy 本輪重跑逐位重現 0.8357 / 0.7898(pass)。
- Metric chain 統一(本輪修正):nb-PESQ 一律 P.862 @8kHz(dir-18 eval.py 舊版在 16kHz 算 nb,已修;`ft_pow2_ep3_full_stoi` 的 nb 欄位作廢,canonical = `ft_pow2_ep3_full_nb8k`)。wb = P.862.2 @16kHz。STOI = pystoi 單聲道 L/R 平均。三套 eval 代碼的 noisy 欄位已驗證逐位一致。
- pystoi 哨兵值:16/6000 VCTK 檔(0.27%)因靜音移除後不足 0.4s 回傳 1e-5,noisy/enh 同時中招且各模型相同 → Δ 與跨模型比較不受影響,絕對 STOI 統一低估約 0.002。未過濾。
- 競品數字來源:P4 = JSSC'24 原文(CHiME-2 −6..+9 平均,50% hop:SDR 14.00 / PESQ 2.73 / STOI 91.44;25% hop:14.31 / 2.80 / 91.61;PESQ band 未標明,內文引 P.862 原文)。P1 晶片論文僅報 word accuracy(IEEE+NOISEX-92, −6..+6);其 PESQ 2.38 / STOI 89.08 為 P4 的 FP 軟體重現。P3 = A-SSCC'25,全文僅 chip summary 一點:SNR tolerance −10 dB / PESQ 2.13(TMSV,音訊+唇部影像),無 noisy、無平均、無 STOI。CHiME-2 noisy wb 錨點 1.95(使用者由第三方 repo eval code 確認 wb)。

## 2. Results

### 2.1 Deploy canonical(`ft_pow2_ep3_full_nb8k`)

| set / range | MBSTOI n→e | STOI n→e | PESQ wb n→e | PESQ nb n→e |
|---|---|---|---|---|
| VCTK −6..15 | 0.7861→0.8357 | 0.8241→0.8683 | 1.2404→1.5932 | 1.8597→2.2985 |
| VCTK −6..9 | 0.7388→0.8027 | 0.7887→0.8429 | 1.1535→1.4163 | 1.6917→2.0838 |
| TIMIT −6..15 | 0.7311→0.7898 | 0.7666→0.8228 | 1.2667→1.6558 | 1.7997→2.2300 |
| TIMIT −6..9 | 0.6744→0.7479 | 0.7212→0.7887 | 1.1523→1.4477 | 1.6181→1.9859 |

ckpt 重選 close-out:ep2(0.8339 / 0.8666 / wb 1.5979)MBSTOI 與 STOI 均劣於 ep3;ep4 新指標 eval 兩度 timeout,以舊 eval MBSTOI 0.8351 < 0.8357 結案。ep3 維持 deploy 選點。

### 2.2 Deploy @ −8/−10 dB(zero-shot,訓練 SNR 下限 −7)

測試點由 −6 dB 檔案精確合成:`noisy(T) = clean + (noisy(−6) − clean) × 10^((−6−T)/20)`,同一噪音實例,與 generator 配方數學等價(`20_pesq_finetune/scripts/gen_lowsnr_ext.py`;clipping ≤ 0.20% samples)。

| set @ SNR | MBSTOI n→e | STOI n→e | PESQ wb n→e | PESQ nb n→e |
|---|---|---|---|---|
| VCTK −8 | 0.4617→0.6055 | 0.6286→0.7012 | 1.0781→1.1009 | 1.3966→1.5103 |
| VCTK −10 | 0.3883→0.5357 | 0.5922→0.6567 | 1.0754→1.0803 | 1.3664→1.4245 |
| TIMIT −8 | 0.3882→0.5148 | 0.5380→0.6085 | 1.0530→1.0792 | 1.3019→1.3749 |
| TIMIT −10 | 0.3232→0.4436 | 0.5012→0.5588 | 1.0447→1.0634 | 1.2704→1.3077 |

對 P3 的敘事:P3 的 −10 dB / PESQ 2.13 依賴唇部影像(TMSV);本設計在分布外 −10 dB、純音訊條件下 MBSTOI Δ+0.1474、STOI Δ+6.45pt(甚至高於 −6 dB 點的 MBSTOI Δ+0.1283)。PESQ 在 −10 僅 Δ+0.058 (nb),低 SNR 主打可懂度。

### 2.3 FP32-512 baseline 與實測縮放比(Run B)

原始全尺寸 BCCTN 復原:`BCCTN/outputs/2025-05-10/19-39-07/.../epoch=98-step=95832.ckpt`(win 400 / fft 512 / untied input_proj MHA;載入 missing=0 / mismatch=0;身分佐證:MBSTOI 0.8535 ≈ 文檔 0.852)。工作環境 `20_pesq_finetune/fp32_512_pesq_ft/code/`。

| set / range | MBSTOI | STOI | PESQ wb | PESQ nb |
|---|---|---|---|---|
| VCTK −6..15 | 0.8535 | 0.8852 | 1.8380 | 2.5554 |
| VCTK −6..9 | 0.8244 | 0.8632 | 1.6382 | 2.3256 |
| TIMIT −6..15 | 0.8134 | 0.8460 | 1.8048 | 2.4573 |
| TIMIT −6..9 | 0.7764 | 0.8155 | 1.5755 | 2.1870 |

**縮放比 R = deploy ÷ FP32-512**(整條「512→128 FFT 縮小 + 結構壓縮 + W4A8 零乘法器」鏈):

| range | MBSTOI | STOI | PESQ wb | PESQ nb |
|---|---|---|---|---|
| VCTK full | 0.979 | 0.981 | 0.867 | 0.899 |
| VCTK −6..9 | 0.974 | 0.976 | 0.865 | 0.896 |
| TIMIT full | 0.971 | 0.973 | 0.917 | 0.907 |

使用者原假設 1.5%:對 MBSTOI/STOI(2~3%)近似成立;對 PESQ(8~13%)低估一個量級。估算 deploy PESQ 時用 ×0.87 (wb) / ×0.90 (nb)。

### 2.4 FP32-512 + torch_pesq w12(PESQ 上限路線)⭐⭐

配方:warm-start 0510_ep98,`BinauralLoss + pesq_weight=12`(torch_pesq),STOI 10 / IPD 10 / ILD 1 / SNR 1 保留,lr 2e-5,VCTK train。12 epochs(top-5 = ep7..11 全為最後五個 → val 未飽和,續訓被 GPU 清理事件中止;見 §3)。評測 ckpt:`outputs/pesqft_w12_lr2e-5_20ep/.../version_8/checkpoints/last.ckpt`。

| range | MBSTOI | STOI | PESQ wb | PESQ nb |
|---|---|---|---|---|
| VCTK −6..15 | 0.8454 | 0.8840 | 2.2438 | 2.8243 |
| VCTK −6..9 | 0.8145 | 0.8613 | 2.0392 | 2.5995 |

續訓到飽和(warm ep11 → +early-stop,`outputs/pesqft_w12_sat`):VCTK full 0.8465 / 0.8850 / **2.2911** / **2.8652**;−6..9 0.8155 / 0.8624 / **2.0862(Δ+0.9327)** / **2.6426(Δ+0.9509)**。續訓僅 +0.047 wb,w12 劑量已收斂;可懂度持平。

vs baseline:wb +0.4056(full)、MBSTOI −0.0081、STOI −0.0012 — FP32 上的 PESQ finetune 幾乎零可懂度代價(deploy 架構上同權重歷史增益僅 +0.23 wb 且 MBSTOI −0.005)。
**−6..9 Δwb = 2.0392 − 1.1535 = +0.8857 > P4 的 +0.78**(同軸、noisy 起點更低 1.15 vs 1.95、含混響 binaural render)。Δnb = +0.9078。

### 2.5 FP32-512 + MetricGAN(11 GAN epochs)

配方:`code/DCNN/metricgan.py` + `code/train_metricgan.py` — SpecDiscriminator(spectral-norm CNN, 171k params, per-ear 平均)以真 ITU wb-PESQ 標籤(每 batch 抽 8 樣本)回歸訓練;G loss = BinauralLoss(pesq_weight=0)+ 12 × MSE(D(enh), 1)。warm-start 自 w12 ep11。中途兩次被清理事件截斷(D 重啟自 scratch),有效 GAN epochs ≈ 11。評測 ckpt:`outputs/mgan_cont1/.../last.ckpt`(best-val == last)。

| range | MBSTOI | STOI | PESQ wb | PESQ nb |
|---|---|---|---|---|
| VCTK −6..15 | 0.8600 | 0.8902 | 1.9375 | 2.5993 |
| VCTK −6..9 | 0.8320 | 0.8690 | 1.7363 | 2.3882 |

vs baseline:**全指標 Pareto 改善**(MBSTOI +0.0065、STOI +0.0050、wb +0.0995、nb +0.0439,full)。vs w12 warm-start 點:PESQ 回落(2.2438→1.9375),可懂度上升——STOI-10 錨 + 弱化的 D 梯度把模型拉回可懂度最優。本輪設定下 MetricGAN 不是 PESQ 上限路線,而是「平衡操作點」。

### 2.6 vs 競品(−6..+9 軸)

| | P1 (P4 重現) | P4 | P3 | Deploy | FP32-512 w12-ft |
|---|---|---|---|---|---|
| PESQ noisy | ≈1.95 | ≈1.95 | 未報 | wb 1.1535 / nb 1.6917 | 同左 |
| PESQ enh | 2.38 | 2.73 | 2.13@−10 (AV) | wb 1.4163 / nb 2.0838 | wb 2.0392 / nb 2.5995 |
| ΔPESQ | +0.43 | +0.78 | — | +0.2628 / +0.3921 | **+0.8857 / +0.9078** |
| STOI | 89.08 | 91.44 | 未報 | 84.29 (Δ+5.42) | 86.13 (Δ+7.26) |
| MBSTOI / ILD / IPD | 見 §2.7(已量測) | 見 §2.7 | 未報 | 0.8027 / 有 | 0.8145 / 有 |

§2.6 原「mono 不可量測」一列已被 §2.7 取代:P1/P4 的網路已重現並以 bilateral 方式量到 MBSTOI/ILD/IPD。

### 2.7 Mono 範式 baseline 實測(P1/P4 網路重現 + bilateral,`21_mono_se_baselines/`)⭐

把 P4 (SNU JSSC'24, TRU-Net 變體, 56.3k params) 與 P1 (NTU JSSC'20 CNN, 495k params) 的**網路**照論文 1:1 重現(FP32、不量化、matched binaural VCTK 拆單耳 SI-SDR 訓練——全對 baseline 有利),測試時左右耳獨立推論(bilateral)後丟同一考卷。量的是**範式**非其 silicon。完整結果:`21_mono_se_baselines/docs/RESULTS_MONO_SE_BASELINES_2026-07-09.md`。同時新增 FP32 雙耳參考(`09_tied_ft`,strict-core 量化線 FP32 祖先,逐位重現 0.8411/0.7966)。

**領銜對比表:空間線索(ILD/IPD error,全均,↓佳)** — 這是範式的決定性欄:

| | VCTK ILD (dB) | VCTK IPD (°) | TIMIT ILD (dB) | TIMIT IPD (°) |
|---|---|---|---|---|
| noisy(未處理) | 3.71 | 15.4 | 3.29 | 21.1 |
| **Deploy (binaural, W4A8 零乘法)** | **3.31 ✓改善** | 15.9 | **3.08 ✓改善** | 20.4 ✓ |
| FP32 雙耳(演算法譜系) | **3.21 ✓改善** | 15.4 | **2.97 ✓改善** | 19.6 ✓ |
| P4/SNU 單耳(重現) | 5.71 ✗惡化 | 14.8 | 5.79 ✗惡化 | 20.2 |
| P1/NTU 單耳(重現) | 5.47 ✗惡化 | 17.1 ✗ | 5.50 ✗惡化 | 22.1 ✗ |

唯一的雙耳護城河:兩個 mono 範式在 **32/32 個(2 集×8 SNR×2 metric)** ILD 測試點**全數惡化於 noisy**(−6dB 近翻倍 6.07→9.25),雙耳(FP32 與 W4A8)全數改善。範式差距(∼1.8×)遠大於量化損失(FP32→W4A8 ILD 僅 +0.09~0.11 dB)。IPD:mono 破壞較輕(mask 沿用 noisy phase),但 NTU 兩集皆劣、SNU 打平——非主打欄。

**單耳品質欄(PESQ wb / STOI,全均)** — 誠實揭露,附 caveat:

| | VCTK PESQ | TIMIT PESQ | VCTK STOI | TIMIT STOI |
|---|---|---|---|---|
| noisy | 1.240 | 1.267 | 0.824 | 0.767 |
| Deploy (W4A8) | 1.593 | 1.656 | 0.868 | 0.823 |
| FP32 雙耳 | 1.678 | 1.758 | 0.874 | 0.830 |
| P4/SNU 單耳 (FP32) | 1.949 | 1.935 | 0.881 | 0.850 |
| P1/NTU 單耳 (FP32) | 1.820 | 1.764 | 0.876 | 0.843 |

mono baseline 的 PESQ/STOI **高於** deploy——三重 caveat:(a) FP32 上限 vs deploy 的 W4A8;(b) mono 純 SI-SDR loss 全力換單耳品質,雙耳 loss 帶 ILD/IPD 項是刻意取捨;(c) deploy 的 PESQ-ceiling 路線(§2.4/2.8:FP32 Δwb +0.9327、QAT 實測 Δ+0.5225)正在把此差距收斂。原論文自報 PESQ(SNU CHiME2 2.73)不可直接比:同網路本考卷僅 1.72(考卷難度位移 −1.0 PESQ);回推 SNU silicon 本考卷 ≈ PESQ 1.68 / STOI 85.5(見 mono doc §2.9)。

**MBSTOI 不列 head-to-head(刻意)**:mono 與雙耳打平甚至 mono 略高(full 與 −6..9 皆然;數字見 mono doc §2.8),因為 MBSTOI 的 **EC 階段搜尋耳間 level/time 補償**,恰好吸收掉單耳處理造成的 ILD 扭曲——它量「最佳耳間補償下的可懂度」,對 ILD 誤差天生免疫。MBSTOI 仍是 deploy 自身的 gate 指標(0.8357),但**不適合當雙耳優勢的 head-to-head 判別欄**,故本對比以 ILD/IPD 領銜。(參考數字備查:MBSTOI full VCTK noisy 0.7861 / deploy 0.8357 / FP32 0.8411 / SNU 0.8383 / NTU 0.8372;TIMIT 0.7311 / 0.7898 / 0.7966 / 0.8035 / 0.8006。)

### 2.8 QAT 驗證實測(deploy 架構 + torch_pesq w12,縮放律對答案)

配方:`run_pesq_finetune.sh` — deploy base 熱啟動、完整零乘法器 stack、BinauralLoss + torch_pesq w=12、lr 2e-5。跑 11 epochs(SIGKILL 截斷;best-val = epoch 10, val 21.6478,仍緩降)。評測:dir-20 eval 鏈(nb@8k + STOI),ckpt `outputs/pesq_finetune/qatverify_w12_lr2e-5_ep20/.../epoch=10-step=5324-val=21.6478.ckpt`。

| range | MBSTOI | STOI | PESQ wb | PESQ nb |
|---|---|---|---|---|
| VCTK −6..15 | 0.8209 | 0.8599 | 1.8660 | 2.5033 |
| VCTK −6..9 | 0.7859 | 0.8332 | 1.6760 | 2.2699 |

**縮放律對答案(−6..9, wb)**:實測 Δ = 1.6760 − 1.1535 = **+0.5225**。
- 預測帶:Δ-ratio 模型 +0.48(以 12ep FP32 讀數)~ 絕對比例模型 +0.61 → **實測落在帶內,驗證成立**。
- ft-pair 實測比例:enhanced R = 1.6760/2.0392 = **0.822**(劣於 baseline-pair 的 0.865);Δ 存活率 = 0.5225/0.8857 = **0.590**(優於 baseline-pair Δ-ratio 0.542)。兩個乘法模型各偏一邊,真值居中。
- 代價:MBSTOI 0.8209(< gate 0.835)、STOI −0.84pt vs deploy——此點是 ablation/可選操作點,主 deploy 仍為 ep3(0.8357)。
- 完整敘事鏈(−6..9, wb):FP32 ceiling Δ+0.9327 → 量化後實測 Δ+0.5225(保留 59% 改善量)→ vs P4 Δ+0.78:deploy-arch PESQ-ft 把差距從 0.52 縮到 0.26。

下一步候選:KD(teacher = w12-sat FP32)+ 小劑量 torch_pesq 的 QAT——teacher 已是 PESQ-優化,不再與目標打架,理論上可把 Δ 推向 +0.6 帶;以及 gate-constrained 版(早停在 MBSTOI ≥ 0.835 的最後一個 epoch)。

### 2.9 QAT 操作點掃描:KD+spatial 與 spatial-rebalance(2026-07-10)⭐

plain-QAT(§2.8)之後兩輪 deploy-arch 變體,均 W4A8 全 stack、可直接走 export 上晶片(VCTK,750/SNR,dir-20 鏈):

| Variant(full 均值) | PESQ wb | PESQ nb | MBSTOI | STOI | ILD dB | IPD deg |
|---|---|---|---|---|---|---|
| Deploy(主表) | 1.5932 | 2.2985 | 0.8357 | 0.8683 | 3.308 | 15.94 |
| KD+spatial(teacher=w12-sat, ILD10/IPD10/STOI2/PESQ12, ep10) | 1.7597 | 2.4465 | 0.8294 | 0.8638 | 3.371 | 16.55 |
| plain-QAT w12(§2.8) | 1.8660 | 2.5033 | 0.8209 | 0.8599 | 3.492 | 17.28 |
| **rebalance ⭐(plain-QAT ep10 續訓, ILD4/IPD12/STOI10/PESQ12, lr1e-5, 15ep)** | **1.8564** | 2.4960 | 0.8234 | 0.8608 | **3.349** | 16.93 |

- **rebalance 幾乎嚴格支配 plain-QAT**:PESQ −0.0096 換 ILD −0.143(收復 78% 空間損傷)、IPD −0.35、MBSTOI +0.0025。推薦的 PESQ-加強變體(取代 §2.8 的 plain-QAT 位置)。
- KD 結論:teacher-as-target 能減緩 PESQ 造成的空間退化,但上限錨在 teacher 的線索水位,無法優於 deploy;直接權重再平衡(本節)更有效率。
- vs mono 範式(§2.7):rebalance(W4A8)wb 1.856 > NTU-FP32 1.820、距 SNU-FP32 1.949 剩 0.093,ILD 3.349 vs 5.47/5.71 —「量化晶片模型品質貼近 FP32 mono、空間保真 1.6~1.7 倍優」成立。
- 所有 PESQ 變體 MBSTOI 均 < gate 0.835,定位 ablation/可選操作點;主 deploy 不變(ep3)。
- ckpt:`outputs/pesq_finetune/qatw12_rebal_ild4_ipd12/`;結果:`results/pesq_eval/qatrebal_verify/`。eval 鏈自本輪起新增 per-ear SI-SDR 欄(rebalance VCTK full 9.21 dB);loss 端新增 `snr_loss_mode=sisdr_tied`(scale-tied 雙耳安全 SI-SDR,未啟用,備 A/B)。

### 2.10 Gate sweep 收官:兩點故事 + PESQ-ceiling ablation vs mono 範式 — appended 2026-07-11

Gate-holding PESQ sweep(27 ckpts)+ rescue 終局 **NO_GATE_PASS + RESCUE_NO_PASS**;完整判決、offset 定律、贏家詛咒見 [RESULTS_GATE_SWEEP_RESCUE_2026-07-11.md](RESULTS_GATE_SWEEP_RESCUE_2026-07-11.md)。採納**兩點故事**,並確立量測 Pareto 階梯(MBSTOI ↔ wb PESQ,c=canonical / s=scaled):

| 操作點 | vMBSTOI | VCTK wb | 定位 |
|---|---|---|---|
| ⭐⭐ deploy ep3 | 0.8357 c | 1.5932 c | 主模型:唯一雙 gate PASS,ILD/IPD/ITD/IC 主張全由它扛 |
| ⭐ `sw_w6cont_s12_v4e1` | 0.8308 c | 1.7092 c | PESQ-boosted 副點:vs deploy +0.1160/+0.0973 wb,MBSTOI −0.0049,ILD 3.3363 仍優於 noisy |
| rebalance(§2.9) | 0.8234 s | 1.8564 s | 更深的 PESQ 檔位(MBSTOI −0.012) |
| w24 ep19 | 0.8133 s | 1.9811 s | PESQ-ceiling ablation(本節主角) |

**w24 ep19 vs mono 範式(P4/SNU、P1/NTU 復現),PESQ 與 ILD 並列**(全 −6..15 均;d69 = −6..9;w24 為 W4A8 零乘法 stack、mono 為 FP32 — 比法對我們不利):

| | VCTK wb(Δ) | VCTK d69 Δwb/Δnb | VCTK ILD(vs noisy 3.7141) | TIMIT wb(Δ) | TIMIT ILD(vs noisy 3.2902) |
|---|---|---|---|---|---|
| w24 ep19(W4A8) | **1.9811(+0.7407)** | **+0.6376**/+0.6552 | **3.5795(−0.13 改善)** | 1.9309(+0.6641) | **3.2943(持平)** |
| SNU repro(FP32) | 1.9494(+0.7089) | +0.5618/**+0.6980** | 5.7056(+1.99 惡化) | **1.9351(+0.6684)** | 5.7940(+2.50 惡化) |
| NTU repro(FP32) | 1.8193(+0.5788) | +0.4307/+0.5834 | 5.4689(+1.75 惡化) | 1.7642(+0.4974) | 5.5032(+2.21 惡化) |

- **賣點句型:把本架構推到 PESQ 天花板(放棄 gate),單耳品質已與 FP32 mono 打平/反超(VCTK wb 贏 SNU +0.032、d69 Δwb +0.076;TIMIT −0.004 打平),而 ILD 仍改善/持平 — mono 範式在任何品質水位都是 +1.75~+2.50 dB 毀滅。**「品質可以買到,空間線索買不到」。
- nb 誠實列輸(VCTK 2.5743 vs 2.6372;d69 Δnb −0.0428):torch_pesq 訓練目標為 wb,「w24 贏 SNU」引用時必須帶 wb 限定詞。
- MBSTOI 不做跨範式 head-to-head(§2.7 既有結論;EC 補償機制的參數級解釋見 gate-sweep RESULTS §2.6)。
- Tier-2 結論不變:**gate-holding** 點對 SNU Δ 在任何頻寬輸 ~0.19–0.20;w24 的 wb 勝利以 gate 為代價。

## 3. Cross-cuts

- **Δ 是唯一可跨集比較的 PESQ 量**:noisy 起點差 0.8(1.15 vs 1.95)且 PESQ 低分段壓縮,絕對值不可直比;本工作在更難條件下仍以 Δ 超過 P4。
- **量化代價 per-metric 不均**:可懂度指標幾乎免費(≤3%),PESQ 付 9–13% — 65-bin STFT + W4A8 犧牲的正是 PESQ 度量的細頻譜結構。
- **PESQ finetune 在 FP32 上效率倍增**:同 w=12,FP32 得 +0.41 wb(可懂度近零損),量化模型上僅 +0.23(MBSTOI −0.005)。順序應為「FP32 ft → QAT」而非「QAT → ft」。
- **w12 val 未飽和**(top-5 = 最後 5 epochs),+0.8857 是 12-epoch 讀數,上限未封頂。
- 訓練/評測多次被 GPU 程序清理事件(SIGTERM,與 Claude client 斷線重啟時間窗吻合,詳見 canary log `fp32_512_pesq_ft/canary_ps.log`)截斷;所有數字均來自完整跑完的 eval,不受影響。

## 4. Recommendation

晶片論文採三層敘事:(1) deploy 實測(gate pass、Δwb +0.26、ΔSTOI +5.4pt、MBSTOI/ILD/IPD 獨家、−10 dB 純音訊 OOD 仍 Δ+0.15 MBSTOI);(2) 演算法層 ceiling:同架構譜系 FP32 以 PESQ-aware finetune 達 Δwb +0.8857,超過最強 mono 競品 P4 的 +0.78 — 證明 PESQ 差距是「目標函數 + 部署精度的取捨」而非架構缺陷;(3) 縮放比外推 deploy 可達 Δ≈+0.61 (wb) / +0.64 (nb),註明為外推。下一步(未跑):w12-ft 權重走一遍 QAT/量化鏈實測外推值;w12 續訓至飽和;MetricGAN 提高 gan_weight / 移除 D 重啟截斷後重估上限。

## 5. Reproduction

```bash
# deploy canonical (nb@8k + STOI)
cd 18_w4a8_pow2_mixed_lmul_hw_handoff_20260530
CKPT=$PKG/ckpt/strict_core_pow2_blockfloat_deploy_ep3_vctk0.8357_timit0.7898.ckpt \
TAG=pow2_ep3_full_nb8k VCTK_ONLY=0 ACT_FP8_MODE=calibrated_pow2_e4m3 \
BCCTN_ACTIVATION_CALIB_PATH=$PKG/ckpt/activation_calibration_p999_b64.json \
GPU_ID=<free> bash run_apot_descale_eval.sh

# -8/-10 dB synthesis + eval
python 20_pesq_finetune/scripts/gen_lowsnr_ext.py
TEST_DIR=... CLEAN_DIR=... OUT=... GPU_ID=<free> bash $PKG/run_lowsnr_ext_eval.sh

# FP32-512 chain (baseline / w12-ft / MetricGAN evals + trainings)
bash 20_pesq_finetune/fp32_512_pesq_ft/run_queue_eval2.sh          # C/D/B
bash 20_pesq_finetune/fp32_512_pesq_ft/run_queue_phase4_evalft.sh  # w12-ft eval
bash 20_pesq_finetune/fp32_512_pesq_ft/run_phase6_evalmgan.sh      # MetricGAN evals
# trainings: code/train.py (+model.pesq_weight=12), code/train_metricgan.py (+model.gan_weight=12)
```

結果目錄:`18_.../results/apot_descale_eval/ft_pow2_ep3_full_nb8k/`、`20_pesq_finetune/results/pesq_eval/{fp32_512_ref_full, fp32_512_pesqft_w12_best, fp32_512_metricgan_bestval, deploy_lowsnr_ext}/`。
