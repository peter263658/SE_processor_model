# RESULTS: Mono-SE paradigm baselines (SNU JSSC'24 / NTU JSSC'20 repro) — bilateral MBSTOI/ILD/IPD on the deploy考卷

Date: 2026-07-09
Base candidate: deploy = `18_.../ckpt/strict_core_pow2_blockfloat_deploy_ep3_vctk0.8357_timit0.7898.ckpt`; baselines = `21_mono_se_baselines/results/{snu,ntu}/best_{snu,ntu}.pt`
Purpose: turn the "N/A" MBSTOI/ILD/IPD cells of the chip-paper comparison table (`comparison_table/`) into measured numbers for the mono-SE paradigm.

## 0. TL;DR

| Variant | Params | Precision | VCTK MBSTOI | TIMIT MBSTOI | VCTK ILD err (dB) | TIMIT ILD err (dB) | Verdict |
|---|---|---|---|---|---|---|---|
| Noisy (unprocessed) | — | — | 0.7861 | 0.7311 | 3.7141 | 3.2902 | reference |
| FP32 binaural (09_tied_ft, quant 線祖先) | 542K | FP32 | 0.8411 | 0.7966 | **3.2149 (改善)** | **2.9743 (改善)** | 同精度範式上限 |
| ⭐⭐ Deploy (binaural, W4A8 zero-mult core) | 542K | W4A8 | 0.8357 | 0.7898 | **3.3079 (改善)** | **3.0846 (改善)** | 唯一同時降噪 + 改善 ILD 的量化核心 |
| SNU repro (mono bilateral) | 56,322 | FP32 | 0.8383 | 0.8035 | 5.7056 (**惡化 +54%**) | 5.7940 (**惡化 +76%**) | MBSTOI 打平, 空間線索失效 |
| NTU repro (mono bilateral) | 495,105 | FP32 | 0.8372 | 0.8006 | 5.4689 (**惡化 +47%**) | 5.5032 (**惡化 +67%**) | 同上 |

**誠實結論：MBSTOI 平均值分不開範式**（FP32 + matched data + SI-SDR loss 的 mono 上限與 W4A8 deploy 打平，TIMIT 甚至 +0.011～0.014）。**決定性的差異在 ILD error**：mono bilateral 在每一個 SNR 點都比「完全不處理」更糟（-6dB 時近乎翻倍：VCTK 6.07→9.25 dB），deploy 則在每個低/中 SNR 點都改善。IPD 兩範式都大致保留（mask-based 沿用 noisy phase），不構成區分欄。

## 1. Setup

- **重現對象**（`comparison_table/` 兩篇單耳晶片論文的網路，非其 silicon）：
  - **SNU**: Park et al., JSSC 2024 (28nm)。演算法基底 = TRU-Net（Choi et al., ICASSP 2021, 論文 ref [6]）+ band-separated 改版。照 Fig. 9 逐層表 1:1 重建：512-pt STFT/hop 256 → 128-D Mel magnitude → 低/高頻雙組 DS-conv (5×1, stride 2) encoder → bi-FGRU(h=32)+FC → BS-TGRU(4 GRU, h=8/16/32/32)+4 FC combine → 雙組 transposed DS-conv decoder → sigmoid mask → pinv-Mel → noisy phase iSTFT。**參數 56,322 = 論文 56k 命中**。ReLU6 全層。
  - **NTU**: Lee et al., JSSC 2020 (40nm 2.17mW)。演算法基底 = Lai et al. (APSIPA, ref [13]) 的簡化。照 Table I：256-pt STFT (Hamming, hop 128) → log-mag 129×7 幀 context → Conv 32@1×15×5 → pool÷3 → Conv 64@32×5×3 → pool÷3 → FC 832→480→129 → sigmoid mask → noisy phase。**參數 495,105**。
- **訓練**（對 baseline 全面有利的設定）：FP32、不量化；binaural VCTK trainset 拆單耳（61,938 train / 13,272 val mono 樣本，matched data）；SI-SDR 波形 loss；Adam 1e-3、batch 128、80 epochs。最終 val SI-SDR：SNU 11.297 dB、NTU 12.268 dB（oracle IRM 上限 ~13.7/13.9 dB，兩範式公平）。
- **評估**：bilateral 推論（左右耳獨立過 mono 網路）→ 與 deploy 完全同一考卷：`eval_bilateral.py`（= `20_pesq_finetune/code/eval.py` 僅換模型載入為 `BilateralWrapper`）、750 pairs × 8 SNR (-6…15dB)、VCTK + TIMIT (unmatched)、metric_workers 32。Gate 對照組：deploy full VCTK 0.8357 / TIMIT 0.7898（`ft_pow2_ep3_full`）。
- 已知考卷行為：每個 SNR 點有 3–5 件 MBSTOI "negative dimensions" 邊角案例（回傳 None 排除）——deploy 的 `eval.log` 同樣有 32 件，非 bilateral eval 引入。

## 2. Results

### 2.1 VCTK per-SNR MBSTOI

| SNR | Noisy | Deploy (binaural W4A8) | SNU repro (mono, FP32) | NTU repro (mono, FP32) |
|---|---|---|---|---|
| -6 | 0.5366 | 0.6649 | 0.6741 | 0.6721 |
| -3 | 0.6376 | 0.7310 | 0.7383 | 0.7358 |
| 0 | 0.7264 | 0.7962 | 0.8011 | 0.7988 |
| 3 | 0.7949 | 0.8403 | 0.8436 | 0.8415 |
| 6 | 0.8474 | 0.8767 | 0.8764 | 0.8759 |
| 9 | 0.8897 | 0.9073 | 0.9050 | 0.9059 |
| 12 | 0.9171 | 0.9260 | 0.9255 | 0.9256 |
| 15 | 0.9395 | 0.9431 | 0.9425 | 0.9419 |
| **AVG** | **0.7861** | **0.8357** | **0.8383** | **0.8372** |

### 2.2 VCTK per-SNR ILD error (dB)

| SNR | Noisy | Deploy | SNU repro | NTU repro |
|---|---|---|---|---|
| -6 | 6.0667 | 4.9692 | 9.2500 | 9.0958 |
| -3 | 5.2598 | 4.4118 | 8.1906 | 8.0779 |
| 0 | 4.5349 | 3.9165 | 7.0065 | 6.7325 |
| 3 | 3.9004 | 3.4662 | 6.0101 | 5.7970 |
| 6 | 3.2572 | 3.0103 | 5.0518 | 4.7462 |
| 9 | 2.6608 | 2.5615 | 4.0930 | 3.8083 |
| 12 | 2.2019 | 2.2150 | 3.3049 | 3.0205 |
| 15 | 1.8313 | 1.9129 | 2.7381 | 2.4728 |
| **AVG** | **3.7141** | **3.3079** | **5.7056** | **5.4689** |

### 2.3 VCTK per-SNR IPD error (deg)

| SNR | Noisy | Deploy | SNU repro | NTU repro |
|---|---|---|---|---|
| -6 | 30.74 | 30.23 | 30.35 | 33.55 |
| -3 | 25.43 | 24.99 | 24.63 | 27.67 |
| 0 | 19.61 | 19.60 | 18.56 | 21.30 |
| 3 | 15.53 | 15.85 | 14.53 | 17.06 |
| 6 | 12.00 | 12.70 | 11.13 | 13.35 |
| 9 | 8.84 | 9.94 | 8.25 | 10.12 |
| 12 | 6.53 | 7.89 | 6.19 | 7.70 |
| 15 | 4.77 | 6.35 | 4.58 | 5.89 |
| **AVG** | **15.43** | **15.94** | **14.78** | **17.08** |

### 2.4 TIMIT (unmatched) per-SNR MBSTOI

| SNR | Noisy | Deploy (binaural W4A8) | SNU repro (mono, FP32) | NTU repro (mono, FP32) |
|---|---|---|---|---|
| -6 | 0.4576 | 0.5792 | 0.5910 | 0.5951 |
| -3 | 0.5614 | 0.6716 | 0.6837 | 0.6828 |
| 0 | 0.6548 | 0.7371 | 0.7517 | 0.7521 |
| 3 | 0.7227 | 0.7840 | 0.8007 | 0.7963 |
| 6 | 0.8034 | 0.8414 | 0.8558 | 0.8502 |
| 9 | 0.8463 | 0.8744 | 0.8911 | 0.8845 |
| 12 | 0.8842 | 0.9021 | 0.9153 | 0.9096 |
| 15 | 0.9181 | 0.9282 | 0.9384 | 0.9339 |
| **AVG** | **0.7311** | **0.7898** | **0.8035** | **0.8006** |

### 2.5 TIMIT per-SNR ILD error (dB)

| SNR | Noisy | Deploy | SNU repro | NTU repro |
|---|---|---|---|---|
| -6 | 5.5795 | 4.7789 | 9.8367 | 9.5535 |
| -3 | 4.9769 | 4.3666 | 8.9530 | 8.6861 |
| 0 | 4.0068 | 3.6477 | 7.2416 | 6.8077 |
| 3 | 3.4514 | 3.3142 | 6.3039 | 6.0216 |
| 6 | 2.8720 | 2.8246 | 4.9879 | 4.6514 |
| 9 | 2.2744 | 2.3250 | 3.9100 | 3.5845 |
| 12 | 1.7972 | 1.9031 | 2.9764 | 2.7444 |
| 15 | 1.3634 | 1.5165 | 2.1426 | 1.9764 |
| **AVG** | **3.2902** | **3.0846** | **5.7940** | **5.5032** |

### 2.6 TIMIT per-SNR IPD error (deg)

| SNR | Noisy | Deploy | SNU repro | NTU repro |
|---|---|---|---|---|
| -6 | 38.56 | 37.02 | 39.20 | 41.39 |
| -3 | 31.59 | 29.93 | 31.13 | 33.26 |
| 0 | 26.33 | 24.91 | 25.44 | 27.59 |
| 3 | 22.11 | 21.04 | 20.73 | 22.83 |
| 6 | 16.81 | 16.29 | 15.38 | 17.28 |
| 9 | 13.72 | 13.69 | 12.19 | 14.06 |
| 12 | 11.03 | 11.20 | 9.74 | 11.32 |
| 15 | 8.63 | 9.03 | 7.45 | 8.86 |
| **AVG** | **21.10** | **20.39** | **20.15** | **22.14** |

### 2.7 PESQ / STOI（單耳品質指標,全 SNR 平均）

| Metric | Noisy | FP32 binaural | Deploy (W4A8) | SNU repro (FP32) | NTU repro (FP32) |
|---|---|---|---|---|---|
| VCTK PESQ (wb) | 1.2404 | 1.6780 | 1.5932 | 1.9486 | 1.8195 |
| TIMIT PESQ (wb) | 1.2667 | 1.7576 | 1.6558 | 1.9351 | 1.7640 |
| VCTK STOI | 0.8241 | 0.8737 | 0.8683 | 0.8808 | 0.8761 |
| TIMIT STOI | 0.7666 | 0.8300 | 0.8228 | 0.8503 | 0.8426 |

Deploy 的 STOI/PESQ 取自 `ft_pow2_ep3_full_stoi`（同 ckpt 重跑加 STOI 欄；MBSTOI/PESQ-wb/STOI 的 noisy 基線與本次 run 完全一致）。PESQ (nb) 基線與 deploy 舊 run 的 noisy 參考有版本漂移（1.86 vs 1.78）,不放入對比。

**單耳品質指標上 FP32 mono baseline 全面高於 W4A8 deploy**（PESQ +0.28~0.36、STOI +0.01~0.03）——預期內：SI-SDR 波形 loss 直接最佳化波形保真、FP32 無量化損失;deploy 的 loss (SNR+STOI+ILD+IPD) 不含 PESQ 項且 PESQ 本就是已知弱項（`20_pesq_finetune` 的動機）。這強化了第 4 節的呈現建議：比較表不要在 PESQ/STOI 欄與 mono 範式對打,主戰場是空間線索欄。

### 2.8 FP32 binaural reference（同精度範式對比行,2026-07-09 重跑）

FP32 雙耳參考 = `09_weight_tied_mha/outputs/finetune/2026-05-09_22-29-20/.../epoch=57-step=56144.ckpt`（strict-core 量化線的 FP32 祖先,tied-MHA）。以 `eval_fp32.py`（同一考卷 + 現行指標組）重跑,**MBSTOI/ILD/IPD 與 09 舊紀錄逐位一致**（VCTK 0.8411/3.2149/15.41、TIMIT 0.7966/2.9743/19.62）,新增 PESQ(wb) 1.6780/1.7576、STOI 0.8737/0.8300。

Per-SNR（VCTK MBSTOI / ILD）：-6: 0.6745/4.7540、-3: 0.7387/4.2367、0: 0.8031/3.7844、3: 0.8464/3.3747、6: 0.8817/2.9495、9: 0.9108/2.5234、12: 0.9286/2.1934、15: 0.9449/1.9032。TIMIT：-6: 0.5908/4.5558、-3: 0.6800/4.1888、0: 0.7475/3.4951、3: 0.7921/3.1897、6: 0.8473/2.7253、9: 0.8793/2.2586、12: 0.9053/1.8722、15: 0.9303/1.5085。原始檔：`results/bilateral_eval/fp32_binaural/`。

**同精度 (FP32 vs FP32) 結論**：MBSTOI 仍打平（VCTK +0.0028 binaural、TIMIT -0.0069）；PESQ/STOI mono 仍略高（loss 設計:mono 純 SI-SDR,binaural 的 loss 帶 ILD/IPD 項,用一點單耳品質換空間保真）；**ILD 差距不變地決定性**（binaural 3.21/2.97 改善 vs mono 5.71/5.79 惡化,~1.8× 差）。量化成本另計：FP32→W4A8 MBSTOI -0.0054/-0.0068、ILD +0.09/+0.11 dB、PESQ -0.08/-0.10——範式差距遠大於量化損失。

### 2.9 與原論文報告分數的校準（PESQ/STOI 回推）

原論文自報分數（各自的 benchmark,非本考卷）：

| 來源 | Benchmark | 模型狀態 | SDR | PESQ | STOI |
|---|---|---|---|---|---|
| SNU JSSC'24 (chip, 50% hop) | CHiME2 WSJ0, -6..9dB | 4-bit log 量化 | 14.00 | 2.73 | 91.44 |
| SNU 論文內對 NTU [15] 的 FP32 重現 | CHiME2 WSJ0, -6..9dB | FP32 | 12.21 | 2.38 | 89.08 |
| NTU JSSC'20 (chip) | IEEE+NOISEX-92 | INT 量化 | 只報 WAR（無 PESQ/STOI） | — | — |

本考卷對齊 SNR 範圍（VCTK, **-6..9dB** 六點平均,bilateral per-ear）：SNU repro PESQ 1.715 / STOI 85.73;NTU repro 1.584 / 85.19;FP32 binaural 1.488 / 84.90;noisy 1.153 / 78.87。

三個結論：
1. **資料集位移主導,絕對值不可直接相減**：同一個 SNU 網路,CHiME2 上 2.73、本考卷 1.72——位移約 **-1.0 PESQ / -5.7 STOI 點**,來源是考卷難度（雙耳 per-ear 評分含 head-shadow 遠耳、HRTF 混音、DEMAND 噪聲;本考卷 noisy PESQ 地板僅 1.15）。他們的 2.73 拿到本考卷不會是 2.73。
2. **排序交叉驗證通過**：SNU 論文內部 SNU>NTU（PESQ +0.35、STOI +2.36 點,惟彼為 chip-vs-FP32）;本重現同向 SNU>NTU（PESQ +0.13、STOI +0.5 點,FP32-vs-FP32）——方向一致,重現可信。
3. **回推實際晶片在本考卷的分數**（paper 內部自報損失轉移）：
   - **SNU silicon ≈ 本 repro − 小額量化損**。其 4-bit log 量化按 Fig. 13 與 INT8 相當、Fig. 10 全部優化僅 -0.15 SDR;按其 SDR↔PESQ↔STOI 斜率（1.79 SDR ↔ 0.35 PESQ ↔ 2.36 STOI 點）折算 ≈ **-0.03 PESQ / -0.2 STOI 點**,即本考卷約 PESQ 1.68-1.69、STOI 85.5。
   - **NTU silicon 上限 = 本 repro,低 SNR 顯著更低**：其自報 DFT 320→256 損 <5% WAR、量化再損高 SNR <1% / **低 SNR 至 14% WAR**——-6dB 端的真實晶片表現遠低於本 FP32 重現。
   - 兩者的 MBSTOI/ILD 結論不受影響：量化只會讓兩耳 mask 更不一致,ILD 惡化只增不減——本表對 baseline 的每一項不確定性都朝對其有利方向取值。

## 3. Cross-cuts

- **MBSTOI 不區分範式**：mask-based mono SE 沿用各耳 noisy phase、每耳可懂度確實提升，MBSTOI 的 better-ear/EC 建模因此跟著提升——與 deploy 差距全 SNR ≤0.01（VCTK）；TIMIT 上 FP32 mono 甚至高 +0.011～0.014（deploy 是 W4A8,量化成本包含在內）。原先由 Westhausen (0.395 vs 0.493) 外推的「MBSTOI 會明顯較低」預期不成立——該文獻是多麥克風、更低 SNR 的助聽器 setup,不可直接移植。
- **ILD error 是決定性欄位**：兩個 mono baseline 在**每個 SNR、兩個資料集共 32 個點全數惡化於 noisy**（相對惡化 +47%～+76%,-6dB 時 VCTK 9.25/9.10 vs noisy 6.07）。deploy 在 -6…6dB 全部改善、僅高 SNR 微幅持平。機制：兩耳 magnitude mask 各自從不同噪聲實現獨立估計 → 同一 T-F bin 縮放不一致 → 響度差直接扭曲；雙耳聯合模型 + ILD/IPD loss 項（`L = SNR + 10·STOI + ILD + 10·IPD`）在結構上就能避免——mono 範式**連把 ILD 放進 loss 都不可能**（forward 只看得到一隻耳朵）。
- **IPD 兩範式打平**（noisy phase 沿用 vs deploy 的複數 mask），不建議做為主張欄位；SNU 高 SNR 的 IPD 甚至優於 deploy。
- SNU vs NTU 幾乎同分（MBSTOI Δ≈0.001–0.003、ILD Δ≈0.2–0.3 dB）——範式主導,網路容量（56k vs 495k）次要。

## 4. Recommendation

比較表的呈現要把主張從「MBSTOI 較高」改為「**同等 MBSTOI 下唯一保留/改善空間線索,且是 W4A8 零乘法核心**」：

- 表格加兩行：`Mono-SE paradigm ref. (network of [SNU ref], retrained on our data, float, per-ear)` 與 NTU 同格式；MBSTOI 欄照實填 0.8383/0.8035、0.8372/0.8006,**ILD error 欄以粗體對比 5.71/5.79 vs 3.31/3.08**（並標注 noisy = 3.71/3.29,mono 比不處理更糟）。
- 敘事主軸：助聽器的空間聽覺（定位、安全、cocktail-party）依賴 ILD/IPD;mono 範式即使 FP32 + matched data 也結構性破壞 ILD;本晶片以 1/9 參數的 56k 級網路（vs NTU 495k）同級 MBSTOI 且唯一改善 ILD——且是量化後 (W4A8) 對 FP32 上限的比較。
- 誠實標注（堵 reviewer）：baselines 為 FP32 上限、matched training、現代 SI-SDR loss;量的是範式非該晶片 silicon。
- 後續可選：FP32 BCCTN 在同考卷的數字（18 的 `control` 是量化變體 0.8355,非 FP32）補「同精度」對照行;以及「bilateral 共用 mask」變體(保 ILD 犧牲降噪)做魯棒性附註。

## 5. Reproduction

```bash
cd /raid/R12K41024/21_mono_se_baselines
# 訓練 + 兩個 bilateral eval(嚴格序列、單 GPU;此機背景任務須 setsid 脫離,見 memory: server-gpu-process-gotchas)
setsid nohup env GPU_ID=1 bash run_full_pipeline.sh > logs/pipeline.log 2>&1 < /dev/null &
# 或單獨:
CKPT=results/snu/best_snu.pt TAG=snu GPU_ID=1 bash run_bilateral_eval.sh
CKPT=results/ntu/best_ntu.pt TAG=ntu GPU_ID=1 bash run_bilateral_eval.sh
# deploy 對照組數字: 18_.../results/apot_descale_eval/ft_pow2_ep3_full/{vctk,timit}_paper_style_results.csv
```

Per-SNR 原始檔：`results/bilateral_eval/{snu,ntu}/{vctk,timit}/snr_*/average_results.txt`（+ `detailed_results.csv` 每檔明細）。訓練曲線：`results/{snu,ntu}/train_log.jsonl`。
