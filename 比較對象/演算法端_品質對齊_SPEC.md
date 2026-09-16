# 品質評估對齊 SPEC（給演算法端顧問）

> 目的：硬體 BCCTN 加速器已完成 gate-level 實作，準備投稿 SE/hearing-aid 晶片論文。
> 需要「能與 5 篇 SOTA SE 晶片**直接對比**」的品質數字進 comparison table。
> 目前模型在 binaural VCTK/TIMIT 上有 MBSTOI，但對手用的是 **mono 的 CHiME2 / VoiceBank+DEMAND**，無法直接比。
> 這份 spec 列出「需要演算法端協助產生/釐清的東西」。

---

## 0. 我方現況（deploy checkpoint）

- 模型：Binaural Complex Causal Transformer（complex DW-sep conv encoder/decoder + complex MHA），16 kHz，W4A8。
- loss 權重：`stoi=10, ipd=10, snr=1, ild=1`（優化**可懂度+空間線索**，非 PESQ）。
- 目前 eval（VCTK / TIMIT，binaural，平均 SNR −6…+15 dB）：

| 指標 | VCTK noisy→enh | TIMIT noisy→enh |
|---|---|---|
| MBSTOI | 0.786 → 0.836 | 0.731 → 0.790 |
| PESQ | 1.240 → 1.593 | 1.267 → 1.656 |
| SegSNR | −0.83 → 1.46 | 0.10 → 2.66 |
| ILD err (dB) | 3.71 → 3.31 | 3.29 → 3.08 |
| IPD err (deg) | 15.43 → 15.94 | 21.10 → 20.39 |

問題：PESQ 絕對值偏低（疑似含殘響 + 低 SNR 起點 + 模型不優化 PESQ），且資料集與對手不同 → **無法和對手的 PESQ/STOI 直接比**。

---

## 1. 要對齊的對象（5 篇 SOTA 晶片的資料集/條件）

| 論文 | 資料集 | SNR | 殘響 | 聲道 | 取樣率 | 報的品質 |
|---|---|---|---|---|---|---|
| P1 NTU JSSC'20 | NOISEX-92+IEEE | −6…+6 | 無 | mono | 16k | WAR +41% |
| P2 PKU CICC'26 | **VoiceBank+DEMAND** | +2.5…+17.5 | 無(anechoic) | mono | 16k | **PESQ 3.34 / STOI 95.1** |
| P3 SNU JSSC'24 | **CHiME2** | −6…+9 | 有(居家) | mono | 16k | SDR 14.0 / PESQ 2.73 / STOI 91.4 |
| P4 MIT VLSID'25 | CHiME2 | −6…+9 | 有 | mono | 8k | PESQ 2.79 |
| P5 Tsinghua ASSCC'25 | TMSV(+影像) | −10 | 不明 | mono+visual | 16k | PESQ 2.13 |

→ 共同 benchmark：**CHiME2**（P3/P4）與 **VoiceBank+DEMAND**（P2）。其中 **VoiceBank ⊂ VCTK 語料家族**，與我方語音最接近。

---

## 2. 需要演算法端協助的三件事

### A.（最優先）釐清目前 binaural VCTK/TIMIT 資料集的精確設定
要寫進論文 dataset 段，也決定 PESQ 為何偏低。請逐項回答：
1. **噪音來源**？（哪個 noise corpus / 噪音類型）
2. **是否含殘響/回音**？用 BRIR/RIR 卷積嗎？**RT60 多少**？HRTF 還是 BRIR？來源（KEMAR / 量測 / 模擬）？
3. **clean target 是 dry（乾淨無殘響）還是 reverberant（含殘響、只去噪）**？
   → 決定任務是「去噪」還是「去噪+去殘響」（後者解釋了低 PESQ）。
4. **SNR 點**與分佈？（目前 eval 用 −6…+15，8 點）
5. **azimuth 設定**：語音/噪音的方位角如何擺？是否同位/異位？
6. 訓練/測試切分、語者數、是否 matched/unmatched（TIMIT 標為 unmatched condition）？

### B. 在共同 benchmark 上產生「可直接比」的數字
1. **首選 VoiceBank+DEMAND**，用**標準協定**（不要自訂）：
   - train SNR `{0, 5, 10, 15} dB`、test SNR `{2.5, 7.5, 12.5, 17.5} dB`
   - DEMAND noise、**anechoic（無殘響）**、16 kHz
2. 用**現有 deploy 模型**跑：
   - 能**直接 inference** 嗎（VoiceBank⊂VCTK，domain 接近）？還是需要 **fine-tune / 重訓**？請評估成本。
   - mono 變體或 binaural-render 皆可，但要**說明清楚是哪種**。
3. 報 **PESQ / STOI / SI-SDR**，目標是對到 **P2 的 PESQ 3.34 / STOI 95.1**。
4.（次選）若有資源，CHiME2 也跑一份（對 P3/P4），但它 reverberant，PESQ 會較低。

### C. 補齊指標與「改善量 Δ」
1. **每個資料集、每個 SNR 點都要報 noisy baseline + enhanced**，並列出 **Δ = enhanced − noisy**（對手多半只報 enhanced，我們報 Δ 較有利、且條件可解釋）。
2. 指標 set（齊全）：
   - **MBSTOI**（雙耳可懂度，主打）
   - **PESQ（per-ear，左右各算）**
   - STOI / ESTOI
   - **SI-SDR**（建議加，比 SegSNR 通用）
   - SegSNR
   - **ILD error、IPD error**（雙耳空間保真，對手沒有 → 我方獨家）
3. SNR 點建議：報 `−6…+9`（對齊 CHiME2 軸）；**可選加 −8/−10**（對上 P5 的 −10dB 招牌）。

---

## 3. 策略註記（讓顧問理解方向）

- **MBSTOI / ILD / IPD 是我方雙耳獨家強項**（對手單聲道量不出）→ 論文主打。
- **PESQ 絕對值不可跨資料集比**（殘響 + 低 SNR 起點 + 模型優化 MBSTOI 非 PESQ）；策略是：(a) 報 Δ，(b) 在共同集（VoiceBank）上跑可比數字，(c) 在論文寫明「含殘響/dereverb 故絕對 PESQ 不可比」。
- **不要為了好看的 PESQ 改 loss 或改任務**——MBSTOI/空間是我方賣點，維持。VoiceBank 那份只是為了「comparison table 有一個可比格子」。

---

## 4. 交付物（希望拿到）

1. **一張完整表**：dataset × SNR × {noisy, enhanced, Δ}，指標含 {MBSTOI, PESQ(L/R), STOI, SI-SDR, SegSNR, ILD, IPD}。
2. **VoiceBank+DEMAND 上的 PESQ / STOI**（標準協定，對 P2）+ 說明是 mono 還 binaural-render、有無 fine-tune。
3. **目前 binaural 資料集設定的文字描述**（A 的答案，可直接放進論文 dataset 段）。
4.（可選）CHiME2 結果（對 P3/P4）。

---

## 5. 給顧問的快速 checklist（最少先回這 4 題）
- [ ] 噪音來源 + 是否含殘響(RT60)？
- [ ] clean target 是 dry 還 reverberant？（=有沒有做 dereverb）
- [ ] VoiceBank+DEMAND 能否直接 inference，還是要 fine-tune？
- [ ] 能否補出每個 SNR 點的 noisy baseline（算 Δ）？
