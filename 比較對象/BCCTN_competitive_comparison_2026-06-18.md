# BCCTN 加速器 vs. SOTA 語音增強晶片 — 競爭力分析（5 篇對照）

> 建立 2026-06-18，更新 2026-06-19（加入 Paper 4 MIT、Paper 5 Tsinghua + 資料集/改善量分析）
> 目標製程：TSMC 28nm　|　比較對象：本資料夾五篇論文
> 數字標註「實測 / 估算 / 投影」，請依標註引用。

---

## 【最新狀態與決策記錄 — 2026-07-01｜以此為準,與下方舊內容衝突處以此為準】

### 最終比較集
P1 NTU(JSSC'20) / P2 PKU(CICC'26) / P3 SNU(JSSC'24) / P5 Tsinghua(ASSCC'25) / 本設計。
**MIT(VLSID'25) 排除**——教授指示:VLSID 非 SSCS 頂級 venue（保留的 JSSC/CICC/ASSCC 皆頂級）。準則=「只比 IEEE SSCS 頂級場」,一致、可辯護。

### 本設計最新數字（改版 RTL 後, post-layout）
- 操作點:**6.5 MHz / 0.81 V**（0.81V = library Vmin,logic+SRAM macro 都卡這）
- Cycles:**31,556**（改版前 50k,−37%）→ latency **4.86 ms @6.5MHz < 5 ms budget ✓ 達 real-time**
- **Power 2.17 mW ｜ Energy/frame 10.85 µJ**（=2.17mW÷200fps）
- Core area 1.36 mm²（⚠️ 待確認改版後是否變）｜ 記憶體 300 kB
- MAC/frame 1.10M(模型)/~1.5M(雙耳)｜ params 337,876(INT4)｜ PE 128 lane(4×32) zero-mult
- 品質:MBSTOI 0.836(VCTK)/0.790(TIMIT);nb PESQ 2.22(全範圍)/2.00(−6…+9);**無殘響(已確認)**

### Energy/frame 排名（raw）
P5 3.64（**需相機**）< **本設計 10.85** < P3 11.75 < P2 CICC 15.2 < P1 17.36
→ **純音訊 SE 中 energy 最低（贏 CICC/SNU/NTU);整體第 2（只輸靠相機的 P5）。**

### 核心決策（定案）
1. **效率軸不硬追**:energy 報 raw **10.85 µJ**（不 normalize）;**不報 TOPS/W**（CICC CIM 8.04 電爆你）、不報 peak throughput、不報 per-second（都對你不利）。
2. **主打 capability**:唯一 binaural / 複數相位 / **self-attention Transformer(純數位)** / 不需相機 / 能報 MBSTOI·ILD·IPD。
3. **品質**:主打 MBSTOI;PESQ 用 **nb**（對手 CHiME2 慣例）、跨資料集不裸比;VoiceBank/CICC 是 **WB**,不同模式不比。
4. **latency 用 cycles(kCycles)**,不 node-normalize;本設計 31.6k 第 2 低（只輸 P1 小模型）。
5. **寫法**:k 小寫(千)、M 大寫(mega);Model Type 標「self-attn(你) vs cross-modal fusion(P5 的 ABF)」;activation 精度 P1=16b/P3=8b/P4=8b/P5=待查。

### CICC(另一顆 Transformer)context 取捨 — 防守論述（本次重點記錄）
- **CICC 主打 long-range context (>200 frames)**:賭「長 context → 更好品質」;novelty = Contextual-Aware KV-cache management（固定 200-frame KV 7.4MB → 平均 1.17MB,6.32×）+ Reconfigurable D-CIM。**是 real-time SE（125 fps）**,只是 attention 窗很長。
- **代價**:4 MB eDRAM + 39.6M MAC/frame + **15.2 µJ/frame**。
- **本設計相反**:**32-frame 因果窗** → 300 kB + 1.1M MAC + **10.85 µJ** + 低延遲,外加 binaural/複數。
- **防守句(可直接用)**:*"CICC targets long-range context (>200 frames) at a 4-MB eDRAM KV-cache and 15.2 µJ/frame; this work targets low-latency streaming with a 32-frame causal window at 10.85 µJ/frame and 300 kB, adding binaural complex processing that CICC lacks."*
- **防「32-frame 窗太短」攻擊**:(1) 你多 binaural + 相位兩個品質維度;(2) 32-frame 對你目標已足(MBSTOI 0.836);(3) 換到低延遲+低記憶體(助聽器優先項) → 刻意取捨,非能力不足。
- ⚠️ 留 CICC 要避:**不報 TOPS/W**(CIM)、**PESQ 不跨資料集比**(它 WB VoiceBank)、**註明 context 差異**(你 energy 低有部分因 context 小)。

### RTL 未來優化 roadmap（要再降功耗、不破壞能力/延遲）
主攻 **SRAM(45%) + clock(28%)**:FF 減量(→register+clock tree)、clock gating、低漏電 SRAM macro。⚠️ **別減 WBUF bank 數**(傷頻寬/引回 DMA 瓶頸)、**別改 stream**(破壞低延遲與完整功耗論述)。

### 待確認
1. Core area 改版後是否仍 1.36 mm²。
2. 2.17mW 是否為 31.6k-cycle 版的 post-layout（同 flow）。
3. P5 activation precision（查證或標「-」）。
4. 若要 normalize energy:僅作輔助分析(energy ∝ node·V²,對 P5 差距從 3× 縮到 ~1.5×),**主表維持 raw**。

---

## 0. 一句話結論

**在「最低功耗 / 最小面積 / 最高 PESQ / 能量-per-frame」這些主流 benchmark 上，本設計贏不了**——這五篇的低功耗陣營（MIT 3.24 µJ、Tsinghua 3.64 µJ、SNU 0.74 mW）全都靠「簡化任務或外加感測器」換低功耗。

**但本設計仍有競爭力，因為它佔一個沒人重疊的能力區位**：

> **全球首顆「16 kHz 寬頻 + 雙耳(binaural) + 複數值(相位感知) + Transformer(attention)」純音訊語音增強晶片，STFT/iSTFT 全部 on-chip，且不需任何額外感測器（無相機）。**

低功耗陣營怎麼換到低功耗：
- **MIT**：8 kHz 窄頻 / mono / magnitude-only / **STFT off-chip**
- **Tsinghua**：**外加攝影機（audio-visual）** / 面積爆炸 5.76 mm² / mono
- **SNU**：mono / conv+GRU 小模型（56k 參數）
- **NTU**：小 CNN / 無 attention

→ 競爭論述：**比「能力 / 面積、能力 / 瓦、系統完整度、感測器最簡」，不是比絕對功耗數字。**

---

## 1. 主規格對照表（raw，各自操作點）

| 項目 | P1 NTU JSSC'20 | P2 PKU CICC'26 | P3 SNU JSSC'24 | P4 MIT VLSID'25 | P5 Tsinghua ASSCC'25 | **本設計 BCCTN** |
|---|---|---|---|---|---|---|
| 製程 | 40 nm | 22 nm | 28 nm | 28 nm | 22 nm ULP | **28 nm** |
| 模型 | 小 CNN+FFT mask | Transformer (TSE) | Sep-Conv+帶分離GRU | CR-CED conv enc-dec | CNN 多模態融合 | **複數 Causal Transformer+complex conv** |
| **模態 / 感測器** | 純音訊 | 純音訊 | 純音訊 | 純音訊 | **音訊+視覺（需相機+IMU）** | **純音訊（雙耳麥克風）** |
| **取樣率** | 16 kHz | — | 16 kHz | **8 kHz 窄頻** | 16 kHz | **16 kHz 寬頻** |
| **雙耳** | ✗ | ✗ | ✗ | ✗ | ✗ | **✓（唯一）** |
| **複數/相位** | ✗ | ✗ | ✗ | ✗ (mag only) | ✗ | **✓（唯一）** |
| **attention** | ✗ | ✓ | ✗ | ✗ | ✗ | **✓** |
| **on-chip STFT** | ✓ | — | ✓ | **✗ off-chip** | ✓ | **✓** |
| 數值精度 | 16-bit | INT8 W/INT4-24 act (CIM) | 4-bit log | 8-bit | 4-8 bit multi-coding | **INT4 W/FP8 act+FP16 attn，零乘法器** |
| Core 面積 | 4.2 mm² | 1.36 mm² | 0.81 mm² | **0.75 mm²** | **5.76 mm²（最大）** | ~1.2 mm² (gate-sim) |
| 片上記憶體 | 327 kB | **4 MB eDRAM**+64+38kB | 35 kB | 75.5 kB | 146 kB(+weight mem) | 300 KiB |
| 電壓 | 0.6–0.9 V | 0.5–1.0 V | 0.8 V | 0.65 V | 0.65–1.2 V | 0.8 V |
| 頻率 | 5 MHz | 10–450 MHz | 2.5 MHz | 18.5–25 MHz | 50–320 MHz | 20 MHz |
| Power | 2.17 mW | 1.9 mW | 0.74 mW | **0.407 mW**ᵃ | **0.228 mW**(core)ᵇ | 4.76mW@20MHz(gate-sim)／**3.324mW@10MHz(post-layout)** 含STFT |
| Latency (@freq) | 1.88 ms @5MHzᶜ | — | 39.96 ms @2.5MHzᶜ | 7.96 ms @18.5MHz | 1.11 ms @50MHz | **2.5 ms@20MHz／5 ms@10MHz** |
| Frame rate | ~125/s | 125/s | 63/s | 125/s | 62.5/s | **200/s（雙耳）** |
| 能量/frame (=P÷fps) | 17.36 µJ | 15.2 µJ/inf | 11.75 µJ | **3.24 µJ**ᵃ | **3.64 µJ** | 16.6 µJ@10MHz／~15.5 µJ(power-gate idle,SRAM須retention) 見§3 |
| MAC/frame | — | 39.6 M | 0.71 M | — | 1.63 M(VA)/0.96M | 1.10 M(模型)/~1.5M(雙耳HW) |
| SNR 容忍 | −3 dB | — | −3 dB | — | **−10 dB（靠視覺）** | 測到 −6 dB（靠雙耳） |
| 主品質指標 | WAR +41%@−6dB | PESQ 3.34/STOI 95.1 | SDR14/PESQ2.73/STOI91.4 | PESQ 2.79 | PESQ 2.13(−10dB) | **MBSTOI 0.836/0.790** |
| 資料集 | NOISEX-92+IEEE | VoiceBank+DEMAND | CHiME2 | CHiME2 | TMSV(國語+影像) | VCTK+TIMIT(雙耳) |
| 性質 | 量產矽 | 量產矽 | 量產矽 | 量產矽 | 量產矽 | **gate-sim 估計** |

> ᵃ **MIT 的 407 µW / 3.24 µJ 不含 STFT/FFT 與 off-chip processing**（ML-core only，論文自承未整合 frequency transform）。本設計 4.76 mW **含** on-chip STFT/iSTFT。
> ᵇ **Tsinghua 228 µW 為 core**；含感測器的 system power = 0.58 mW（image sensor 0.25 mW + IMU 0.1 mW）。
> ᶜ Latency 定義各篇不一（compute-only vs 含 look-ahead 的 total）。跨表中 JSSC'20≈9.88 ms、ISSCC'23≈7.96 ms（processing）；SNU 原文 total 39.96 ms。
> P2 MAC/frame 39.6 M 大是因為長 context（>200 frame）。Pn 的 PESQ/品質取自各原文 + MIT/Tsinghua 交叉表（已比對一致）。

**能量/frame 慣例** = Power ÷ frame rate（全週期平均）。本設計 4.76 mW 為 PrimeTime gate-sim 量「pre-load+compute 一個 frame」(~2.5 ms 視窗)平均；若 idle clock-gated，全週期平均 ≈ 2.38 mW → **~11.9 µJ/frame**；若視為全 5 ms 週期平均則 23.8 µJ/frame（最壞，待確認 §8）。

---

## 2. Process scaling 正規化（→28 nm）

慣例（亦為 SNU 採用）：**面積 ∝ s²、功耗 ∝ s（線性）**，s = 節點比。

### 面積（→28 nm 等效）
| | 原始 | 28 nm 等效 |
|---|---|---|
| P1 NTU | 4.2 (40nm) | 2.06 mm² |
| P2 PKU | 1.36 (22nm) | 2.20 mm² |
| P3 SNU | 0.81 (28nm) | **0.81 mm²** |
| P4 MIT | 0.75 (28nm) | **0.75 mm²** |
| P5 Tsinghua | 5.76 (22nm) | **9.33 mm²（最大）** |
| **BCCTN** | 1.2 (28nm) | **1.2 mm²** |

→ 本設計 1.2 mm²：小於 P1/P2(scaling後)/**P5(9.3 大很多)**，大於 P3/P4。但 P3/P4 是 mono 小模型、P4 不含 STFT；**「面積 ÷ 能力」本設計領先，且對 P5（音訊-視覺、需相機）面積完勝 ~5×。**

### 功耗（→28 nm 等效）
| | 原始 | 28 nm 等效 |
|---|---|---|
| P1 | 2.17 mW | ~1.52 mW |
| P2 | 1.9 mW | ~2.42 mW |
| P3 | 0.74 mW | **0.74 mW** |
| P4 | 0.407 mW (ML-only) | **0.407 mW** |
| P5 | 0.228 mW core | ~0.29 mW core |
| **BCCTN** | 4.76mW@20MHz(gate-sim)／3.324mW@10MHz(post-layout) 含STFT | （28nm，免 scaling）|

→ raw 功耗本設計最高（最弱軸），P3/P4/P5 同/近節點無 scaling 藉口。**降頻至 10 MHz = 3.324 mW（post-layout）；無電壓 headroom（library 最低 0.81V、SRAM macro 同）**——對手各自靠窄頻/不含STFT/外加相機簡化。**這條軸放棄硬拼，改打能力差異化（§9）。**

---

## 3. raw 功耗的操作點與 DVFS（實測 — 兩個流程，不可直接比）

**已實測兩點**（⚠️ gate-sim 與 post-layout 不同流程，**不可直接比較**；20 MHz post-layout 待補）：

| 操作點 | 流程 | Power | latency | duty |
|---|---|---|---|---|
| 20 MHz | gate-sim | 4.76 mW | ~2.5 ms | 50%（算完 idle 2.5ms）|
| **10 MHz** | **post-layout** | **3.324 mW** | ~5 ms | 100%（剛好填滿 frame）|

1. **降頻是唯一可用的 DVFS 槓桿**：50k cycles/frame，10 MHz 剛好 5 ms 即時。
2. ❌ **0.6 V 不可行（更正先前投影）**：library 最低 **0.81 V**、WBUF/KVBUF macro 也只到 0.81 V → **無電壓 headroom**。先前「0.6V→1.3mW / 6.7µJ」**作廢**。
3. **10 MHz post-layout 實際功耗拆分**（PrimeTime `CHIP_post_10MHz.power`）：

| 成分 | Power | % |
|---|---|---|
| Cell Internal（動態）| 1.897 mW | 57.1% |
| Net Switching（動態）| 0.395 mW | 11.9% |
| Cell Leakage | 1.032 mW | **31.0%** |
| **Total** | **3.324 mW** | 100% |

按 group：**memory(SRAM) 1.510 mW (45.4%，含 0.799 mW leakage＝全部 leakage 的 77%)** > clock_network 0.921 mW (27.7%) > combinational 0.536 (16.1%) > register 0.348 (10.5%)。

→ **動態佔 69%、leakage 僅 31%**（leakage 不高）。意義：
   - **race-to-idle / clock-gating 有效**（多數是動態，隨 duty 下降）。
   - **最大耗電是 SRAM(45%) 與 clock tree(28%)，不是 leakage** → 優化打「低漏電 SRAM macro／bank power-gating／clock-gating」，不是追電壓。
   - **energy/frame @10MHz（100% duty）= 3.324 mW × 5 ms = 16.6 µJ**。
   - **20MHz + power-gate idle**：理想省 idle leakage 2.6µJ→14µJ，但 **77% 的 leakage 是 SRAM（WBUF 全模型常駐、KVBUF 存歷史，idle 須保留→只能 retention，不能整個 gate）**；實際只 gate 得到 logic leakage(~0.23mW)→省 ~0.6µJ → **realistic ~15.5–16 µJ**。
   - **power-gating 對 energy/frame 幫助有限**（leakage 31% 且多為不可 gate 的 SRAM）；其真正價值在 **VAD/standby 靜音段切斷漏電 → 降「平均/日常功耗」**（系統級賣點，與 energy/frame 是兩回事）。
   - ⚠️ 此 16.6 µJ（post-layout）比先前 gate-sim 估的 11.9 µJ 高，主因 **post-layout clock tree（28%）**。論文用 post-layout 值。

---

## 4. TOPS / TOPS·W⁻¹

**慣例**：1 MAC = 2 OP；零乘法器(shift-add)以等效 MAC×2 計。TOPS = OP/s ÷ 10¹²；TOPS/W = TOPS÷Power。

**硬體 MAC 寬度**（[Param_def.sv:53-55]+[PEG.sv]）：W4A8 **128 lane**、FP16 **64 lane**（NUM_PE_PG=32×NUM_PEG=4）。

### 峰值 Peak
| 模式 | N_MAC | @20 MHz | Peak TOPS/W (÷4.76mW) |
|---|---|---|---|
| W4A8 | 128 | **5.12 GOPS** | **1.08 TOPS/W** |
| FP16 | 64 | 2.56 GOPS | 0.54 TOPS/W |

### 實效 Sustained
| MAC/frame | 即時(×2×200fps) | active(÷2.5ms) |
|---|---|---|
| 1.10 M(模型) | 0.44 GOPS / 0.092 TOPS/W | 0.88 GOPS / 0.185 TOPS/W |
| ~1.5 M(雙耳HW) | 0.60 GOPS / 0.126 TOPS/W | 1.2 GOPS / 0.25 TOPS/W |

利用率 ≈ 17%（含 STFT/控制/SRAM 等待，FP16 attn 只 64 lane）。

### 對比
| | BCCTN peak | BCCTN eff. | P1 | P2 | P3 | P4 | P5 |
|---|---|---|---|---|---|---|---|
| TOPS/W | 1.08 | 0.09–0.25 | 1.2–2.18 | **8.04**(CIM) | — | 3.68 | — |
| 領頭指標 | — | — | GOPS/W | µJ=15.2 | µJ=11.75 | **µJ=3.24** | **µJ=3.64** |

→ **不要用 TOPS/W 當主打**（peak 灌水、eff. 被低利用率懲罰；這類 fixed-rate 小 workload 晶片，業界都改報 µJ/frame）。以 ~11.9 µJ/frame 為主、peak 1.08 TOPS/W 為輔（加慣例註腳）。

---

## 5. 雙耳合計 MAC 拆解

Binaural 資料流（[binaural_attention_model.py:133-162]）：**attention 跑 1 次**（兩耳 concat 成 256-dim）、**encoder/decoder 各跑 2 次**（左右耳）。`binaural-total = 1×attn + 2×enc + 2×dec`。

| 元件 | 每次 real MAC | 次數 | 小計 |
|---|---|---|---|
| Encoder (DW-sep complex) | ~0.20 M | ×2 | 0.40 M |
| Decoder (transpose+50%skip+grouped PW) | ~0.20 M | ×2 | 0.40 M |
| attention (in_proj+QKV g4+QK+scoreV+out_proj+output_proj) | ~0.77 M | ×1 | 0.77 M |
| **雙耳合計** | | | **≈ 1.5 M** |

自洽檢查：單程(enc/dec 各1次)=1.17 M ≈ 文件 1.10 M（差 6%）✓。文件 1.10 M = enc/dec 算一次的「模型邏輯 MAC」；雙耳硬體實做 ≈ 1.5 M。
**論文建議**：MAC/frame 報 **1.10 M**（QAT 模型一致、與 lane budget 自洽），描述寫明 binaural、需強調吞吐時附 ~1.5 M。

---

## 6. 逐軸勝負記分表

| 軸 | 結果 | 說明 |
|---|---|---|
| 架構新穎性 | 🏆 **贏** | 唯一 binaural+複數+寬頻 Transformer；五篇無一同時具備 |
| 雙耳+相位 | 🏆 **贏（唯一）** | 對手皆 mono+magnitude-only；助聽器空間聽覺核心價值 |
| 寬頻 16 kHz | 🏆 **贏 MIT** | MIT 僅 8 kHz 窄頻 |
| 系統完整度 | 🏆 **贏 MIT** | on-chip STFT/iSTFT；MIT 407µW 不含 STFT |
| **感測器最簡** | 🏆 **贏 Tsinghua** | 純音訊、無相機/IMU；Tsinghua 靠攝影機（AR眼鏡級系統） |
| **面積（vs P5）** | 🏆 **贏（壓倒）** | 1.2 vs 5.76 mm²（換28nm 9.3）≈5× |
| 延遲 | 🏆 **贏 P3** | 2.5 ms vs P3 39.96 ms；與 P4/P5 同級 |
| 片上記憶體 | 🏆 **贏 P2** | 300 kB vs 4 MB eDRAM |
| 面積 ÷ 能力 | ✅ **贏** | raw 面積非最小(>P3/P4)，但模型重得多+雙耳+含STFT |
| 能量/frame（校正複雜度） | ✅ **平手偏贏 P1/P2/P3** | ~11.9 µJ 跑重模型；raw µJ 輸 P4/P5 |
| raw 面積 | ❌ 輸 P3/P4 | 1.2 vs 0.75/0.81（同28nm） |
| raw 功耗 | ❌ 輸全部 | 4.76 mW；DVFS ~1.3 mW 仍 > 對手（但對手各有簡化） |
| raw PESQ | ❌ 輸/不可比 | 見 §7、§11 |
| TOPS/W | ❌ 輸 P2 | D-CIM，別拼 |
| 成熟度 | ⚠️ 風險 | gate-sim vs 量產矽 |

---

## 7. 品質呈現策略（PESQ 風險）

- 招牌用 **MBSTOI 0.836**（雙耳專屬，對手量不出 → 優勢）+ **ILD/IPD/SegSNR**（雙耳保真，對手沒有）。
- **不要省略 PESQ**（會被讀成藏分數，且五篇都報 → 你整列空白）；改為「報 per-ear PESQ + 解釋」。
- 你 PESQ ~1.6 偏低主因：(1)資料集不同，(2)**疑似含 BRIR 殘響**（noisy PESQ 在 +15dB 仍只 1.58，PESQ 重罰殘響），(3)模型優化 MBSTOI/IPD/SNR 非 PESQ（loss `stoi=10,ipd=10,snr=1,ild=1`）。→ 絕對 PESQ 跨資料集不可比，要寫清楚。
- **最高 CP 值**：跑單聲道變體在 **VoiceBank+DEMAND**，拿可直接對 P2 (+1.37 PESQ) 的乾淨數字（VoiceBank ⊂ VCTK，語料重疊）。

---

## 8. 待確認項

1. **4.76 mW 量測視窗**：active(2.5ms)→11.9 µJ/frame；全5ms週期→23.8 µJ。（已知 PrimeTime 量 pre-load+compute 一個 frame，含 on-chip STFT；需確認視窗是否=2.5 ms。）
2. **1.2 mm²/4.76 mW 是否含 SRAM macro**（repo 僅見 FFT128 模組報告 32,100 µm²/1.10 mW）。
3. 雙耳合計 MAC 精確值（1.1 M vs 1.5 M）→ 建議跑 profiler 定案。
4. 0.6 V/10 MHz DVFS 投影需 timing/leakage 收斂驗證。
5. 建議補測 **−8/−10 dB SNR**，對上 Tsinghua 的 −10dB 招牌（雙耳低 SNR 應更有利）。

---

## 9. 競爭力重評估 — 直接回答「還有競爭力嗎？」

**有，但「低功耗」論述已死，定位必須轉。**

### 9.1 打不過的地方（承認）
- 同/近節點下，功耗(P4 0.407/P5 0.228 mW)、面積(P3 0.75/P4 0.81)、能量/frame(P4 3.24/P5 3.64 µJ)、PESQ(P4 2.79) 都輸。raw 數字戰打不贏。

### 9.2 仍唯一/勝出的地方（論文骨架）
低功耗陣營都「用簡化或外加感測器換功耗」：

| 對手 | 它怎麼省 | 你的反制 |
|---|---|---|
| MIT (3.24µJ) | 8kHz窄頻/mono/mag-only/STFT off-chip | 16kHz寬頻/雙耳/複數/含STFT 完整系統 |
| Tsinghua (3.64µJ) | **外加攝影機**/面積5.76mm² | 純音訊無相機/面積1.2mm²(1/5)/可放耳內 |
| SNU (0.74mW) | mono conv+GRU 56k 小模型 | 雙耳複數 Transformer 338k |
| PKU (Transformer) | mono實數/4MB eDRAM | 雙耳複數/300kB |

**最有力 framing**——把「雙耳」定位成你的 auxiliary modality，對照 Tsinghua 的「視覺」：
> Tsinghua 用**外加視覺感測器**達 −10dB 抗噪；本設計用**雙耳空間線索 + 複數相位 + attention（零額外感測器、無相機）**達低 SNR 抗噪——同目標、零額外硬體、1/5 面積、可放耳內。

### 9.3 論文必做 3 動作
1. Comparison table **加「Modality / Sensor」與「System scope」兩列**，標明 MIT 不含STFT、Tsinghua 需相機——擋住 3.24/3.64 µJ 直接比。
2. 品質補共同資料集（VoiceBank+DEMAND）單聲道 PESQ/STOI；主打 MBSTOI/ILD/IPD。
3. 功耗用 DVFS 投影 + 能力正當化；誠實標 gate-sim。

**底線**：raw 數字戰打不贏，但「能力 / 系統完整度 / 雙耳空間聽覺 / 寬頻相位 / 感測器最簡」這個區位，五篇沒人能跟你比。**競爭力在，定位要轉。**

---

## 10. 資料集對照

| | 資料集 | 取樣率 | 聲道 | 與你關係 |
|---|---|---|---|---|
| P1 NTU | NOISEX-92 噪音+IEEE sentences | 16 kHz | mono | 無關 |
| P2 PKU | **VoiceBank+DEMAND** | — | mono | **最接近**：VoiceBank⊂VCTK 同語料家族，但 mono vs 你雙耳 |
| P3 SNU | **CHiME2 (WSJ0)** | 16 kHz | mono | 不同（CHiME2 是硬體SE圈共同benchmark） |
| P4 MIT | **CHiME2** | 8 kHz | mono | 不同 |
| P5 Tsinghua | **TMSV（國語+影像）** | 16 kHz | mono+影像 | 不同（音訊-視覺） |
| **你** | **VCTK+TIMIT（雙耳渲染）** | 16 kHz | **binaural** | — |

要點（P2 排除後）：
- **沒有任何一篇用跟你相同的資料集**；沒人用 TIMIT、沒人用雙耳渲染。
- **CHiME2 是剩餘對手（P1/P3/P4）的共同 benchmark，且全用 NB-PESQ** → 要跟最多篇可比就**跑 CHiME2、報 NB**（它 mono，你 binaural 仍是差異）。
- P2(VoiceBank/WB) 已排除（CIM）→ **不再需要 WB、不再需要 VoiceBank**。
- 你測到 −6 dB（CHiME2 也 −6），但你雙耳 + 你的 noisy 起點更低 → **條件不比對手寬鬆。**

---

## 11. 改善量（Δ = enhanced − noisy）對照

### 絕對 enhanced PESQ（NB；主要呈現）— P2(CIM) 已排除
| 排名 | 晶片 | 資料集 | enhanced PESQ (NB) |
|---|---|---|---|
| 1 | P4 MIT | CHiME2 (8kHz) | 2.79 |
| 2 | P3 SNU | CHiME2 | 2.73 |
| 3 | P1 NTU | CHiME2 (SNU FP32 re-impl) | 2.38 |
| **4** | **你 VCTK** | VCTK | **2.22**(−6…+15)／2.00(−6…+9) |
| 5 | **你 TIMIT** | TIMIT | **2.14** |
| 6 | P5 Tsinghua | TMSV(−10dB+視覺) | 2.13 |

→ 你 NB PESQ ~2.0–2.22 與 CHiME2 群（2.38–2.79）同級距、贏 P5；資料集不同（你 VCTK/TIMIT vs 他們 CHiME2）→ 要完全可比就跑 **CHiME2（NB）**。
> **P2(PKU) 排除於品質/效率比較**：D-CIM（compute-in-memory）為不同設計範式，與純數位不可直接比。丟 P2 後 → **全部 NB-PESQ（不再需要 WB）**、乾淨同資料集目標 = **CHiME2**（非 VoiceBank）；且你成為被比者中**唯一 Transformer-based 純數位**。

### ⚠️ 以下 Δ/SNR 對齊分析為次要（CHiME2 noisy 不可靠，見下）
1. **PESQ 模式**：剩下 P1/P3/P4 全 CHiME-2 = **NB-PESQ** → 你用 nb 數字即可（wb 只在已排除的 P2 才需要）。
2. **對齊 SNR**：對 P3/P4 用 **−6…+9**（你 nb −6…+9 = 2.00）。

### 各篇 noisy baseline（含出處 — 你之後要引用）
| 論文 | 資料集 | noisy baseline | 模式 | 出處 |
|---|---|---|---|---|
| ~~P2 PKU~~（CIM，排除）| ~~VoiceBank+DEMAND~~ | ~~1.97 / 92.1 (WB)~~ | — | 已排除；留存：Valentini 標準值 + P2 Fig 9 |
| P3 / P4 | CHiME-2 (avg −6…+9) | PESQ **2.58** / STOI **82.84** | **NB** | **文獻估值** → Yang, Pandey & Wang, *Time-Domain Speech Enhancement for Robust ASR*, **Interspeech 2023**（arXiv:2210.13318），**Table 1「mixture」列** |
| P5 Tsinghua | TMSV | 未報 | — | — |
| P1 NTU | NOISEX+IEEE | 未報（只報 WAR +41%@−6dB）| — | 原文 Fig 19 |

> CHiME-2 noisy 完整 per-SNR（NB, ARN paper Table 1，mixture）：PESQ −6/−3/0/+3/+6/+9 = 2.13/2.33/2.49/2.66/2.85/3.04（avg 2.58）；STOI% = 73.68/77.80/81.34/85.16/88.12/90.93（avg 82.84）。
> ⚠️ CHiME-2 noisy 是**文獻估值**，隨 reference 類型變動：**STOI（82.84）可用**（P3/P1 的 STOI Δ 正常）；**PESQ（2.58）不可用**（套 JSSC'20 得負 Δ，見下）。P2 既排除，VoiceBank 不再是目標 → **乾淨同資料集目標 = CHiME2，報絕對 NB-PESQ**。

### Δ 對照（次要；P2 已排除、CHiME2 PESQ Δ 不可靠）
| 對手 | 對齊 / 模式 | 你 PESQ Δ (nb) | 對手 PESQ Δ | 你 MBSTOI Δ | 對手 STOI Δ |
|---|---|---|---|---|---|
| P3 SNU | −6…+9 / NB | +0.391 | **不可靠**（見下）| +0.064 | **+8.6**（91.44−82.84）|
| P4 MIT | −6…+9 / NB | +0.391 | **不可靠** | +0.064 | —（MIT 未報 STOI）|

**為何 CHiME2 PESQ Δ 不可靠**：用 ARN noisy(NB 2.58) 套 P3 Table II → P3 ΔPESQ +0.15、**但 JSSC'20 ΔPESQ = 2.38−2.58 = −0.20（負！）**。JSSC'20 SDR +10.7 dB 明明大幅去噪，PESQ 不可能負 → 證明 **P3 的 CHiME2 noisy < 2.58（約 ~2.0，reference/子集不同）**。結論：**CHiME2 的 PESQ Δ 文獻取不到、不要寫**；**STOI Δ 可用**（ARN 82.84，兩邊正常：P3 +8.6、JSSC'20 +6.2）。→ 改用上方「絕對 PESQ」呈現。

### 每個 SNR 點的 Δ（你的低 SNR 強項）
| SNR | ΔMBSTOI | Δ nb PESQ |
|---|---|---|
| **−6** | **+0.128** | +0.173 |
| −3 | +0.093 | +0.275 |
| 0 | +0.070 | +0.365 |
| +6 | +0.030 | +0.516 |
| +9 | +0.017 | +0.566 |
| +15 | +0.003 | +0.605 |

→ MBSTOI 改善集中在**低 SNR**（−6 dB +0.128，越高越接近天花板）；nb PESQ 改善則隨 SNR 升高。

### 解讀（P2 排除後）
- **主要呈現＝絕對 nb PESQ**（上方表）：你 ~2.0–2.22，與 CHiME2 群（2.38–2.79）同級距、贏 P5。
- **可懂度**：你 ΔMBSTOI +0.064（−6…+9）；對照 P3 ΔSTOI +8.6（不同指標、量級可比）。ILD/IPD 對手沒有。
- **真正站得住＝低 SNR**：−6 dB ΔMBSTOI **+0.128**，**純音訊（雙耳麥克風）**達成 → 對照 P5 需**攝影機**才撐 −10 dB。
- **想要完全可比的硬數字 → 跑 CHiME2（NB）**（P2 既已排除，VoiceBank 不再需要）。

---

## 12. 資料集嚴格度對照（test condition）

| | 取樣率 | SNR 範圍 | 殘響/回音 | 空間/聲道 | 噪音來源 |
|---|---|---|---|---|---|
| P1 NTU | 16 kHz | −6…+6 dB | ✗ 純加性 | mono | NOISEX-92（babble/white/factory…5 類）|
| P2 PKU | 16 kHz | **+2.5…+17.5 dB（底最高=最寬鬆）** | ✗ anechoic | mono | DEMAND |
| P3 SNU | 16 kHz | −6…+9 dB | **✓ CHiME-2 居家殘響** | mono | 真實居家（吸塵器/電視/童聲）|
| P4 MIT | **8 kHz（窄頻=較易）** | −6…+9 dB | ✓ CHiME-2 殘響 | mono | 同 CHiME-2 |
| P5 Tsinghua | 16 kHz | **−10 dB（底最低）** | 不明（疑純加性）| mono **+ 視覺輔助** | TMSV 加噪 |
| **你 BCCTN** | 16 kHz | **−6…+15 dB** | **✗ 無殘響（已確認，純加性）** | **binaural 空間化（azimuth）** | 待確認 |

**判讀（更正版：你的低絕對分數來自 wb/nb 模式 + 雙耳 + 低 SNR，不是殘響）：**
1. **唯一雙耳空間化**（檔名 `_az<方位>_snr<snr>`，clean/noisy 同方位配對，[eval.py:923-947]）：別人皆 mono；你除去噪外還要保 ILD/IPD，且 PESQ 用雙耳訊號本就吃虧。
2. **無殘響**（已確認）：與 P1/P2 同（純加性）；P3/P4 才有 CHiME-2 殘響。所以你**不**靠殘響當「難」的理由。
3. **SNR 跨度最廣含重噪**：−6…+15；P3/P4 −6…+9；**P2 最低僅 +2.5 dB（最寬鬆）**。
4. ✅ **更正**：之前 noisy PESQ「+15dB 仍 1.58」的疑慮，已查明純粹是 **wideband vs narrowband** 模式差（wb 1.58 / nb 2.40），**與殘響無關**（資料集確認無殘響）。對 P2 報 wb、對 P3/P4 報 nb 即可。
5. **你的 noisy 輸入品質本就較低**：nb noisy @−6dB = 1.36 vs CHiME-2 nb noisy @−6dB = 2.13 → 你的測試輸入更差（雙耳 + 噪音 + 低 SNR），但這是跨資料集比較，僅供脈絡。
6. **P2 的 3.34 在最寬鬆條件（+2.5dB 起、anechoic、mono）取得；P5 的 −10dB 有視覺輔助抵消。**

**論文防線**：comparison table 加「Test condition（SNR floor / reverb / spatial / PESQ mode）」列，標出 P2 +2.5dB-anechoic-WB、P3/P4 −6dB-reverb-NB、P5 視覺輔助、P4 8kHz。

> 註：資料集生成腳本不在本 repo（資料預渲染於 `/raid/R12K41024/BCCTN/Dataset/`）。**已確認**：binaural 空間化(azimuth)、SNR −6…+15、clean/noisy 同方位配對、**無殘響（純加性）**、PESQ 低是 wb/nb 模式差。**待確認**：噪音來源 corpus、HRTF/BRIR 空間化來源。

---

## 13. Latency vs 操作頻率（latency 必須配頻率看）

各篇 latency 在**不同操作頻率**下量的，不能直接比 ms。用 `cycles = latency × freq` 正規化才公平。

| 晶片 | latency（原文）| @ 操作頻率 | ≈ cycles | 正規化@20MHz | 備註 |
|---|---|---|---|---|---|
| P1 NTU | 1.88 ms (compute) | 5 MHz | ~9.4k | ~0.47 ms | 小 CNN；跨表另列 9.88ms＝含 frame buffer 的 total |
| P2 PKU | **~8 ms（推估）** | 10 MHz | ~80k | ~4.0 ms | 由 15.2µJ÷1.9mW 推得（與 125fps 一致）；可隨 freq↑至 450MHz 降低 |
| P3 SNU | **39.96 ms (total)** | 2.5 MHz | ~100k | ~5.0 ms | **含演算法 look-ahead**；跨表的 7.96ms 是 processing-only |
| P4 MIT | 7.96 ms | 18.5 MHz | ~147k | ~7.36 ms | 8 kHz、8 ms 約束 |
| P5 Tsinghua | 1.11 ms (processing) | **50 MHz** | ~55.5k | ~2.78 ms | processing only |
| **你 BCCTN** | **2.5 ms@20MHz／5 ms@10MHz** | 20 / 10 MHz | **50k** | 2.5 ms | **causal、無 look-ahead** |

**判讀：**
1. **raw ms 受頻率左右**：P5 的 **1.11 ms 是 @50 MHz**（高 clock、耗電換來）；正規化到 20 MHz ≈ 2.78 ms，**與你 2.5 ms 同級**。別被 1.11 ms 嚇到。
2. **正規化@20MHz 後排序**：P1 0.47ms（小模型）< 你 2.5ms ≈ P5 2.78ms < P2 4.0ms < P3 5.0ms(total) < P4 7.36ms。**你跑「雙耳複數 Transformer」只用 50k cycles，比 P2(80k)/P4(147k)/P5(55k) 都少。**
3. **你 causal（無未來 frame）→ total latency 低**；P3 的 **39.96 ms** total 含 look-ahead（吃未來 frame），是真實使用者延遲最差的。這是你可主打的「低演算法延遲」點。
4. ⚠️ latency 定義不一（compute-only / processing / total-incl-lookahead），論文比較表要標清楚你報的是哪種（你目前是 processing、causal）。

---

## 附錄：資料來源

- **P1** — Y.-C. Lee, T.-S. Chi, C.-H. Yang, "A 2.17-mW Acoustic DSP Processor With CNN-FFT Accelerators…," IEEE JSSC 55(8), Aug. 2020.（Table II/III, Fig.18-19）
- **P2** — Q. Zhi et al. (Peking Univ.), "A 22nm 15.2µJ/Inference Speech Enhancement Accelerator with Contextual-Aware KV-Cache Management and Reconfigurable D-CIM Engine," IEEE CICC 2026.（Fig.7 spec, Fig.8-10, Table I）VoiceBank+DEMAND, PESQ 3.34/STOI 95.11, MAC/frame 39.6M。
- **P3** — S. Park, …, D. Jeon, "A Real-Time Speech Enhancement Processor for Hearing Aids in 28-nm CMOS," IEEE JSSC 2024（會議版 ISSCC'23）.（Table II, Fig.27/30-32）CHiME2, SDR14/PESQ2.73/STOI91.4。
- **P4** — D. V. Kochar, M. Ashok, A. P. Chandrakasan (MIT), "A 0.75mm² 407µW Real-Time Speech Audio Denoiser with Quantized Cascaded Redundant Convolutional Encoder-Decoder…," VLSID 2025.（Table I, Fig.2/5/8-9）8 kHz, CHiME2, PESQ 2.79；**407µW/3.24µJ 為 ML-only，不含 STFT**。
- **P5** — N. Wu, S. Wang, …, H. Jiang (Tsinghua/Fudan/NTU), "A 3.64µJ/frame Visual-Aided Speech Enhancement Processor Tolerating −10dB SNR Input for Hearing Aids in 22nm CMOS," IEEE ASSCC 2025, Paper 20.1.（Fig.6 comparison, Fig.7 summary）**Audio-Visual（需相機）**, 5.76mm², 228µW core/0.58mW system, −10dB, PESQ 2.13, TMSV。
**外部 noisy baseline 引用（算 Δ 用）：**
- **VoiceBank+DEMAND noisy**（PESQ 1.97 WB / STOI 92.1）— 資料集標準值（C. Valentini-Botinhao et al., 2016/2017）；亦見 CMGAN（WB-PESQ 3.41）、MP-SENet（WB-PESQ 3.50, arXiv:2308.08926）。
- **CHiME-2 noisy / mixture**（NB-PESQ avg 2.58、STOI 82.84%，per-SNR 見 §11）— **Y. Yang, A. Pandey, D. Wang, "Time-Domain Speech Enhancement for Robust Automatic Speech Recognition," Interspeech 2023, arXiv:2210.13318, Table 1（mixture 列）**。⚠️ 文獻估值、NB、reference 類型相依，套 P3/P4 僅供脈絡。

- **本設計 BCCTN** — repo `/home/r12k41024/Research`：品質 `10_algo/results/apot_descale_eval/ft_pow2_ep3_full/{vctk,timit}_paper_style_results*.{txt,csv}`；模型/MAC `RESULTS_ARCH_PROGRESSION.md`, `10_algo/docs/STATUS_LOW_POWER_2026-05-19.md`, `10_algo/code/DCNN/models/{model,binaural_attention_model}.py`, `10_algo/code/config/config.yaml`；MAC陣列/數值 `01_RTL/Define/Param_def.sv`, `01_RTL/engine/CE/PEG.sv`；記憶體 `10_algo/docs/SRAM_MEMORY_MAP_SPEC_2026-06-03.md`；面積/功耗/延遲 gate-sim PrimeTime（使用者提供 ~1.2 mm²/4.76 mW@20MHz,0.8V/~2.5 ms，含 on-chip STFT；FFT128 模組報告 `02_SYN/Report/FFT128.{area,power}`）。
