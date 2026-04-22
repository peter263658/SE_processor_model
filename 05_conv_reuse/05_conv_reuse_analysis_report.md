# 05_conv_reuse — Conv Weight Reuse 優化分析報告

**分析日期**：2026-04-16  
**模型**：BCCTN（Binaural Speech Enhancement, ICASSP 2024）  
**Checkpoint**：`05_conv_reuse/model.ckpt`  
**分析腳本**：`05_conv_reuse/reuse_analyzer.py`

---

## 任務說明

本資料夾的任務為分析 BCCTN 模型卷積層的 **weight reuse 機會**，目標是降低 DMA 搬運量與 on-chip SRAM 需求。

分析面向共四項：

| # | 分析面向 | 核心問題 |
|---|---|---|
| 1 | **Cross-layer weight similarity** | 哪些 layer pair 的 weights 相似度高，可考慮共用？ |
| 2 | **Low-rank factorization potential** | PW conv 和 MHA 投影矩陣是否低秩？可節省多少 DMA？ |
| 3 | **Temporal activation reuse** | 相鄰 frame 之間有多少 activation 可重用？需要多大 cache？ |
| 4 | **Depthwise vs Pointwise DMA 佔比** | DW 和 PW conv 各自佔 weight DMA 的多少？主要瓶頸在哪？ |

### 模型架構背景

```
Input → STFT → Encoder(6 stages) → MultiHeadAttn(slide win=31) → Decoder(6 stages) → iSTFT → Output
```

- 每個 encoder/decoder stage：`ComplexDepthwiseSeparableConv2d`（DW + PW）
- Conv kernel: `(5,1)`，stride: `(2,1)`（freq 方向 stride 2，time 方向 stride 1）
- Channel 數：`[2, 16, 32, 64, 128, 256, 256]`（含 input）
- MHA：`embed_dim=256`，`num_heads=32`，sliding window=31
- L/R channel 已共用 encoder/decoder weights（binaural shared）

---

## 分析結論

### 1. Cross-Layer Weight Similarity（權重相似度）

分析方法：
- 計算 cosine similarity（flatten 後，shorter tensor 補零）
- 計算 normalized L2 distance
- 比較 SVD 奇異值譜（正規化後的 cosine sim）
- Encoder-Decoder mirror pair：比較 Enc_PW 的 transpose 與 Dec_PW 的 non-skip 半段

**全部結果如下：**

| Layer A | Layer B | Shape A | Shape B | Cosine Sim | L2 Norm Dist | SVD Sim | 備註 |
|---|---|---|---|---|---|---|---|
| Enc0_PW | Enc1_PW | (8,1,1,1) | (16,8,1,1) | 0.249 | 0.621 | 1.000 | — |
| Enc1_PW | Enc2_PW | (16,8,1,1) | (32,16,1,1) | 0.436 | 0.566 | 0.986 | — |
| Enc2_PW | Enc3_PW | (32,16,1,1) | (64,32,1,1) | 0.400 | 0.580 | 0.981 | — |
| Enc3_PW | Enc4_PW | (64,32,1,1) | (128,64,1,1) | 0.393 | 0.590 | 0.985 | — |
| Enc4_PW | Enc5_PW | (128,64,1,1) | (128,128,1,1) | 0.556 | 0.483 | 0.997 | — |
| Enc0_DW | Enc1_DW | (1,1,5,1) | (8,1,5,1) | 0.353 | 0.653 | 1.000 | — |
| Enc1_DW | Enc2_DW | (8,1,5,1) | (16,1,5,1) | 0.551 | 0.493 | 0.969 | — |
| Enc2_DW | Enc3_DW | (16,1,5,1) | (32,1,5,1) | 0.624 | 0.459 | 0.985 | — |
| Enc3_DW | Enc4_DW | (32,1,5,1) | (64,1,5,1) | 0.593 | 0.471 | 0.986 | — |
| Enc4_DW | Enc5_DW | (64,1,5,1) | (128,1,5,1) | 0.623 | 0.458 | 0.999 | — |
| Dec0_PW | Dec1_PW | (128,256,1,1) | (64,256,1,1) | 0.465 | 0.600 | 0.970 | — |
| Dec1_PW | Dec2_PW | (64,256,1,1) | (32,128,1,1) | 0.357 | 0.620 | 0.989 | — |
| Dec2_PW | Dec3_PW | (32,128,1,1) | (16,64,1,1) | 0.326 | 0.612 | 0.990 | — |
| Dec3_PW | Dec4_PW | (16,64,1,1) | (8,32,1,1) | 0.281 | 0.609 | 0.984 | — |
| Dec4_PW | Dec5_PW | (8,32,1,1) | (1,16,1,1) | 0.298 | 0.792 | 1.000 | — |
| **Enc0_PW^T** | **Dec5_PW_ns** | (8,1) | (1,8) | **0.955** | 0.495 | 1.000 | **唯一候選** |
| Enc1_PW^T | Dec4_PW_ns | (8,16) | (8,8) | 0.614 | 0.441 | 0.967 | — |
| Enc2_PW^T | Dec3_PW_ns | (16,32) | (16,16) | 0.512 | 0.518 | 0.968 | — |
| Enc3_PW^T | Dec2_PW_ns | (32,64) | (32,32) | 0.544 | 0.500 | 0.976 | — |
| Enc4_PW^T | Dec1_PW_ns | (64,128) | (64,64) | 0.547 | 0.498 | 0.982 | — |
| Enc5_PW^T | Dec0_PW_ns | (128,128) | (128,128) | 0.775 | 0.336 | 0.978 | — |

**結論：**

- **相鄰 encoder layers 不可 weight sharing**：所有 PW adjacent pairs 的 cosine sim 均低於 0.60，差異顯著，各層功能角色不同
- **Depthwise filter 的 SVD spectra sim 很高**（0.97–1.00），但 DW filter 本身只有 5 個 weights/channel，共用沒有實質 DMA 效益
- **唯一 U-Net weight tying 候選**：Enc0 PW^T vs Dec5 PW（cos sim = 0.955），但此 pair 的矩陣極小（8×1 vs 1×8），節省量接近零，不具硬體實作價值
- Enc5↔Dec0 mirror（最大的 128×128 pair）cos sim = 0.775，未達 0.9 門檻，**不建議 weight tying**

---

### 2. Low-rank Factorization Potential（低秩分解潛力）

分析方法：對每層 PW conv weight matrix `(Cout, Cin)` 做 SVD，計算捕捉 95%/99% Frobenius norm 所需的 rank r，以及理論 memory saving：`saving = 1 − r(Cin+Cout)/(Cin×Cout)`

**PW Conv 層（real part）：**

| Layer | Shape | rank@95% | rank@99% | eff_rank@95% | eff_rank@99% | saving@95% | saving@99% |
|---|---|---|---|---|---|---|---|
| Enc0_PW | (8,1) | 1/1 | 1/1 | 1.00 | 1.00 | −12.5% | −12.5% |
| Enc1_PW | (16,8) | 7/8 | 8/8 | 0.88 | 1.00 | −31.2% | −50.0% |
| Enc2_PW | (32,16) | 12/16 | 15/16 | 0.75 | 0.94 | −12.5% | −40.6% |
| Enc3_PW | (64,32) | 24/32 | 30/32 | 0.75 | 0.94 | −12.5% | −40.6% |
| Enc4_PW | (128,64) | 48/64 | 59/64 | 0.75 | 0.92 | −12.5% | −38.3% |
| Enc5_PW | (128,128) | 69/128 | 94/128 | 0.54 | 0.73 | −7.8% | −46.9% |
| Dec0_PW | (128,256) | 64/128 | 101/128 | 0.50 | **0.79** | **+25.0%** | −18.4% |
| Dec1_PW | (64,256) | 52/64 | 61/64 | 0.81 | 0.95 | −1.6% | −19.1% |
| Dec2_PW | (32,128) | 26/32 | 31/32 | 0.81 | 0.97 | −1.6% | −21.1% |
| Dec3_PW | (16,64) | 13/16 | 16/16 | 0.81 | 1.00 | −1.6% | −25.0% |
| Dec4_PW | (8,32) | 6/8 | 8/8 | 0.75 | 1.00 | +6.2% | −25.0% |
| Dec5_PW | (1,16) | 1/1 | 1/1 | 1.00 | 1.00 | −6.2% | −6.2% |

> saving 為負數代表 factorization 反而增加參數量（小矩陣做分解會有 overhead）

**MHA 投影矩陣（256×256）：**

| Layer | rank@95% | rank@99% | eff_rank@95% | saving@95% |
|---|---|---|---|---|
| MHA_re Q | 133/256 | 183/256 | 0.52 | −3.9% |
| MHA_re K | 135/256 | 184/256 | 0.53 | −5.5% |
| MHA_re V | 121/256 | 176/256 | 0.47 | **+5.5%** |
| MHA_re Out | 104/256 | 166/256 | 0.41 | **+18.8%** |
| MHA_im Q | 135/256 | 185/256 | 0.53 | −5.5% |
| MHA_im K | 137/256 | 185/256 | 0.54 | −7.0% |
| MHA_im V | 115/256 | 172/256 | 0.45 | **+10.2%** |
| MHA_im Out | 97/256 | 161/256 | 0.38 | **+24.2%** |
| input_proj (r) | 134/256 | 184/256 | 0.52 | −4.7% |
| input_proj (i) | 132/256 | 183/256 | 0.52 | −3.1% |
| output_proj (r) | 93/256 | 158/256 | 0.36 | **+27.3%** |
| output_proj (i) | 94/256 | 159/256 | 0.37 | **+26.6%** |

**結論：**

- **Encoder PW conv：全部不適合低秩分解**。矩陣偏小（最大 128×128），rank@99% 幾乎是 full rank，factorization 會增加參數量而非減少
- **Dec0 PW (128×256) 是唯一例外**：rank@95% = 64/128，可節省 25% 參數量 → 值得考慮
- **MHA Out projection（re & im）最具潛力**：rank@95% ≈ 97–104/256（eff_rank 0.38–0.41），saving@95% 分別為 +18.8% 和 +24.2%。`output_proj` 亦達 27% saving
- MHA Q/K/input_proj 的 rank 較高（0.52–0.54），factorization 效益不明顯
- 奇異值衰減圖存放於 `results/singular_values/*.png`

---

### 3. Temporal Activation Reuse（時序 Activation 重用）

#### 分析框架

模型以 streaming 模式逐 frame 處理，每個 frame = 1 個 STFT time bin：

```
win_len=128, win_inc=80
FFT_len=128 → 65 freq bins
Encoder conv: kernel=(5,1), stride=(2,1), padding=(2,0)
```

#### 頻率維度空間重用（Freq Spatial Reuse）

當 kernel 在 freq 方向滑動時，相鄰兩個輸出位置的輸入重疊：

```
kernel=5, stride=2 → 重疊 = (5-2)/5 = 60%
```

每個 encoder stage 的 input 規模（streaming 每次 1 time step）：

| Layer | Freq in | Freq out | Ch in (half) | 每 frame input (Bytes, complex64) |
|---|---|---|---|---|
| Enc0 | 65 | 33 | 1 | 520 B |
| Enc1 | 33 | 17 | 8 | 2,112 B |
| Enc2 | 17 | 9 | 16 | 2,176 B |
| Enc3 | 9 | 5 | 32 | 2,304 B |
| Enc4 | 5 | 3 | 64 | 2,560 B |
| Enc5 | 3 | 2 | 128 | 3,072 B |

頻率 spatial reuse ratio = **60%（所有層一致）**

#### 時間方向 Cross-Frame Reuse

由於 conv kernel 在 time 方向的尺寸為 `kernel_t = 1`、stride = 1：

> **Conv 層完全不需要 cross-frame activation cache。** 每個 time step 的計算與前一 frame 的 activation 無關。

STFT 原始樣本 overlap 為 `(128−80)/128 = 37.5%`，但這是時域的重疊，在頻域 activation 上並無直接可 cache 的結構。

#### 注意力層的時序 Cache

Attention 使用 sliding window = 31 frames，因此必須快取最近 31 個 encoder 輸出：

```
Cache size = 31 frames × 256 complex elements/frame × 8 Bytes = 62 KB (complex64)
           = 31 frames × 256 elements/frame × 1 Byte  =  7.7 KB (FP8)
```

---

### 4. Depthwise vs Pointwise Weight DMA 佔比

| Layer | DW bytes (FP8) | PW bytes (FP8) | PW% |
|---|---|---|---|
| Enc0 | 10 | 16 | 61.5% |
| Enc1 | 80 | 256 | 76.2% |
| Enc2 | 160 | 1,024 | 86.5% |
| Enc3 | 320 | 4,096 | 92.8% |
| Enc4 | 640 | 16,384 | 96.2% |
| Enc5 | 1,280 | 32,768 | 96.2% |
| Dec0 | 1,280 | 65,536 | 98.1% |
| Dec1 | 640 | 32,768 | 98.1% |
| Dec2 | 320 | 8,192 | 96.2% |
| Dec3 | 160 | 2,048 | 92.8% |
| Dec4 | 80 | 512 | 86.5% |
| Dec5 | 10 | 32 | 76.2% |
| **合計** | **4,980 B (4.9 KB)** | **163,632 B (159.8 KB)** | **97.0%** |

**MHA（re+im, FP8）：** 655,360 B（**640 KB**）

```
DMA 組成：
  DW conv  :   4.9 KB  (1.9%)
  PW conv  : 159.8 KB  (61.3%)
  MHA      : 640.0 KB  (最大，比所有 PW conv 合計大 4×)
```

**DW filter（5 weights/channel）極小，硬體可常駐於 register file，DMA overhead 幾乎可忽略。**

---

## 總結與給顧問的討論重點

### 六個核心問題的答案

| # | 問題 | 答案 |
|---|---|---|
| Q1 | 相鄰 encoder layer 有無 weight sharing 機會（cos sim > 0.9）？ | **否。** 所有相鄰 PW pair 的 cos sim ≤ 0.556，差異過大 |
| Q2 | Encoder-Decoder mirror pair 可否 weight tying（U-Net 式）？ | **僅 Enc0↔Dec5 達標（cos sim = 0.955），但矩陣尺寸極小（8×1），節省量趨近於零，不具實作價值** |
| Q3 | PW conv 與 MHA projection 有效秩為何？低秩分解值得嗎？ | **PW conv 不值得**（矩陣太小，rank 接近 full）。**MHA Out & output_proj 值得考慮**（rank@95% ≈ 0.38–0.41，saving +18–27%） |
| Q4 | 相鄰 frame 之間有多少 activation 可重用？ | **Conv 層：0%（kernel_t=1）。Attention：需 cache 31 frames 的 encoder output** |
| Q5 | 需要多大 on-chip cache 才能利用 temporal reuse？ | DW freq-row cache：全部 encoder 合計 **5.8 KB**；Attention context：**62 KB（complex64）/ 7.7 KB（FP8）** |
| Q6 | PW vs DW 的 weight DMA 佔比？ | **PW 佔 97%**，DW 可完全忽略。MHA 是真正的 DMA 瓶頸（比所有 PW 合計大 4×） |

### 優先行動建議（按 ROI 排序）

| 優先順序 | 建議 | 預期效益 | 需要 re-training？ |
|---|---|---|---|
| **P1（高）** | **W4 量化 MHA**（Wq/Wk/Wv/Wo）+ per-group scale | DMA 640 KB → 160 KB（4× 降低）| 是（QAT） |
| **P2（高）** | **DW freq-row sliding buffer**：以 5 row ring buffer 處理 DW conv，避免重複讀取 | 60% 降低 DW freq 讀取 DMA，雖 DW 佔比小，但 latency 有幫助 | 否（硬體優化） |
| **P3（中）** | **MHA Out & output_proj 低秩分解**（rank ≈ 100/256）| saving 約 +18–27%，等效 MHA DMA 再降 10–20% | 是（fine-tune） |
| **P4（中）** | **Dec0 PW 低秩分解**（128×256，rank@95% = 64）| saving +25%，但此層 bytes 佔比不高 | 是（fine-tune） |
| **P5（低）** | **Attention cache 使用 FP8 壓縮儲存** | 62 KB → 7.7 KB（8× 降低 on-chip SRAM） | 否 |
| **不建議** | Weight sharing 於相鄰 encoder layer | cos sim 太低，需 re-training 後效能風險高，收益小 | — |
| **已完成** | L/R channel 共用 encoder/decoder weights（binaural shared） | 已實作，節省 2× | — |

### 討論建議

1. **MHA 才是主角**：DMA 640 KB 遠超所有 PW conv（160 KB），weight reuse 優化應聚焦 MHA，而非 conv
2. **Conv layer 的 reuse 機會有限**：架構本身（channel 倍增、跳接）決定了各層扮演不同角色，強行 weight sharing 會傷害模型能力
3. **Low-rank factorization 需實測**：MHA Out proj 的 rank 分析顯示潛力，但複數域的 factorization 在硬體實作上有額外複雜度，需與 T2（quantization）方向一起評估
4. **Freq-row buffer 是零成本優化**：DW conv 的 60% freq spatial reuse 可直接在硬體 datapath 實作，不需 re-training，適合硬體端先行

---

*分析腳本：`05_conv_reuse/reuse_analyzer.py`*  
*原始數據：`05_conv_reuse/results/`*  
*執行指令：`conda activate bcctn && python reuse_analyzer.py --checkpoint model.ckpt`*
