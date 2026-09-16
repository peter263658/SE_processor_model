# 架構演進 — 精簡結果

每個階段整理硬體架構變化，以及它換到的結果（參數 / MAC / 品質）。
品質數字是該階段完整 paper-style VCTK 與 TIMIT MBSTOI。Gate：
VCTK ≥ 0.835（hard floor 0.8302）、TIMIT ≥ 0.7880。

這裡有兩種參數觀點：**RTL** = 晶片實際儲存/計算的權重；
PyTorch `state_dict` 在 masked/grouped layer 上可能和 RTL 實際權重數不同，
表格中會標出。MAC = 每 5 ms frame。

下面的演進從最原始 full-size BCCTN reference 開始
（`fft_len=512`、`embed_dim=512`），再分成兩類變化：

- **架構縮小 / 結構壓縮**：減少 tensor size、儲存權重與 MAC workload。
- **數值格式 / 硬體運算轉換**：拓撲大致固定，改變 precision、scale 與 multiplier 實作。

---

## Original BCCTN → hardware-sized working point（pre-09）

這幾列說明為什麼後面的 09→18 工作不是直接從 paper-like BCCTN 開始，
而是從已經縮小過的 baseline 開始。Original model 在這裡作為品質與尺寸參考。

| Stage / source | Architecture / analysis result | Params / workload implication | Quality / decision |
|---|---|---|---|
| **Original BCCTN reference** (`/raid/R12K41024/BCCTN/config/model.yaml`) | `win_len=400`, `win_inc=100`, `fft_len=512`; 6 個 encoder/decoder stage `kernel_num=[16,32,64,128,256,256]`; `embed_dim=512`, `num_heads=32`; real/imag MHA 不 tied；decoder dense | **3,973,288 params** = encoder 274,988 + transformer 3,151,360 + decoder 546,940。Transformer alone 約佔 79% params。Archived flow shape：STFT `[1,256,321]`、transformer `[1,256,4,321]`。Profiled 計算量（對 `model.yaml` arch 掛 hook、random init）≈ **9.4 G MACs / 2 s** = **≈ 29 M MAC / 100-sample frame**（~4.7 G/s）：convT 4.86 G + conv 2.43 G + MHA 1.77 G + linear 0.34 G。Full conv（`(5,2)` kernel）+ dense decoder + untied 32-head full `T×T` attention；複數 = 實際執行的 4 個 real sub-conv。**自己的 basis**（hop 100, fft 512）；未 normalize 到 deploy 的 1.10 M multiply-equivalent/frame。 | Full-size quality reference：**VCTK 0.8519 / TIMIT 0.8114** (`BCCTN/results_May21/...`)。對目標 WBUF/MAC budget 太大。 |
| **Hardware-sized FP8/QAT working point** (`main_model` / `01_utilization_analysis`) | 改成 `fft_len=128`, `win_len=128`, `win_inc=80`; 保留 6 stages；transformer 降成 `embed_dim=256`, `num_heads=16`, sliding `attn_win=31`; conv path 用 depthwise-separable | 09/tied-MHA/decoder grouping 之前的參數基準變成 **830,300**。Utilization analysis：attention 是 **77.6%** MAC；decoder 15.0%；encoder 7.4%。1-second profiler 用自己的 convention 算到 **1.687G MACs**；這個數字只用來判斷瓶頸歸因，不當作下面 normalized per-frame MAC。 | Hardware-sized FP8/QAT baseline 品質：**VCTK 0.8399 / TIMIT 0.7966** (`main_model/results_May21/...`)。這是 09+ 的實際起點。 |
| **01 Utilization analysis** | Weight-stationary 256-MAC 分析顯示整體是 compute-bound，但 layer bottleneck 不同：早期/depthwise conv 是 activation-DMA/under-utilized，attention projection 是 compute-heavy，QK/Score×V 是 activation-transport heavy | W4A8 幾乎不改 average utilization（45.3%→45.4%），所以 W4A8 **不是**為了提高 MAC utilization。真正收益是 weight SRAM/DMA 與 WBUF residency。 | 方向：先打 attention weight size / transport，再移除 softmax 與 multiplier cost。 |
| **02 Weight bit-width sensitivity** | 在 FP8-QAT model 上做 weight-only PTQ sweep：activation 保持 FP8，weight 測 INT8/INT4/INT3/FP4 | Uniform INT4 給 **50% weight/DMA reduction**，local metric loss 可接受（SI-SNR −0.56 dB、ILD +0.21 dB、IPD +0.64°）。INT3 mixed precision 雖然更省 bytes，但全網路一起套時 cumulative error 失敗（SI-SNR −3.08 dB）。 | 決策：主路徑用 **W4A8**；不要走 W3。用 per-group/per-head scaling，不用 global scale。 |
| **07 Mixed-precision W4A8 QAT design** | 大多數 conv/linear/MHA weight 用 INT4，activation 用 FP8，accumulator 用 fp16。Decoder pointwise outlier 保留 FP8；MHA scale per-head；QK/Score×V 需要 fp16 accumulator | 早期發現：decoder PW stage 0/3/4 有 outlier；MHA head group 不均勻；QK dynamic range 超出 FP8。 | 這成為後續 W4A8 + APoT scales + L-Mul + pow2 block-float activations 的數值模板。 |

Net original → hardware-sized baseline：參數 **3.97M → 0.83M**（約 **−79%**），
這還是在 09→14 結構壓縮之前。品質下降約 0.012 VCTK MBSTOI、0.015 TIMIT MBSTOI，
但 model 變小到可以開始符合目標硬體限制。

計算量沿同一條線下降：原始 profile 出 **≈ 29 M MAC/frame**（hop-100 basis），
final deploy 是 **1.10 M multiply-equivalent/frame**——約 **20–27× fewer MAC**
（依 basis 而定：hop、FFT、複數乘法慣例不同），來自 full→depthwise-separable conv、
dense→grouped decoder + top-50% skip、untied full attention → tied + windowed +
grouped-QKV + top-4，以及較低的 FFT/freq 解析度。原始數字是 profile（forward hooks、
random init）出來、報在自己的 basis；**未** normalize 到 deploy frame。

---

## Structural compression（folders 09 → 14，從 hardware-sized baseline 開始）

| Stage | HW architecture change | Params | MAC/frame | VCTK | TIMIT |
|---|---|---|---|---|---|
| Hardware-sized baseline（pre-09 untied, dense decoder） | `fft_len=128`, `embed_dim=256`, `num_heads=16`; FP8/QAT working point | **830,300** | ~1.67 M | 0.8399 | 0.7966 |
| **09 Tied complex MHA** | `MHA_re == MHA_im`：一份 weight set，datapath 跑 2× 而不是 4× | 830,300 → **567,132**（−31.7%；MHA-local −50%） | MHA passes 4→2 | 0.8383 | — |
| **09→11 Shift-topK pow2 softmax** | top-4 + pow2 weights、無 mantissa；**移除 exp/div LUT**；4-entry sparse prob row | — | softmax mult/div → shift | 0.8360 | — |
| **12 Decoder grouped pointwise** (`groups_x=4`, `groups_skip=2`) | real `Conv2d(groups)`、block-diagonal；無 dynamic indexing | 567,132 → **485,332**（−14%） | decoder PW −62%；whole 1.67→**1.44 M**（−14%） | 0.8361 | 0.7923 |
| **12 Decoder skip subset**（top-50% by L1） | 只保留 top-50% skip channels（fixed index） | （包含在上列） | skip bandwidth 減半 | 0.8352 | 0.7919 |
| **13 FP16 attention accumulator** | attention scratch 降成 fp16 | — | — | 0.8352 | 0.7914 |
| **13 L-Mul M4 on QK + Score×V** | FP multiplier backend → L-Mul add/shift | — | attention multiplier energy 下降（count 不變） | 0.8364 | 0.7925 |
| **14 Grouped QKV projection** (`qkv_groups=4`, 30-ep QAT) | block-diagonal Q/K/V；只存在 diagonal 64×64 blocks | RTL Q/K/V 196,608 → **49,152**（−75%）；total RTL **337,876** | QKV −75%、attention −28%、whole 1.39→**1.10 M**（−21%）；**duty 1.02→0.81** | 0.8353 | 0.7881 |
| **14 Sparse top-4 q0.8 packet + bit-exact L-Mul** | packetized sparse prob；52 bits/row、5-bit local idx | — | Score×V 改成 packet gather | 0.8360 | — |
| **= 14 composite（production）** | grouped-QKV-4 + packet + L-Mul | PyTorch 485,332 / **RTL 337,876**（相對 09 −59.3%） | **1.10 M**（相對 dense QKV −21%） | **0.8354** | **0.7880** |

Net 09→14：RTL weights **−59.3%**，MAC/frame **1.67 M → 1.10 M**，
VCTK/TIMIT 兩個 gate 都過。QKV W4 weight set 96 KB → 24 KB，所以能在 32 KB WBUF 中一 pass 放下。

---

## MAC reduction steps（只列 workload 有變的步驟）

這張表只列 normalized neural-core workload 有變的階段。後續 15→18 數值階段維持同樣的
1.10 M multiply-equivalent workload，只改每個 operation 的硬體實作方式。

| Step | MAC / workload change | Why it changes |
|---|---|---|
| Hardware-sized FP8/QAT baseline | 建立 normalized working point：**~1.67 M MAC/frame** | `fft_len=128`, `embed_dim=256`, `num_heads=16`, sliding `attn_win=31`；original BCCTN 512/512 model 沒有用 final frame basis normalize |
| 09 Tied complex MHA | MHA real passes **4 → 2** | real/imag MHA weight tied；tied block 跑兩個 shared real-MHA pass，最後合併 `(a−b)+i(a+b)` |
| 12 Decoder grouped pointwise | whole model **1.67 M → 1.44 M/frame**（−14%）；decoder pointwise local −62% | decoder pointwise layers 改成 static block-diagonal grouped conv：`groups_x=4`, `groups_skip=2` |
| 14 Grouped QKV projection | whole model **1.39 M → 1.10 M/frame**（−21%）；QKV local −75%；attention local −28%；duty **1.02 → 0.81** | `qkv_groups=4` 只保留 diagonal 64×64 Q/K/V blocks |

註：12 skip subset 主要讓 skip activation bandwidth 減半，已包含在 grouped-decoder 結果中。
Top-4 sparse attention 改變 transport 與 multiplier 實作，不改 normalized MAC count。

---

## Parameter / stored-weight reduction steps（只列 storage 有變的步驟）

這張表分開看 logical PyTorch parameter count 和 RTL stored-weight basis；
在 masking/grouping 後，晶片不會儲存不存在的 off-diagonal weights。

| Step | Parameter / stored-weight change | Why it changes |
|---|---|---|
| Original BCCTN → hardware-sized baseline | **3,973,288 → 830,300** params（−79.1%） | FFT/model downsizing：`fft_len=512→128`, `embed_dim=512→256`, `num_heads=32→16`，轉成硬體可承受的 working point |
| 09 Tied complex MHA | **830,300 → 567,132** params（−31.7%；MHA-local −50%） | 一份 shared MHA weight set 同時服務 tied real-MHA 兩個 pass |
| 12 Decoder grouped pointwise | **567,132 → 485,332** params（−14.4%） | decoder pointwise 變成 grouped/block-diagonal；fixed top-50% skip subset 移除不用的 skip input |
| 14 Grouped QKV projection | PyTorch count 維持 **485,332**，但 RTL stored weights 變成 **337,876**；Q/K/V RTL **196,608 → 49,152**（−75%） | `qkv_groups=4` 代表 off-diagonal Q/K/V blocks 在 RTL 中不存在 |
| W4A8 / INT4 storage | logical weight count 不變，但 INT4 layer 的 stored bytes 約為 FP8 的 **一半** | final RTL weights 除了明確列出的 FP8 outlier tensors 外都是 INT4 |
| Cumulative original → final RTL basis | **3,973,288 → 337,876**（−91.5%），尚未計入 INT4 byte savings | hardware-sized model + tied MHA + grouped decoder + grouped QKV |

如果要畫單一條「parameter / stored-weight reduction」曲線，請用下面這組 series。
第一點是 original PyTorch parameter count，因為原始 BCCTN 不是 RTL-targeted；
最後一點切到 RTL stored-weight basis，因為 grouped QKV 會移除晶片根本不儲存的 weights。

| Plot point | Value to plot | Basis |
|---|---:|---|
| Original BCCTN (`fft_len=512`, `embed_dim=512`) | **3,973,288** | PyTorch params = encoder 274,988 + transformer 3,151,360 + decoder 546,940 |
| Hardware-sized FP8/QAT baseline | **830,300** | PyTorch params |
| 09 tied complex MHA | **567,132** | PyTorch params after MHA tying |
| 12 decoder grouped pointwise / skip subset | **485,332** | PyTorch params after decoder grouping |
| 14 / 18 final RTL stored-weight basis | **337,876** | RTL stored/computed weights after grouped QKV；畫 chip storage 請用這個點 |

---

## 為什麼架構演進中保留 W4A8

早期分析不是因為 W4A8 讓 PE array 更忙才選它；真正原因是它改變了 memory / weight residency 問題：

- Original attention dominates model：01 analysis 量到 attention 佔 **77.6%** MAC。
- MHA weights 是最大的 reusable object。02 report 估 MHA real+imag FP8 matrices 是 **512 KB**，
  遠超過 64 KB WBUF；uniform INT4 在後續結構 grouping 之前就能先降到 **256 KB**。
- Layer-wise PTQ 顯示 INT4 是最低且穩定的 uniform weight width。INT3 單層看起來可以，
  但整網路一起套會因 cumulative error 失敗。
- 最後的 14 grouped-QKV 才讓 W4 真正放進更小的 WBUF：QKV weight storage 從 **96 KB → 24 KB**，
  可放進 final 32 KB WBUF target。
- 後續 18 工作保留同樣 W4A8 topology，但移除剩下的 multipliers：
  INT4×activation 變 shift-add，scale 變 APoT/pow2，FP8 outlier/attention multiply 變 L-Mul。

所以 W4A8 首先是 **memory/SRAM/DMA 選擇**，其次才是 multiplier-removal platform。
最後 MBSTOI 能 recovery 是因為 QAT/finetune 是在 deployed arithmetic 下做的，不是 W4A8 本身自動帶來。

---

## Numerical / multiplier-free（folders 15 → 18）

從 14 composite 之後，structure（params、MAC count）不再改；這些階段改的是
**arithmetic and bit-width**，不是 topology。目標是移除 neural core 中所有 multiplier，
並把 weight memory 減半。

| Stage | HW architecture change | Weight mem / arithmetic | VCTK | TIMIT |
|---|---|---|---|---|
| **W4A8** | INT4 weights + FP8 activations；scale：PW per-group(32) / DW per-channel / MHA per-head | weight SRAM/DMA 相對 FP8 **減半** | （約等於 0.8354） | — |
| **APoT weight scale** (`apot_clip[-16,-1]`) | per-group scale = `2^a ± 2^b` = 2 shift + 1 add（原本是 real mult） | scale mult → shift-add | 0.8350 | 0.7892 |
| **Conv shift-add** | INT4 code × FP8 = shift-add MAC + delayed per-group scale | conv multiplier → shift-add | — | — |
| **Attention L-Mul m8 + FP8-outlier L-Mul m4** | QK/Score×V 和 12 個 FP8 PW layers 用 L-Mul | attention/outlier multiplier → exp-add + mantissa-add | — | — |
| **BN fold → APoT[-9,4]** | fold 成 `a_q·x + b`，effective `a=γ/√(var+ε)` 量化 | BN multiplier → 2 shift + 1 add | — | — |
| **PReLU pow2[-8,0]** | slope → `±2^k` | PReLU multiplier → sign-mux + 1 shift | — | — |
| **= 18 strict core（exact descale）** | **整個 neural core zero-multiplier** | 只用 shift / add / mux；無 mult、無 divider、無 exp LUT | 0.835531 | 0.790234 |
| **= 18 DEPLOY（delayed APoT descale, FT）** | + FP8-outlier descale = delayed 2-term APoT shift-add `[-16,-6]`（無 per-output mult）；activation 仍是 **dynamic** per-frame amax/descale | strictly shift/add/mux | 0.8365 | 0.7912 |
| **★ = 18 pow2 BLOCK-FLOAT DEPLOY (FT)** | + activation = **e4m3 raw code + per-tensor pow2 shared exponent**；activation descale fuse 到 weight descale，成為單一 post-acc **exponent offset**（shift-add）。**沒有 runtime activation amax/descale**。eager==fused exactly（relL2=0）⇒ eval 就是 HW | strictly shift/add/mux；**zero runtime activation scaling** | **0.8357** | **0.7898** |

Net 14→18：params/MAC 不變，weights 用 INT4（SRAM/DMA 減半），**neural core 每個 multiply 都被 shift/add/mux 取代**。
Exact-descale strict core（0.835531 / 0.790234）仍然代表 12 個 FP8 outlier layer 每個 output 需要一次 real multiply
來套 per-tensor descale；**deploy point** 把這個 descale 改成 delayed 2-term APoT shift-add（`[-16,-6]`），
再在這個 arithmetic 下做 5-epoch recovery finetune，分數變成 **0.8365 / 0.7912**，
VCTK/TIMIT gate 都過，且比有 multiplier 的 14 composite（0.8354 / 0.7880）更好。

（MAC/frame 在這裡是 **multiply-equivalent algorithmic count**：也就是 dataflow 會做多少個原本的 MAC。
實際硬體裡每個都已經是 shift-add，不是 hardware multiplier。它是 workload/throughput 指標，不是 multiplier 數量。）

**pow2 block-float activation deploy（current）。** Delayed-APoT deploy（0.8365）仍然會對 *activation*
做 dynamic scaling（per-frame amax → scale → descale，有 runtime work）。Block-float deploy 把每個 activation
改成 **e4m3 raw code + one per-tensor pow2 shared exponent**（offline calibrated，`floor(log2 448/amax)`）。
因為 activation scale 是單一 2 的次方，`LMul(code·2^k, w) == 2^k·LMul(code, w)` 精確成立；
所以 activation descale `2^-k` 可以 fuse 進每層 post-accumulation weight descale，變成純 **exponent offset**
（weight APoT a/b exponents shift by −k；shift-add、無 multiply）。結果是：**所有 runtime activation amax/descale 都移除**，
每個 quantized layer 只保留一次 post-acc descale，而且 `eager==fused` bit-exact
（48 個 outlier L-Mul calls 驗證 relL2 = 0），所以驗證出的 MBSTOI **就是**部署硬體數字。
在 pow2 activations 下做 5-epoch recovery FT 後，分數為 **VCTK 0.8357 / TIMIT 0.7898**（兩個 gate 都過）。
相對 dynamic-activation deploy 的代價是 VCTK −0.0008，換到 zero runtime activation scaling。
例外：MHA `out_proj` input 是 fp16 Score×V accumulator（W4A16、INT4 shift-add），不是 block-float A8；
QK activation descale 必須在 QK accumulator 上、softmax 前套完（logit temperature）。

Zero-multiplier core claim 不包含 DSP front/back-end 與獨立單元：
STFT / iSTFT、window multiply、waveform-domain mask apply。

---

## Headline

| | Weights / params basis | MAC/frame | Multipliers in core | VCTK / TIMIT |
|---|---|---|---|---|
| Original BCCTN reference (`fft_len=512`, `embed_dim=512`) | 3,973,288 PyTorch params | ≈ 29 M MAC / 100-sample frame（~4.7 G/s）— 自己的 basis，**未** normalize 到 1.10 M deploy frame | yes | 0.8519 / 0.8114 |
| Hardware-sized FP8/QAT baseline（pre-09） | 830,300 | ~1.67 M | yes | 0.8399 / 0.7966 |
| Production（14 composite） | 337,876 | 1.10 M | yes | 0.8354 / 0.7880 |
| Strict core（18, exact descale） | 337,876 (INT4) | 1.10 M | **none** | 0.835531 / 0.790234 |
| Deploy（18, delayed APoT descale, dynamic act） | 337,876 (INT4) | 1.10 M | **none** | 0.8365 / 0.7912 |
| **★ Deploy（18, pow2 block-float act）** | 337,876 (INT4) | 1.10 M | **none** + **no runtime act scaling** | **0.8357 / 0.7898** |

從 original full-size BCCTN reference 到 final RTL weight set，
stored/computed parameter basis 從 **3,973,288 → 337,876**（約 **−91.5%**），
這還沒把 INT4 storage 帶來的 byte savings 算進去。相對 original reference，
品質代價約為 −0.016 VCTK MBSTOI / −0.022 TIMIT MBSTOI；
final neural core 則移除了所有 general multiplier、divider 與 exp LUT。

Exact-descale strict core 把 per-tensor FP8-outlier descale 存成 exact fp32 scale
（代表 12 個 outlier layers 每個 output 需要一次 real multiply）。**Deploy point**
把它換成 **delayed 2-term APoT descale**（`2^a±2^b`，range `[-16,-6]`），
fold 進 L-Mul output；整條路完全 shift-add，沒有 multiplier。
在 delayed APoT descale 下做 5-epoch recovery finetune
（`BCCTN_OUTLIER_DELAYED_DESCALE=1`）後，分數為 **VCTK 0.8365 / TIMIT 0.7912**，
高於 gate 0.835 / 0.7880，也高於 exact-descale 0.8355 / 0.7902。

Current deploy ckpt（完整路徑）：
`/raid/R12K41024/18_w4a8_pow2_mixed_lmul_hw_handoff_20260530/ckpt/strict_core_pow2_blockfloat_deploy_ep3_vctk0.8357_timit0.7898.ckpt`

Key source anchors:

- Original BCCTN config/result：`/raid/R12K41024/BCCTN/config/model.yaml`,
  `/raid/R12K41024/BCCTN/results_May21/vctk_paper_style/`。
- Hardware-sized FP8/QAT baseline result：`/raid/R12K41024/main_model/results_May21/vctk_paper_style/`。
- Utilization motivation：`/raid/R12K41024/01_utilization_analysis/01_utilization_analysis_report.md`。
- W4A8 bit-width motivation：`/raid/R12K41024/02_weights_bitwidth/02_weights_bitwidth_analysis_summary.md`。
- Mixed-precision QAT design：`/raid/R12K41024/07_mixed_precision_qat/07_mixed_precision_qat_handoff.md`。
- Per-op hardware mapping：`HANDOFF_HW_ZERO_MULTIPLIER_CORE.md`（§3.4 = descale）。
