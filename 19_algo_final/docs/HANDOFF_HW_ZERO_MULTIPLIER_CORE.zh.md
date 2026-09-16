# 交接文件 — 零乘法器神經推論核心（硬體對應）中文版

> 本檔為 `HANDOFF_HW_ZERO_MULTIPLIER_CORE.md` 的中文對照版，供自己閱讀。
> 內容以英文版為準；公式 / 程式碼 / config 名稱維持原文不翻譯。

對象：BCCTN 雙耳語音增強推論核心（encoder convs + bottleneck MHA + decoder convs
+ BN + activations）。本文逐個 op 規範硬體要實作的算術，以及每個 op 如何對應到
**移位 / 加減 / mux + 小型常數表**。**整個神經核心不需要任何通用乘法器。**

不在範圍內（DSP 前/後端，獨立單元，仍用乘法器）：見 §6。

---

## 0. 主張與慣例（Claim & convention）

- **主張**：神經推論核心內每一個乘法都被換成 移位 + 加/減 + 號位 mux，加上載入時
  預先算好的常數。**runtime 沒有通用乘法器、沒有除法器、沒有 `exp` LUT。**
- **權重 scale 粒度依層型而定**（不是單一規則）：Conv pointwise = 每 32 個 input
  channel 一組；Conv depthwise = 每個 output channel；MHA Q/K/V/O 投影 = 每個 head。
  見 §1 / §3.1。
- **Block-float（區塊浮點）activation 邊界**：activation 進入每個 Conv/Linear/MHA
  算術區塊時，型態是 **e4m3 raw code（原始碼）+ 每個 tensor 一個 pow2 共享指數**
  （block floating-point）。共享指數 `k` 是離線校正常數（`k = floor(log2(448/amax))`）；
  **runtime 不做任何 amax / scale / descale**。activation 的 descale `2^-k` 被融合
  （fuse）進對應的「累加後 descale」，變成純粹的**指數 offset**：
  - 一般「實數」區塊：layer 輸入只有一個 `k`。
  - **複數 conv / projection 區塊**：同一個實數 sub-conv 權重會同時被乘到 real 與
    imag 兩個 activation slice；當 `k_real != k_imag` 時，硬體必須**依 branch 選對應的
    fused descale**（`k_real` 或 `k_imag`）（見 §3.1a）。
  因為 `k` 是單一 2 的次方，`LMul(code·2^k, w) == 2^k·LMul(code, w)` 完全成立，所以這個
  融合是 **bit-exact**（eager==fused，已驗證 relL2 = 0）—— 軟體 MBSTOI 等於部署硬體
  的數字。BN 與 PReLU（§3.5–3.6）用較寬精度算 folded affine；其輸出要在
  **下一個算術區塊輸入處重新量化回 e4m3（block-float）**。
  *例外*：MHA 內部 `out_proj` 直接吃 fp16 的 Score×V 累加值（W4A16，INT4 移位加），
  那裡沒有 block-float 指數（見 §2、§4）。
- 所有 per-channel / per-group 常數（§5）都是從訓練好的 checkpoint 一次算好、載入晶片；
  runtime datapath 永遠不乘。

---

## 1. 數值格式（Number formats）

| Tensor | 格式 | 備註 |
|---|---|---|
| Conv/Linear/MHA-proj 權重 | INT4 對稱，code `q ∈ [-7,7]` | 4-bit 有號，qmax=7 |
| 進 Conv/Linear/MHA 區塊的 activation | **block-float**：e4m3 (1-4-3) raw code + 每 tensor pow2 共享指數 `k` | 靜態校正 `k=floor(log2 448/amax)`；無 runtime scale；descale `2^-k` 融合進累加後權重 descale（§0、§5） |
| Decoder PW-outlier 權重（12 層） | FP8 e4m3，**per-tensor scaled**（`code=fp8(w·448/amax)`、descale `amax/448`） | 保留 FP8，非 INT4；見 §3.4 |
| Conv **pointwise** 權重 scale | APoT `2^a ± 2^b`，**每 32 一組**，`a,b ∈ [-16,-1]` | 2 移位 + 1 加 |
| Conv **depthwise** 權重 scale | APoT，**每 output channel**，`a,b ∈ [-16,-1]` | 2 移位 + 1 加 |
| **MHA Q/K/V/O proj** 權重 scale | APoT，**每 head**，`a,b ∈ [-16,-1]` | 2 移位 + 1 加 |
| BN folded scale | APoT `2^a ± 2^b`，**每 channel**，`a,b ∈ [-9,4]` | 2 移位 + 1 加 |
| PReLU 斜率 | pow2 `±2^k`，**每 channel**，`k ∈ [-8,0]` | 1 移位 + 號位 |
| Conv MAC 累加器 | 寬 FP（≥fp16） | group-partial 與 APoT scale 之後的 rounding 點必須明定（§3.1） |
| Attention datapath 累加器 | fp16 | |

---

## 2. Op → 硬體摘要

| 核心 op | 算術 | 硬體 |
|---|---|---|
| Conv / Linear（PW 每 32 一組、DW 每 channel） | INT4 `q` × block-float e4m3 act + APoT scale | 移位加 MAC + 延後融合 APoT/pow2 scale |
| MHA in_proj / 外層複數投影 | INT4 `q` × block-float e4m3 act + **每 head** APoT scale | 移位加 MAC + 延後 scale；複數投影依 real/imag branch 選指數 |
| MHA 內部 out_proj | INT4 `q` × fp16/寬 Score×V 累加值 + **每 head** APoT scale | W4A16 移位加 MAC 例外；仍無乘法器 |
| Attention QK^T | L-Mul (m=8) | 指數加 + mantissa/sig-code 加（邏輯 8b；以 10b/16b slice 實作）+ offset |
| Attention softmax | top-4、exp→pow2（無 mantissa）、pow2 分母 | 移位 + 移位除法，無 exp LUT / 除法器 |
| Attention Score×V | L-Mul (m=8) over top-4 packet | 指數加 + mantissa/sig-code 加 + gather |
| Decoder PW outlier（12 個 FP8 層） | FP8 × FP8 L-Mul (m=4) | 指數加 + mantissa/sig-code 加（邏輯 4b；6b 可塞進一個 8b slice） |
| BatchNorm | folded per-channel affine `a_q·x + b` | 2 移位加（APoT）+ 加 |
| PReLU | `x≥0 ? x : (±2^k)·x` | 號位 mux + 1 移位 |

---

## 3. 各 op 細節

### 3.1 Conv / Linear / MHA 投影 — W4A8 block-float 移位加 MAC + 延後 APoT scale

**Conv-PW / grouped-Linear**（scale group = 32 input channels）：

```
partial = Σ_{i in group}  q_i · code_i
out     = Σ_groups  (APoT_scale_g · 2^-k_branch) · partial_g
```

`code_i` 是已移除 block-float 指數的 raw e4m3 activation 值。一般實數區塊
`k_branch = k`；複數區塊則依該實數 MAC branch 用到的 real/imag slice 選 `k_branch`
（見 §3.1a）。

scale-group 邊界**依層型而定**（同一條 datapath、不同常數粒度）：**Conv-DW** = 每
output channel 一個 scale（input 不分組）；**MHA Q/K/V/O 投影** = 每 head 一個 scale
（`APoT_scale_head` 套在該 head 的累加上）。

- `q_i · code_i`：`q_i` 是 4-bit 有號整數（|q|≤7 ⇒ ≤3 有效位元）。乘積 = 對
  e4m3 值 `code_i` 做 `q_i` 的移位加（CSD 形式 ⇒ ≤2 次加/減）。無乘法器。
- 移位加乘積累加在寬 FP 累加器。
- **延後 scale（delayed scale）**：一個 scale group 累加完後，把 partial sum 乘上
  `APoT_scale · 2^-k_branch`。pow2 的 activation 因子是指數 offset；APoT 部分是
  `2^a ± 2^b`（2 個 barrel shift + 1 加/減），再跨 group 累加。scale 可從加總提出，
  所以「先累加再 scale」與「先把權重 dequantize」在代數上**等價**——但只有在
  **精確/寬累加**下才 bit-identical。硬體**必須**定義 rounding 點（每個 group-partial
  後、APoT scale 後）；不同順序的 FP16/窄累加可能不同。
- **scale 粒度依層型**（晶片要存的）：Conv pointwise = 每 32 input channel 一個 APoT
  code；Conv depthwise = 每 output channel；**MHA Q/K/V/O 投影 = 每 head**。皆用
  `a,b ∈ [-16,-1]`。
- **APoT 標準形**：第二項 `±2^b` 是*可選的*——只有 `b < a` 時才有（否則 0）。權重 scale
  量值恆 < 1，整體 scale 為正；只存**殘差（第二項）號位**。（BN，§3.5，可能還需整體號位。）
- L 與 R channel 共用同一份權重（WBUF 單份）。

Config trace：`int4_scale_format=apot_clip`、`int4_group_size=32`、
`BCCTN_APOT_{A,B}_{MIN,MAX}=[-16,-1]`、`qat_scope=full`。

### 3.1a 複數 conv/投影的 branch 排程 — per-branch activation 指數

複數 conv/投影用四個實數 branch MAC。原始碼慣例：

```text
apply_complex(fr, fi, x)
y_real = fr(x_real) - fi(x_imag)
y_imag = fr(x_imag) + fi(x_real)
```

因此同一個實數 sub-conv 權重（`fr` 或 `fi`）會被兩個 activation branch 重複使用。當
`k_real != k_imag` 時，硬體**不能**假設每個權重只有一個 fused activation 指數。branch
scale 排程如下：

```text
fr 配 x_real: APoT(fr) · 2^-k_real
fr 配 x_imag: APoT(fr) · 2^-k_imag
fi 配 x_imag: APoT(fi) · 2^-k_imag
fi 配 x_real: APoT(fi) · 2^-k_real
```

Decoder split-pointwise 各 stage 有獨立的 `x` 與 `skip` 輸入（`groups_x=4`、
`groups_skip=2`）。分別算 x-path 與 skip-path 的複數 pointwise，加上 `point_r/point_i`，
再餵給 depthwise transpose stage。若 real/imag activation 指數不同，每個 branch 選對應的
fused descale 變體。TB/export 產物明確編成 `*_fused_act_kreal` 與 `*_fused_act_kimag`；
詳見
[`fused_descale_summary.json`](../hw_patterns_final/tb_pattern/weights/fused_descale_summary.json)。
`branches_collapse=true` 表示兩個變體數值相同、硬體可共用常數；`branches_collapse=false`
表示兩者不同。目前已知不 collapse 的情況：**decoder stage 3（`k_real=5`、`k_imag=6`）**。

`mattn.input_proj` 與 `mattn.output_proj` 也是複數投影。那裡要保留各自的 real/imag
activation 指數；目前 TB export 把這些 MHA 複數投影標為「未融合成單一 per-weight descale」。

### 3.2 Attention QK^T 與 Score×V — L-Mul (m=8)

L-Mul 在沒有乘法器下近似浮點乘 `x·y`：

```
sign(out) = sign(x) xor sign(y)
exp(out)  = exp(x) + exp(y) + offset(m)        # 整數加 + 常數
mant(out) = mant(x) + mant(y)                  # m=8-bit mantissa 加（+ 進位到指數）
```

- m = 8 mantissa bits。硬體 = 1 個指數加法器 + 1 個（8-bit）mantissa 加法器 + 常數
  offset；無乘法器。
- QK scale：`qk_scale_mode=pow2`；`head_dim=16` 時 `1/sqrt(head_dim)=1/4=2^-2`。
- QK 的 block-float 指數**必須在 softmax 前**套用，因為它會改變有效 temperature：

  ```text
  q_eff     = q_code · 2^-k_q
  k_eff     = k_code · 2^-k_k
  score_eff = LMulAcc(q_code, k_code) · 2^-(k_q + k_k) · 2^-2
  ```

  `2^-(k_q+k_k+2)` 因子是 QK 累加後、softmax 前的指數/移位。不可延到 softmax 之後。
- Score×V 跑在 **top-4 sparse packet**（§3.3）上：gather 選中的 4 個 V row，逐一 L-Mul
  乘上其機率再累加。機率 packet `p_eff = p_code · 2^-8`、block-float V `v_eff = v_code · 2^-k_v`：

  ```text
  attn_out_eff = LMulAcc(p_code, v_code) · 2^-(8 + k_v)
  ```

  在 MHA 內部 `out_proj` 之前套用。該 `out_proj` 是 W4A16 例外：它吃 attention 累加值，
  而非重新量化的 A8 block-float 邊界。
- QK 與 Score×V 都用 fp16 累加。

Config trace：`attention_datapath.lmul_enabled=true`、`lmul_targets=[qk]`、
`prob_packet.score_v_mode=lmul`、`lmul_mantissa_bits=8`、`lmul_bit_exact=true`、
`fp16_accumulator=true`。

### 3.2a L-Mul 精確整數 datapath（RTL 參考）

這是部署核心實際用的 bit-exact 算術（source：
`DCNN/utils/complexPyTorch/complexLayers.py::_lmul_multiply_bit_exact`，由
`lmul_bit_exact=true` 啟用）。是 m=4（decoder outlier，§3.4）與 m=8（attention
QK / Score×V，§3.2）兩個 L-Mul 單元的契約。

**每個 operand 解碼。** QK^T 與 decoder outlier 的兩個 operand 都是 FP8 e4m3（block-float
raw code）。但 **Score×V 的機率 operand 是 q0.8 packet code `c`**（8-bit 無號、scale 256），
**不是** e4m3——**不要把它轉成 e4m3**。直接把 code unpack 成 L-Mul operand 欄位（source 是把
dequant 的 `c/256` 餵進同一條 m8 L-Mul，`complexLayers.py::_q08_packet_lmul_matmul`）：

```text
if c == 0: product = 0
n       = floor(log2 c)        # 8-bit CLZ / priority encoder, n ∈ 0..7
exp_p   = n - 8                # /256（q0.8 scale）折進指數——無除法器
frac8_p = (c - 2^n) << (8 - n) # normalize 到 m8 fraction；8-bit code ⇒ 無損
sign_p  = 0
```
這就是 `frexp(c/256)`。V 是 e4m3/block-float operand。然後**同一個 m8 mantissa 加法器**照跑：
`sig_code = 256 + frac8_p + frac8_v + 16`、carry-renorm、`exp_out = exp_p + exp_v + carry − k_v`。
只新增 q0.8 unpack 前端（CLZ + shift + zero-detect）；**沒有獨立 scaler**——`/256` 與 V 的
descale `2^-k_v` 都是 L-Mul 指數路徑上的指數 offset。

**一般 operand 解碼（e4m3 operand——QK、decoder outlier、以及上面的 V）：**

```text
x = (-1)^sx · 2^ex · (1 + fx)        # ex = 無偏指數, fx ∈ [0,1) 為小數部
fx_code = round_or_floor(fx · 2^m)   # m-bit 小數 code，整數 ∈ [0, 2^m - 1]
```
e4m3 operand 的存放 mantissa 是 3 bits，所以 `fx_code = mantissa << (m-3)`
（`m≥3` ⇒ e4m3 mantissa **無損**承載；低 `m-3` 位為 0）。

**L-Mul 的「乘法」= 一個小整數加**（取代 mantissa 乘積）：

```text
sig_code = (1<<m) + fx_code + fy_code + offset_code         # 整數加
           # offset_code = round(2^m · 2^-l(m));  l(4)=3 ⇒ offset_code=2
           #                                       l(8)=4 ⇒ offset_code=16
carry    = sig_code >= (1<<(m+1))                            # 1-bit 比較
sig_code = carry ? (sig_code >> 1) : sig_code                # renormalize（丟 LSB）
exp_out  = ex + ey + carry + Δexp_fused                      # 整數加
sign_out = sx XOR sy
product  = (-1)^sign_out · 2^exp_out · (sig_code / 2^m)      # significand = 1.f，m-bit
```

- `Δexp_fused` 把 block-float activation descale `2^-k`（§0）與 weight descale 的
  pow2 部分折進來——純加法**指數 offset**，無乘法。
- **operand 加法寬度**（slice 要容納的位元），因 `(1<<m)+fx+fy+offset ≤ 3·2^m`：
  - **m=4**：`sig_code ∈ [18, 48]` ⇒ **6-bit 無號加**（即「m4 6-bit」預算）。
  - **m=8**：`sig_code ∈ [272, 782]` ⇒ **10-bit 無號加**。
- **每次 L-Mul 硬體**：1 個號位 XOR + 1 個指數加法器 + 1 個 `(m+2)`-bit mantissa 加法器
  + 1 個 carry 比較 + 條件式 1-bit 移位/+1。**無乘法器、無 LUT。** `product` 接著進
  fp16 累加器（指數對齊 + 加 = 一般 FP 累加；那個 aligner/adder 是唯一的寬算術，且是
  **加法器**不是乘法器）。
- L-Mul 是**近似**乘法：丟掉 `fx·fy` cross-term、加一個固定 `offset_code` 重新置中
  bias。部署模型就是在這個算術下訓練的（`lmul_bit_exact=true`），所以它是契約，不是
  要去「修正」的近似。

### 3.3 Softmax — `shift_topk_pow2_clamp8_nomantissa`（機率區塊）

每個 attention row（序列位置）：

1. **Top-4 select**：取最大的 4 個分數，其餘丟 0。
2. **exp → pow2，無 mantissa**：`w = 2^clamp(floor(score'), -8, 0)`（純 2 的次方；丟掉
   `1+frac` mantissa 項）。= 1 移位。
3. **分母 → pow2**：`denom = round_pow2(Σ_top4 w)`；除以 2 的次方 = 1 移位。
4. 輸出 **4-entry sparse 機率 packet**，量化成 q0.8（8-bit、scale 256），每筆附 5-bit
   local index（≈52 bits/row，相對 dense FP16 row 的 512 bits）。

機率區塊內：無 `exp` LUT、無除法器、無通用乘法器。

Config trace：`attention_softmax.mode=shift_topk_pow2_clamp8_nomantissa`、
`prob_packet.{top_k=4, local_idx_bits=5, prob_bits=8, prob_scale=256}`。

### 3.4 Decoder pointwise outliers — FP8 × FP8 L-Mul (m=4)

12 個 outlier pointwise 層保留 FP8（非 INT4）權重：
`decoder.model.{0,3,4}.0.pointwise_{r,i}_{x,skip}`。用 m=4 mantissa 的 L-Mul 相乘——
足以承載 FP8 e4m3 的 3-bit mantissa 而不額外截斷，但 **L-Mul 仍是近似 FP8 乘法**
（mantissa 加取代 `(1+m_x)(1+m_y)`，丟掉 cross-term）。硬體同 §3.2，只是窄（4-bit）加法器。

**權重格式 = PER-TENSOR SCALED FP8（不是 raw e4m3）。** 訓練模型對這些權重用 per-tensor
動態 scale，所以晶片要存*已 scale* 的 code，不是 `w.to(e4m3)`：

```
s      = 448.0 / amax(|w|)                 # per-tensor 純量（448 = e4m3 max）
code   = fp8_e4m3(w · s)                    # 存在晶片上的 e4m3 bits
weight = code · (amax/448) = code / s       # == 驗證過的 module.weight（bit-exact）
```

若改存 `fp8_e4m3(w)` 是**錯的**（小量值權重浪費 e4m3 動態範圍；實測端到端 ~4% 相對輸出
L2 誤差）。export 現在寫 `w_fp8.e4m3.bin = fp8(w·s)`、per-tensor `fp8_scale`（`448/amax`）
與 `fp8_descale`（`amax/448`），外加 APoT 編碼的 descale。

**Descale 與零乘法器主張 — 已解決（延後 APoT descale，recovery-finetune）。** per-tensor
descale（`amax/448`）是*每層單一常數*，可從逐元素 L-Mul 提出：
`Σ (code_i / s)·a_i = (1/s)·Σ code_i·a_i`。所以它在 fp16 累加器上**每個輸出套一次**，不是
每個 MAC——即**延後（累加後）scale**，和 INT4 conv 的延後 APoT scale（§3.1）一樣。部署核心
把它實作成 **2-term APoT（`2^a ± 2^b`）指數 offset / 2 移位加，折進 L-Mul 輸出**（無乘法器），
export 成 `fp8_descale_apot_*`。

這個 APoT descale 是精確 `amax/448` 的近似（2-term APoT ⇒ 每層最多 ~9% 誤差；把它一次性
折進*部署*權重會吃 ~4% 輸出 L2、VCTK 掉到 0.8335）。因此用 **5-epoch 在延後 APoT descale 下
訓練的 recovery finetune**（`BCCTN_OUTLIER_DELAYED_DESCALE=1`）收尾：模型適應部署算術後回升到
**VCTK 0.8365 / TIMIT 0.7912**——雙 gate 都過、也贏 exact-descale 部署（0.8355 / 0.7902）。
所以嚴格零乘法器核心（descale = 移位加）就是部署點；不需要 per-output 乘法。
**APoT 指數範圍 = `[-16, -6]`**（資料驅動：12 個 descale 指數落在 `[-14,-8]`；4-bit 欄位留
margin，不 clamp）。

**目前部署 = pow2 block-float activations（涵蓋以上）。** 延後 APoT 權重 descale 不變，但
activation 現在是 block-float（§0/§1/§5）：activation descale `2^-k` 折進*同一個*累加後 descale
作為指數 offset。一般 branch 把 APoT a/b 指數移 layer 輸入的 `k`；decoder split-pointwise 複數
branch 則由 TB/export 產 `kreal/kimag` fused 變體、硬體依該 branch 用到的 activation slice 選
變體（§3.1a）。因為 `k` 是單一 2 的次方，`eager==fused` 為 bit-exact（12 個 outlier 的 48 個
L-Mul call 全部驗證 relL2 = 0），所以驗證分數就是硬體數字。pow2 recovery FT 得
**VCTK 0.8357 / TIMIT 0.7898**（雙 gate），且**移除所有 runtime activation amax/scaling**。

部署 ckpt：`ckpt/strict_core_pow2_blockfloat_deploy_ep3_vctk0.8357_timit0.7898.ckpt`
（activation 校正表 `ckpt/activation_calibration_p999_b64.json`；block-float 指數 export 在
`export_out_strict18/activation_blockfloat/`）。
前一版 dynamic-activation 部署：`strict_core_apot_descale_deploy_ep2_vctk0.8365_timit0.7912.ckpt`。

Config trace：`low_power.outlier_fp8_lmul.enabled=true, mantissa_bits=4`；
source quantizer = `DCNN/utils/fp8_qat.py::FP8QuantizeSTE`（per-tensor amax scale）；
延後 descale runtime = `all_lmul_multiply.py`（`BCCTN_OUTLIER_DELAYED_DESCALE`），APoT
範圍 `BCCTN_OUTLIER_DESCALE_APOT_{MIN,MAX}=[-16,-6]`。

### 3.5 BatchNorm — folded per-channel affine（APoT scale + bias）

BN **不可**折進複數 conv（實數路徑在 real 與 imag 輸出都要用到權重 ⇒ 會讓 conv 權重加倍）。
它是**獨立的 per-channel affine**。real 與 imag 部分各自獨立。

每 channel 預先算（載入時，從凍結 running stats）：

```
a   = gamma / sqrt(running_var + eps)
a_q = APoT(a),  a,b 指數 ∈ [-9,4]
b   = beta - a_q · running_mean            # 常數，與 a_q 一起折
```

Runtime：`y = a_q · x + b` =（2 移位 + 1 加）·x + 加。無乘法器。

- **號位儲存**：不像 conv 權重 scale（恆正），`gamma` 可能為負 ⇒ BN scale `a` 可能為負。
  存**整體號位**加殘差（第二項）號位，即 `a_q = sign·(2^a ± 2^b)`。第二項可選（`b < a` 才有）。
- **關鍵**：量化*有效* scale `a = gamma/sqrt(var+eps)`，**不是** `gamma` 本身——否則
  `1/sqrt(var+eps)` 會在推論時重新引入真乘法。
- 指數範圍 `[-9,4]` 由訓練 checkpoint 驅動：folded `|a| ∈ [0.0037, 14.8]`、
  `log2|a| ∈ [-8.07, +3.89]`，約 34% channel 有 `|a| ≥ 1`（需要正指數——不可沿用 conv
  權重 scale 的 `[-16,-1]`）。

Config trace：`low_power.bn_affine.scale_mode=fold_apot`、
`pow2_min_exp=-9, pow2_max_exp=4, tie_real_imag_scale=false`。

### 3.6 PReLU — pow2 斜率（號位 mux + 移位）

每 channel（real 與 imag 各自獨立）：

```
y = x            if x >= 0
y = (±2^k) · x   if x <  0          # k ∈ [-8,0]，每 channel 存號位
```

Runtime = 號位 mux + 1 移位。無乘法器。

- 斜率 `α` 量化成 `±2^k`，`k ∈ [-8,0]`。資料驅動：`|α| ∈ [0.006, 0.40]`（皆 < 1）。
  **號位保留**（有些學到的斜率為負）。

Config trace：`low_power.prelu_pow2.enabled=true, min_exp=-8, max_exp=0`。

---

## 4. Dataflow / 記憶體（結構）

- **PE array**：64 個 complex MAC lane，1-D flat。
- **PE 算術 slice — 一個共用加法器涵蓋全部 4 個 mode。** PE 核心是單一可組態的
  加/移位單元；4 個算術 mode 只差在 operand 怎麼餵進去、以及整數加多寬。用一個
  **6-bit base slice**（剛好容納 L-Mul m=4 mantissa 和，§3.2a）組成，可 **chain**
  （slice 間帶進位）成 12-bit（2 slice）或 18-bit（3 slice）。兩種角色共用此加法器：
  - **INT4 權重 mode（W4A8, W4A16）**——*significand 移位累加*。INT4 code `q`（|q|≤7，
    CSD ⇒ ≤2 個有號 digit、移位 0..3）gating **activation significand** 的移位副本；
    加法器把它們加起來。activation significand = 4 bits（e4m3，hidden-1+3）for W4A8、
    11 bits（fp16，hidden-1+10）for W4A16。權重指數 / APoT-pow2 / activation `2^-k`
    之後以延後 scale（指數 offset）套用。
  - **L-Mul mode（m4, m8）**——*mantissa 加*（§3.2a）：`sig_code = (1<<m) + fx + fy +
    offset`，一個加，再 carry-renorm。

  | mode | 加法器加的東西 | 需要整數寬度 | 6-bit base slice 數 |
  |---|---|---|---|
  | **L-Mul m4**（decoder outlier §3.4） | `(1<<4)+fx+fy+2`，≤48 | **6 b** | **1** |
  | **W4A8**（INT4×e4m3 §3.1） | 4-b significand `<<`0..3，±2 項 | **≈8 b**（7-b 量值 + 號位） | 2（chain） |
  | **L-Mul m8**（QK / Score×V §3.2） | `(1<<8)+fx+fy+16`，≤782 | **10 b** | 2（chain） |
  | **W4A16**（out_proj §3.2/§4） | 11-b significand `<<`0..3，±2 項 | **≈15 b** | 3（chain） |

  所以 6-bit base：m4 = 1 slice；W4A8 與 m8 = 2 串接；W4A16 = 3 串接。
  （設計註：改用 **8-bit** base slice 可讓 **m4 與 W4A8 各**塞進一個 slice、m8 + W4A16
  各塞兩個——chain 較少、每 slice 面積稍大。兩種 base 都可行，依面積/timing 預算選。
  任何 mode 都不需要真乘法器。）
  - **MHA 內部 out_proj** 是唯一的 W4A16 使用者（輸入 = 晶片上 fp16/寬 Score×V 累加值、
    非 A8 邊界）。重用同一個串接加法器；**不要**另外做一套 W4A16 PE array。
- **L-Mul 指數路徑**：L-Mul 不只是 mantissa 加法器，還需號位 XOR、指數加、mantissa 進位調整、
  以及 scale/指數 offset 路徑。因為 activation pow2 指數與延後 scale 都融合進此路徑，不要只
  按 raw 4-bit FP8 指數加法器設計；用有號指數 offset datapath（建議 ≥7 bits）或在累加前用
  既有 fp16 指數 datapath。
- **權重重用**：L 與 R channel 共用權重——WBUF 單份。
- **Buffers**：WBUF = 32 KB，UFBUF = 4 KB。feature maps 全程留在晶片上。
- **MHA Q/K/V 投影**：block-diagonal grouping `qkv_groups=4`——只存/算對角的 64×64 區塊
  （off-diagonal 權重不存在）。為達 64-lane @ 5 MHz 即時 duty（attention MAC ≈ 1.10 M/frame、
  duty ≈ 0.81）所必需。
- **Decoder pointwise**：grouped `groups_x=4`、`groups_skip=2`；skip 輸入只用 L1 量值前 50% 的
  channel（固定 index，載入一次）。Real `Conv2d(groups=…)` + channel slice——存的與算的權重都
  變少。split 複數 pointwise 時，若 real/imag activation 指數不同，每個實數 MAC branch 排對應的
  `kreal/kimag` fused descale 變體。
- **Attention 機率傳輸**：top-4 sparse packet、5-bit local index、q0.8 機率 ⇒ ≈52 bits/row
  （相對 dense FP16 512 bits）。
- **Per-group 權重 scale**：group_size = 32（每 32 input channel 一個 APoT code）。

---

## 5. 載入時常數（從 checkpoint 一次算好；runtime 不乘）

| 常數 | 粒度 | 格式 |
|---|---|---|
| INT4 權重 code `q` | 權重元素 | 4-bit 有號 |
| **Activation block-float 指數 `k`** | **activation 邊界（per-tensor）** | 有號整數（`k=floor(log2 448/amax)`）；descale `2^-k` 融合進累加後 descale。複數 conv/投影 branch 可能需各自的 `kreal/kimag` fused 變體；`branches_collapse=true` 表相等。表：`export_out_strict18/activation_blockfloat/`、`hw_patterns_final/tb_pattern/weights/fused_descale_summary.json` |
| Conv-PW 權重 APoT scale | 每 32 input ch | (a-exp, b-exp, residual-sign)，a,b∈[-16,-1] |
| Conv-DW 權重 APoT scale | output channel | (a-exp, b-exp, residual-sign)，a,b∈[-16,-1] |
| MHA Q/K/V/O proj APoT scale | **head** | (a-exp, b-exp, residual-sign)，a,b∈[-16,-1] |
| Decoder PW-outlier FP8 descale `amax/448` | **層（per-tensor）** | 精確 fp32 `fp8_descale` + APoT（a,b∈[-16,-6]，資料驅動 4-bit）；在 fp16 累加器上以延後移位加套一次（§3.4） |
| BN folded scale `a_q` | channel（r & i） | APoT（a,b∈[-9,4]）+ **整體號位** + 殘差號位 |
| BN folded bias `b` | channel（r & i） | 加法常數 |
| PReLU 斜率 | channel（r & i） | (k-exp, sign)，k∈[-8,0] |
| Decoder skip 前 50% index | 層 | 固定 channel index |

---

## 6. 範圍外 — DSP 前/後端（不在零乘法器核心主張內）

以下仍用乘法器、是獨立單元（跑在晶片 DSP 前/後端，不是神經核心）：

- STFT / iSTFT。
- 分析/合成 window 乘法。
- 波形域 mask 套用（`apply_mask`）。

若日後需要「端到端到波形的零乘法器」主張，這些要另外處理；本文明確排除它們。

---

## 7. 逐 op 算術 checklist（sign-off）

- [ ] Activations：**block-float** e4m3 raw code + 每 tensor pow2 指數 `k`（靜態校正、無
      runtime amax/scale）。descale `2^-k` 融合進對應的累加後 descale。複數 conv/投影 branch
      必須選對應的 `kreal/kimag` fused 變體；`branches_collapse` 只是共用優化、不是排程假設。
      指數表從 `activation_blockfloat/` 載入。QK descale 在 QK 累加後、softmax 前套用。
- [ ] Conv/Linear/proj：INT4×block-float e4m3 raw-code 移位加 + 延後 APoT/pow2 scale——粒度：
      Conv-PW/Linear 每 32、Conv-DW 每 output channel、MHA proj 每 head。
- [ ] MHA 結構（§8.2/§9.1）：外層複數 `mattn.input_proj` → **tied ComplexMHA = 2 個 real
      pass**（`a=M(emb.real)`、`b=M(emb.imag)`，共用權重）→ 合併 `(a−b)+i(a+b)` → 外層複數
      `mattn.output_proj`。每個 pass 各保留**自己的** K/V 歷史 ⇒ **2× K/V ring（≈32 KB）**，不是 1×。
- [ ] MHA 內部 out_proj：唯一 W4A16 例外；重用同一 INT4 移位加 PE 的 chained-slice 模式，
      輸入來自 Score×V fp16/寬累加器。
- [ ] QK^T、Score×V：L-Mul m=8、fp16 累加。QK 在 softmax 前套 `2^-(k_q+k_k+2)`；Score×V 在
      內部 W4A16 `out_proj` 前套 `2^-(8+k_v)`（Score×V 機率 operand 是 **q0.8 packet code**，
      非 e4m3——§3.2a）。
- [ ] PE 加法器 slice（§4）：一個共用 add/shift 單元、以 **6-bit base slice** 串接組成。
      L-Mul m4 = 1 slice（6 b）；W4A8 = 2（≈8 b）；L-Mul m8 = 2（10 b）；W4A16 = 3（≈15 b）。
      （8-bit base 為替代方案：m4+W4A8 各 1 slice、m8+W4A16 各 2——依面積/timing 選。**不要**
      硬寫死「2×8-bit」。）
- [ ] L-Mul 指數路徑：號位 XOR + 指數加 + mantissa 進位 + 有號 scale/指數 offset；不要只按
      4-bit FP8 指數加設計。
- [ ] Softmax：top-4、2^⌊·⌋（無 mantissa、clamp 指數）、pow2 分母。無 exp LUT / 除法器。
- [ ] Decoder PW outliers（12）：per-tensor scaled FP8（`code=fp8(w·448/amax)`、descale
      `amax/448`）、FP8×FP8 L-Mul m=4；descale 折進 L-Mul 指數（pow2/APoT）——§3.4。
- [ ] BN：folded `a_q·x + b`、`a_q` = `gamma/sqrt(var+eps)` 的 APoT、指數∈[-9,4]。
- [ ] PReLU：pow2 斜率、號位 mux + 移位、指數∈[-8,0]。
- [ ] 神經核心內無通用乘法器、無除法器、無 exp LUT。

---

## 8. 逐層運算與維度表（frame-by-frame）

**Frame-by-frame 慣例。** 核心**每個 pass 處理一個 STFT frame**（hop = `win_inc=80`
sample ⇒ 每 5 ms @ 16 kHz 一個 frame）。conv kernel 是 `(5,1)` = **頻率 5-tap、時間
1-tap**，所以每個 conv 只混頻率、各 frame 在 conv 階段**彼此獨立**；擷取到的 tensor 也
證實時間軸 `T` 在所有 conv 後不變。**唯一**跨 frame 的依賴是 MHA，它在 causal `attn_win=31`
window 上做 attention ⇒ 需 31-frame K/V 歷史（§9）。以下 feature map 都是每 frame 的
`[C, F]`（複數：一個 real plane + 一個 imag plane）；`MAC/frame` 計實數乘法等效次數 =
`Σ Cout·Cin_per_group·F_out` × **4**（複數 conv：`fr·xr, fi·xi, fr·xi, fi·xr`，§3.1a）。

**雙耳。** STFT → 每耳一個 spectrogram（L, R）。encoder 與 decoder **每耳各跑一次、共用權重**；
兩耳**只在 bottleneck 串接**（`每耳 flatten C·F=128，stack L‖R = 256 = embed_dim`），所以
MHA 對兩耳一起做 attention 再拆回。`head_dim = 256/16 = 16`。

### 8.1 Encoder（每耳；頻率 conv `k=(5,1)`、stride `(2,1)`、每 stage 後接 BN+PReLU）

| stage | depthwise（per-out-ch INT4，**W4A8**） | pointwise 1×1（per-group-32 INT4，**W4A8**） |
|---|---|---|
| enc0 | `[1,64]→[1,32]` | `[1,32]→[8,32]` |
| enc1 | `[8,32]→[8,16]` | `[8,16]→[16,16]` |
| enc2 | `[16,16]→[16,8]` | `[16,8]→[32,8]` |
| enc3 | `[32,8]→[32,4]` | `[32,4]→[64,4]` |
| enc4 | `[64,4]→[64,2]` | `[64,2]→[128,2]` |
| enc5 | `[128,2]→[128,1]` | `[128,1]→[128,1]` |

channel 是**每 real/imag path**（複數 channel 數 `kernel_num` =
`[16,32,64,128,256,256]` 是這的 2 倍）。每 stage 輸出（L1 前 50% channel）留給對應的
decoder skip（§9）。頻率：`64→32→16→8→4→2→1`。

### 8.2 Bottleneck MHA（一 frame = 一個 query；L‖R 合併；`embed=256, heads=16, head_dim=16`）

**兩層巢狀——不可混為一談（source：`model.py::MultiAttnBlock.forward`、
`complexLayers.py::_forward_tied`）：**

1. **外層複數投影** `mattn.input_proj`，attention 之後再 `mattn.output_proj`——複數
   `GroupedLinear`（`fc_r/fc_i`）、`256→256`、**W4A8**、per-head APoT。它們包住 attention、
   **每 frame 各跑一次**（但是複數 ⇒ §3.1a per-branch real/imag 指數；export 為**未融合**）。
2. **Tied ComplexMHA**（`self.mattn`，`MHA_re ≡ MHA_im = M`）——把內部 real MHA **用共用權重
   跑兩次**：`a = M(emb.real)`、`b = M(emb.imag)`，再合併 `out = (a−b) + i(a+b)`。所以內部
   MHA 是 **2 個 real pass**，各有自己的 Q/K/V 與**各自的 K/V 歷史**（§9.3 ⇒ 2× ring）。

每個**內部 pass**（`_mha_attention_forward`，causal window `attn_win=31`）：

| op | 維度（每 frame、每 pass） | grouping | MAC mode | weight scale |
|---|---|---|---|---|
| 內部 **in_proj** Q,K,V | `256→256` ×3 | **qkv_groups=4** block-diagonal：每 Q/K/V 4 個獨立 `64→64` 區塊（off-diag 不存在；比 dense −75%） | W4A8 | per-head APoT |
| QK^T | 每 head `q[16]·K_win[31,16] → 31` 分數 | 16 heads；causal window 31 | **L-Mul m8** | softmax 前套 `2^-(k_q+k_k+2)`（§3.2） |
| softmax | 每 head 31 取 top-4 → q0.8 packet | per head | 移位/pow2（§3.3） | — |
| Score×V | 每 head `top4 · V[16] → out[16]`（prob = q0.8） | 16 heads，top-4 gather | **L-Mul m8** | 套 `2^-(8+k_v)` |
| 內部 **out_proj** | `256→256` | dense（`out_proj_groups=1`） | **W4A16**（輸入 = fp16 Score×V 累加值） | per-head APoT |

每 frame 完整 MHA block = `input_proj`（一次）→ [內部 MHA ×2 pass，共用權重] → 合併
`(a−b)+i(a+b)` → `output_proj`（一次）。`qkv_groups=4` 時每個 in_proj 區塊（64 寬）橫跨 4 head。

### 8.3 Decoder（每耳；先 pointwise 再 transpose-conv；每 stage 頻率 `×2`）

每 stage：`point = pointwise_x(x, groups_x=4) + pointwise_skip(skip_subset, groups_skip=2)`，
再 `depthwise_transpose`（INT4 **W4A8**、per-out-ch、頻率 ×2），再 BN+PReLU。pointwise
存的權重 shape = `[Cout, Cin/groups]`。**skip** = 對應 encoder stage 的 L1 前 50% 子集
（固定 index，載入一次）。

| stage | pointwise_x（`groups_x=4`） | pointwise_skip（`groups_skip=2`，top-50%） | dw-transpose | pointwise MAC mode |
|---|---|---|---|---|
| dec0 | `[128,1]→[128,1]` | skip`[64,1]→[128,1]` | `[128,1]→[128,2]` | **L-Mul m4**（outlier） |
| dec1 | `[128,2]→[64,2]` | skip`[64,2]→[64,2]` | `[64,2]→[64,4]` | W4A8 |
| dec2 | `[64,4]→[32,4]` | skip`[32,4]→[32,4]` | `[32,4]→[32,8]` | W4A8 |
| dec3 | `[32,8]→[16,8]` | skip`[16,8]→[16,8]` | `[16,8]→[16,16]` | **L-Mul m4**（outlier） |
| dec4 | `[16,16]→[8,16]` | skip`[8,16]→[8,16]` | `[8,16]→[8,32]` | **L-Mul m4**（outlier） |
| dec5 | `[8,32]→[1,32]` | skip`[4,32]→[1,32]` | `[1,32]→[1,64]` | W4A8 |

12 個 **outlier** pointwise conv 是 decoder stage **0, 3, 4**（`pointwise_{r,i}_{x,skip}`），
FP8×FP8 **L-Mul m4** + per-tensor 延後 APoT descale（§3.4）；它們也是 per-branch
`kreal/kimag` 的情況（§3.1a，stage-3 `k_real=5,k_imag=6`）。decoder 輸出 = 每耳
`[1,64]` 複數 = 複數 ratio **mask**。

### 8.4 前/後端（DSP，§6 — 不在 zero-mult core）

STFT（每耳，1 frame → 64-bin 複數）、複數 mask 乘（`mask · spectrogram`）、iSTFT +
overlap-add → 增強波形。

---

## 9. Controller 排程與資料載入（每 frame）

controller 每個 hop 跑一次 **frame pipeline**。權重不可能一次全進 WBUF（32 KB）——INT4
RTL 權重共約 169 KB——所以**權重逐層從 DRAM stream 進 WBUF**（W4 packing 相對 FP8 正是
讓這個 DMA 塞進 duty 的關鍵；見 CLAUDE.md W4A8 分析）。feature map 與 skip 子集留在晶片上
（UFBUF）；attention 的 K/V 歷史是獨立的 ring buffer。

### 9.1 每 frame 執行順序（一個 hop）

```text
for 每個新的 STFT frame t（每耳 e ∈ {L,R}，除非另註）:
  S0  DSP 前端:   x_spec[e] = STFT(frame_t[e])             # [1,64] 複數  (§6)
  S1  Encoder enc0..enc5（共用權重、頻率 conv）:
        for s in 0..5:
          load Wdw[s], Wpw[s], BN[s], PReLU[s] from DRAM→WBUF   # 逐 stage stream
          h = PReLU( BN( pw( dw(h) ) ) )                        # [C_s, F_s]
          UFBUF.store_skip(s, top50%_index[s] 套到 h)           # 給 decoder skip
  S2  Bottleneck pack:  v_e = flatten(h_e)            # 每耳 128
  --- 兩耳都做完 ---
  S3  emb = concat(v_L, v_R)                          # 256 = embed_dim
  S4  MHA block（model.py MultiAttnBlock + tied ComplexMHA）:
        load W_input_proj（per-head APoT，複數 fc_r/fc_i）
        emb = mattn.input_proj(emb)                   # 外層複數投影 256→256, W4A8（一次）
        # tied ComplexMHA = 2 個 real-MHA pass，共用權重 M，再合併:
        for plane p in {emb.real → a, emb.imag → b}:  # 2 pass，各有自己的 K/V 歷史
          load W_in_proj（每 Q,K,V 4 個對角 64×64 區塊）, W_out_proj  # 兩 pass 共用
          Q,K,V = in_proj(p)                          # W4A8, qkv_groups=4
          KVring[p].push(K_t, V_t); evict frame t-31  # 31-frame causal window（每 plane 各一份！）
          for head h in 0..15:
            score = LMul_m8(Q_h, KVring[p].K[:,h]) · 2^-(k_q+k_k+2)   # 31 分數
            prob  = softmax_top4_pow2(score)                          # 4-entry q0.8 packet
            o_h   = LMul_m8(prob, KVring[p].V[top4,h]) · 2^-(8+k_v)    # prob=q0.8, [16]
          pass_out = out_proj(concat o_h)             # W4A16（輸入 fp16 累加值）
        mha = (a - b) + i(a + b)                       # 複數合併
        load W_output_proj（per-head APoT，複數）
        attn = mattn.output_proj(mha)                 # 外層複數投影 256→256, W4A8（一次）
        v_L, v_R = split(attn)
  S5  Decoder dec0..dec5（共用權重、每耳）:
        for s in 0..5:
          load Wpw_x[s], Wpw_skip[s], Wdw_T[s], BN, PReLU, skip_index[s]
          skip = UFBUF.load_skip(5-s)                  # 對應 encoder stage（同一 frame t）
          point = pw_x(x, g=4) + pw_skip(skip, g=2)    # 複數；per-branch kreal/kimag
          x = PReLU( BN( dw_transpose(point) ) )       # 頻率 ×2
        mask[e] = x                                    # [1,64] 複數
  S6  DSP 後端:  y[e] = iSTFT( mask[e] · x_spec[e] )  + overlap-add   # (§6)
```

### 9.2 controller 逐層載入

| 層類別 | 權重載入（DRAM→WBUF） | scale/常數載入 | activation 來源 |
|---|---|---|---|
| encoder dw/pw、decoder dw-T、MHA **內部** in_proj | INT4 code（4-bit packed） | 每 group/ch/head 的 APoT（a/b-exp、residual-sign）+ **fused** `−k` 指數（§5） | UFBUF（當前 frame）；raw e4m3 code |
| MHA **外層** `mattn.input_proj` / `mattn.output_proj`（複數 `fc_r/fc_i`） | INT4 code | per-head APoT + per-branch `kreal/kimag`（§3.1a，未融合） | UFBUF；每 frame 各一次 |
| MHA 內部 in_proj / out_proj（**2 個 tied pass 共用**） | 載入一次、real & imag pass 重用 | per-head APoT（in_proj）；out_proj W4A16 | UFBUF；每 plane 的 K/V ring |
| decoder pw stage 0/3/4（outlier） | e4m3 code（`fp8(w·s)`） | `fp8_descale_apot_*` **每 branch** `_fused_act_kreal/kimag`（§3.1a/§5） | UFBUF cat(x, skip 子集) |
| decoder pw stage 1/2/5 | INT4 code | per-group APoT + fused `−k` | UFBUF cat(x, skip 子集) |
| MHA QK / Score×V | —（只有 activation） | QK `2^-(k_q+k_k+2)`、Score×V `2^-(8+k_v)`；softmax pow2 常數 | Q/K/V 暫存 + **K/V ring** |
| BN / PReLU | folded `a_q`+bias / pow2 斜率+號位 | §5 表 | conv 後 inline |

### 9.3 Buffer 與容量註記

- **WBUF（32 KB）**：放當前層的權重；double-buffer（算 `s` 時 load `s+1`）以隱藏 DRAM
  延遲。最大單一權重 tensor：MHA out_proj 256×256 INT4 = 32 KB（填滿 WBUF）——排程時不要
  跟別的 tensor 同時 resident，或 by head tile。
- **UFBUF（4 KB）**：當前 frame activation（最大 `[8,64]` 複數 ≈ 2 KB/耳）**加上**留著的
  encoder skip 子集（top-50%）；隨 frame 推進重用。skip 是**同一 frame**（conv 只動頻率）
  ⇒ 不需跨 frame skip buffer。
- **K/V ring（獨立，≈32 KB）**：tied ComplexMHA 跑 **2 個 pass**（real plane、imag plane），
  各有**自己的** K/V 歷史，所以 ring 要存**兩個 plane** 的 K 與 V：`31 × 256 × 2(K,V) ×
  2(real,imag) × 1 B（e4m3 code）+ 共享指數 ≈ 32 KB`。每 plane circular：push 當前 frame 的
  K,V、evict frame `t-31`。這是唯一跨 frame 保留的 state，要明確 size（不屬 UFBUF）。
  （只有在每 frame 重算 imag-pass 歷史時 16 KB 才夠——非預設。）
- **Skip index ROM**：每 decoder stage 的 top-50% channel index（固定，§5），開機載入一次。
- **L/R**：encoder/decoder 跑兩次（權重對兩耳都 resident，所以 weight DMA 攤在 2 耳上）；
  兩耳**只串接一次**進 256 寬 bottleneck；tied ComplexMHA 接著仍跑它的 **2 個 real pass**
  （real/imag plane，§8.2）。「L‖R 合併」指的是串接，不是單一 attention pass。

### 9.4 即時預算

64 個 complex MAC lane；attention MAC ≈ 1.10 M/frame；在 5 MHz lane clock 下 duty ≈ 0.81
（留給 conv 階段 + DSP）。`qkv_groups=4` block-diagonal 投影（MHA-proj MAC −75%）與 decoder
grouping / skip 子集是把每 frame MAC 壓進即時預算的關鍵——排程要保留它們（不要展開成 dense）。

### 9.5 累加後 scaling / BN / PReLU——單一共用 PPU pool

所有 post-MAC 修正都是**同一個 shift-add / APoT / pow2 primitive**，對同一筆資料是**循序**、
對不同 output 可**pipeline**：

```text
PE reduce → delayed descale（per scale-group；fused activation 2^-k，§0/§3.1）
          → cross-group sum
          → BN APoT affine + bias（per output，§3.5）
          → PReLU pow2（per output，§3.6）
          → requant 成 e4m3 block-float（下一層輸入）
```

同一筆資料不會同時搶（element i 在 PReLU、i+1 在 BN、i+2 在 descale）；第 N 層的 BN/PReLU
藏在第 N+1 層的 MAC+descale 後面。

**用單一 PPU pool 共用——不要切成 dedicated descale/BN/PReLU lane。** MHA 佔約 78% 算力且
**沒有 BN/PReLU**；dedicated BN/PReLU lane 在整段 attention 會 idle。共用 pool 在 MHA 期間做
projection / QK / Score×V descale、在 conv 期間做 descale+BN+PReLU ⇒ PPU 利用率接近滿。

**Sizing**——`PPU_lanes ≈ PE × (PPU_ops/frame) / (MAC_ops/frame)`。檢過的每 frame 負載：

| PPU op 類 | ops/frame | reduction / scale |
|---|---:|---|
| conv/decoder delayed descale + 4 個 MHA projection descale | ≈ 48 K | 16 – 256 |
| BN APoT affine（10,752 elem；`decoder.model.5.0` 沒有） | ≈ 21.5 K | per-output |
| PReLU pow2（10,752 elem） | ≈ 10.8 K | per-output |
| **共用 PPU 總負載** | **≈ 80 K** | |

> **QK `2^-(k_q+k_k+2)` 與 Score×V `2^-(8+k_v)` 不是 PPU op。** 它們是折進 **PE 內 L-Mul
> 指數加法器**的指數 offset（§3.2/§3.2a；Score×V 的 q0.8 `/256` 也折在那裡）。不額外花成本、
> 不占 PPU lane——共用 pool 只做 conv/decoder descale + MHA-projection descale + BN + PReLU。

- **real multiply-equivalent MAC ≈ 1.40 M/frame**（複數 = 3 個 real mult、2 耳）
  ⇒ `128 × 80K/1.40M ≈ 7.3` ⇒ **8 個 PPU lane**。
- 若以文件的**normalized 1.10 M MAC/frame** 排，或不允許小層 throttle ⇒ ≈ 9.4 ⇒ **10–12 個**。

**建議：8 個共用 PPU lane（baseline）；保守抓 10–12。** 每個 lane 只是小的 shift-add
（2 shift + 1–2 add + sign-mux）；8–12 相對 128 PE 可忽略。（MAC basis 註：source 的
`apply_complex` 是 naïve 4-mult 複數；這裡的 MAC 數用 3-mult 複數等效以對齊 Headline 的
1.10 M /「MAC-equivalent」口徑。PPU **比例**、即 ~8 lane，對 ×3-vs-×4 的選擇是 robust 的。）

**不要照小層的瞬時 worst-case 配**（改 throttle PE 寬度或 FIFO-drain）：

| 小層 | reduction / scale | 滿 128 PE 的 naïve lane 數 | 為何不配 |
|---|---:|---:|---|
| enc0 pointwise | 1（`Cin=1`） | 128 | `Cout=8`，本來就塞不滿 PE array |
| depthwise 5-tap（per-ch） | 5 | ~26 | channel 少、DMA-bound |
| ConvTranspose / deconv | ≈ 2.5（stride-2 input share） | ~51 | decoder 小層；非瓶頸 |
| Score×V scale `2^-(8+k_v)` | — | **n/a（在 PE 內）** | L-Mul 指數路徑的指數 offset，非 PPU lane（§3.2a） |
| QK scale `2^-(k_q+k_k+2)` | — | **n/a（在 PE 內）** | L-Mul 指數路徑的指數 offset，非 PPU lane（§3.2） |

conv 小層（enc0-pw、depthwise、deconv）scale:MAC ratio 差但絕對 op 數很小、塞不滿 128 PE；
FIFO-drain，或把它們的 PE 寬度 cap 到 ≈ `PPU_lanes × reduction`。QK / Score×V 的 scale **完全不是**
PPU 工作——它們跑在 PE 內的 L-Mul 指數路徑。

---

## 附註：本文定位

本文是 **算術 / datapath 合約（arithmetic contract）**：明確規範每個 op 的數值格式、
各 scale 的 APoT/pow2 指數範圍、各模式的加法器位元寬、累加精度、buffer 大小、grouping、
即時 duty，以及 §8 逐層維度、§9 frame-by-frame 排程與資料載入計畫。

**尚未涵蓋**（屬完整 RTL microarchitecture，留給硬體團隊）：cycle-accurate control FSM、
pipeline timing、詳細 buffer address schedule、逐 cycle 的 rounding 順序。這些不影響上述
算術正確性，但實作 RTL 時需另行定義（§3.1 已標明 rounding 點「必須明定」）。
