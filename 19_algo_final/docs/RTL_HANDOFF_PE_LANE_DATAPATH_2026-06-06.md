# RTL_HANDOFF: PE-Lane Datapath 審查與雷點 — Conv/Attention MAC core

Date: 2026-06-06
Base candidate: `strict_core_pow2_blockfloat_deploy_ep3` (md5 `673de90a44dc13a4770cafc5a223d161`)
Golden input: `p225_p225_009_az-40_snr+0.0.wav` (noisy_testset/snr_0dB)
架構參考: [HANDOFF_HW_ZERO_MULTIPLIER_CORE.md](HANDOFF_HW_ZERO_MULTIPLIER_CORE.md) · pattern 索引 `../hw_patterns_final/tb_pattern/manifest.json` + `layer_config.json`

對象:要寫 PE-lane 內部邏輯的硬體顧問。
目的:lane 本體(純組合、4 mode、輸出 FP16)沒問題,可以開工;但 **group / post-acc 層有 5 個踩了會是 bug 或重做的雷**,先看這份。整體 mode 0/1/2/3 ↔ 四條 datapath、dealer/OACC/FMA 兩層 scale 的骨架方向正確。

---

## 0. TL;DR — 五個不能踩的雷

| # | 雷點 | 踩了的後果 | 必須怎麼做 |
|---|---|---|---|
| ⭐⭐ ① | **FP16 tree headroom**:descale 是 post-acc,32-way tree 累的是 pre-descale 和,被放大 2⁶~2¹⁶ | 某 partial > 65504 → Inf → 該輸出整條壞;MBSTOI 抓不到偶發 Inf | tree 內部加法位寬 **> FP16**(fp32/定點),dealer 後才 round 回 FP16;或至少 saturating add(clamp ±65504)別讓它變 Inf |
| ② | **pointwise descale 延到 OACC 之後** | **算錯**(不是誤差):per-group scale 各組不同,無法提到跨組和之外 | dealer 必須在 32-tree 與 OACC **之間**(逐組先套再跨組累加) |
| ③ | **把 conv 的 `BN+PReLU+e4m3 cast` 寫死在所有 post-path** | attention 三條路全錯;QK 溫度錯 | post-acc 要 **mode select**:conv-FMA / QK(descale+1/√d→softmax)/ raw-fp16-forward |
| ④ | **e4m3 輸出 cast 的 rounding 不對 golden** | raw code bit 不一致,乘積就不再 bit-exact | cast 用「下一層」的 k(input-boundary 指數),mantissa rounding 對齊 `fp8_qat` 實作 |
| ⑤ | **decoder split-pointwise 假設一個 weight 一個 descale** | stage 3 real/imag 的 k 不同,套錯半條 | dealer 依「餵進來的 activation slice 是 real 還 imag」選 `*_fused_act_kreal` / `kimag` |

實證結論(§3):golden 這筆輸入 **兩種情境都沒爆**(延後 peak 18339、per-group peak 11310,vs FP16 65504),但 **一個 32-group 的理論最壞值 = 32×(7×448) = 100352 > 65504**,所以 ① 的保險還是要做。

---

## 1. 受審架構(以 `layer_config.json` 為準)

| 項目 | 值 |
|---|---|
| lane 4 模式 | 0 = W4A8 / 1 = FP8 L-Mul m4 / 2 = FP16 L-Mul m8 / 3 = W4A16 |
| 陣列 | 32 lane/group × 4 group;8-bit operand(mode 0,1)→128 單 lane;含 16-bit operand(mode 2,3)→64 配對 |
| INT4 weight | `int4_symmetric_qmax7` → q ∈ **[−7, 7]**(−8 不用) |
| activation | block-float e4m3:raw code = `e4m3(x · 2^scale_pow2_exp)`,**無 per-activation runtime descale** |
| descale 融合 | `final_post_acc_descale_exp = weight_APoT_exp − scale_pow2_exp`,純指數 offset(shift-add,無乘法);pointwise per-group APoT 2-term,exp ∈ **[−16, −6]** |
| conv pointwise scale | positive APoT `2^a ± 2^b`,**per 32 input channels**(且 per output channel) |
| conv depthwise scale | positive APoT,per output channel |
| BN | folded affine `α = γ/√(var+eps)`,signed APoT **[−9, 4]**,conv bias 融入 BN bias |
| PReLU | signed pow2 slope **[−8, 0]**,real/imag 各自純量 |
| MHA | embed_dim **256**、head_dim **16**、num_heads **16**;tied;proj weight int4 qmax7;proj scale positive APoT [−16,−1] per head |
| QK | `lmul_m8`;`qk_scale` = pow2 `2^-2`(= 1/√16,**未** fuse 進 Q) |
| Score×V | `top4_packet_lmul_m8`;softmax `shift_topk_pow2_clamp8_nomantissa`;prob_bits 8 / top_k 4 / prob_scale 256 |
| out_proj | **W4A16**:weight int4 APoT,輸入是 Score×V 的 fp16 累加器,**無 activation k** |

L-Mul 常數(你方已驗 bit-exact):m4 = 18/32/16/fshl6、m8 = 272/512/256/fshl2 — lane 內部照此,本文不再覆述。

---

## 2. 三個 blocking 點(詳述)

### ① delayed scaling 的真正風險:32-way tree 的 FP16 headroom(最優先)

descale 是 **post-accumulation** 的 2-term APoT,指數到 **2⁻¹⁶**。所以 32-way tree 累的是「pre-descale」乘積和,**相對真實輸出被放大 2⁶~2¹⁶ 倍**;dealer 之後才縮回來 → **爆點在 tree(dealer 之前),不是 OACC**。

- 單一乘積在 FP16 是 **bit-exact**:W4A8 的 `q × decode(e4m3)`,significand ≤ 7 bit,FP16 的 11-bit mantissa 放得下 → **這不是精度問題,是 range/overflow**。
- FP16 max = 65504 ≈ 2¹⁵·⁹⁹。最兇的 2⁻¹⁶ 層,只要該 group 真實貢獻 ~O(1),pre-descale 和就頂到天花板。一旦某 partial 超過 → Inf → 該輸出整條壞,**斷崖,不是漸進掉分**。
- 「FP16 tree 實測不掉分」是看 **MBSTOI**,但 MBSTOI 抓不到偶發 Inf(整體平均不動)。**必須用 worst-case |partial sum| 驗,不能只看分數**(實證見 §3)。

**建議**:tree 內部加法位寬做成 **> FP16**(fp32 或定點 partial),只在 dealer 處(descale 後)round 回 FP16;pre-descale 的 2¹⁶ 膨脹由更寬 internal carry 吸收,dealer 後數值已小、FP16 綽綽有餘。成本低,是便宜保險。退一步至少做 **saturating add**(clamp 到 ±65504)而非讓它溢出成 Inf。

### ② descale 順序被「強制」成先套再累加(per-group 在 OACC 前)

pointwise descale 是 per-32-input-channel,**每組不同**:

```
true_out[oc] = Σ_group [ descale[oc,g] × Σ_{ic∈g} q[oc,ic]·x[ic] ]   # descale[oc,g] 各組不同
```

不同的 `descale[oc,g]` 無法從跨 group 的總和裡提出來 → 順序鎖死:**32-tree(組內)→ dealer(逐組 descale)→ OACC(跨組)**。dealer 擺在 tree 與 OACC 之間是對的,**千萬不要為了省 dealer 移到 OACC 之後 → 對 pointwise 是算錯**。

- depthwise(per-out-ch)/ MHA proj(per-head)/ QK(2⁻ᵏ):scale 在 contraction 軸上是常數 → 先套後套代數等價,同位置照用。
- 要跟 golden 對齊的硬條件:HW「32 lane 一組」要對到 golden 裡「共用同一個 descale code 的那 32 個 input channel」。**group 邊界錯位 = 算錯**(§5 probe-2 比對)。

### ③ FMA 是 conv 專用,attention 走別的 post-path

`post-MAC FMA (BN per-ch + PReLU pow2 + FP8 cast)` 只對 conv/pointwise 成立。三條 attention 路:

| mode | post-acc 要做什麼 | 不能做 |
|---|---|---|
| 2 QK | `× 2^-2`(1/√d)+ `2^-k` descale 套在 QK 累加器上 → softmax | 無 BN/PReLU/e4m3 cast |
| 2 Score×V | 輸出 raw fp16,直接餵 out_proj | 不 cast e4m3 |
| 3 out_proj (W4A16) | 只有 weight APoT descale、**無 activation k** → 輸出 fp16 | 無 activation 指數 |

manifest 硬約束 `qk_descale_placement`:**QK 的 `2^-k` 必須在 softmax 之前**套在 QK 累加器(它設定 logit 溫度),不能延到 score×V / attention output。

→ dealer/post-acc 要有 **mode select**:conv-FMA / QK(descale+√d→softmax)/ raw-fp16-forward。別把 `BN+PReLU+cast` 寫死。

---

## 3. 實證:FP16 headroom(golden 實測,已對 golden 驗證)

把 golden 那筆輸入的**所有 W4A8 conv pointwise** 層的 MAC,重建出「descale 延後」會出現在 FP16 累加器的 pre-descale 數值,比較兩情境峰值。

| 情境 | FP16 累加器峰值 | @ 層 | vs 65504 | 結果 |
|---|---|---|---|---|
| **延後**(整條 input 累加完才 descale) | **18339** | encoder.model.5.0.pointwise_r | 3.6× headroom | 不爆 |
| **逐 group**(現設計) | **11310** | encoder.model.2.0.pointwise_i | 5.8× headroom | 不爆 |
| 理論最壞(單一 32-group) | **100352** = 32×(7×448) | — | **> 65504** | **會爆** |

代表性逐層(channel-major,單位 = FP16 累加器值):

| 層 | [oc,ic] | max\|FULL-accum\| | max\|32-grp tree\| |
|---|---|---|---|
| encoder.model.5.0.pointwise_r | [128,128] | 18338.8 | 8421.0 |
| encoder.model.5.0.pointwise_i | [128,128] | 16481.9 | 7485.0 |
| encoder.model.4.0.pointwise_r | [128,64] | 15556.1 | 11208.5 |
| encoder.model.4.0.pointwise_i | [128,64] | 13777.8 | 10381.0 |
| encoder.model.2.0.pointwise_i | [32,16] | 11310.0 | 11310.0 |

讀法:
- 深層(128ch)延後版約是 per-group 版的 ~2.2×(4 組疊加);淺層(≤32ch,單組)兩者相等。
- descale 後的 OACC / 真實輸出 ~O(1)(golden range `[−4.86, 3.58]`),FP16 完全沒壓力 → **風險只在 dealer 前的 tree**。
- **3.6× / 5.8× 是單一 golden 輸入,不是部署保證**;calibration 只鎖 e4m3 單點(±448),鎖不住「和」。理論最壞 100352 > FP16 → ① 的保險要做。

**沒測到的 out_proj**:mode 3 W4A16,累加深度 256(全模型最深),但輸入是 fp16 attention 輸出(無 2⁻ᵏ 放大),per-element 小很多;不在 e4m3 pattern 內,故未量到。其輸入在 `debug_reference/layer_io_fp32` 為 fp32,可另外驗(§5 沒列入,需要再跟我說)。

**驗證方法(可信度)**:
- e4m3 解碼器經 self-test(`0x38→1.0, 0x40→2.0, 0x30→0.5, 0x7E→448`)。
- INT4 取自 `export_reference_debug/weights/*/w_int4.raw_s8.bin`(一 byte 一 code,免拆 nibble),全部驗證 ∈ [−7,7]。
- 對 golden 輸出比對:重建 `O_true = 2^k · Σ_g scale[oc,g] · T_g`,channel-major relL2 ≈ **9%** vs position-major ≈ 130% → **layout 確定 channel-major**;9% = 這層 W4A8 對 fp32 reference 的量化誤差(`layer_io_fp32` 是 fp32 參考,非量化路徑),非重建 bug。
- pre-descale tree 值本身是 INT4 × 精確 e4m3 的整數運算 → **數值精確**。
- per-group scale 實測確實逐組不同,例:`encoder.model.5.0.pointwise_r` row0 = `[0.046875, 0.02734375, 0.02734375, 0.0703125]`(= `2⁻⁵+2⁻⁶`, `2⁻⁵−2⁻⁸`, `2⁻⁵−2⁻⁸`, `2⁻⁴+2⁻⁷`,都是 APoT 2-term)→ 佐證 ②。

---

## 4. FMA 細節確認點

1. **它不是通用 FMA**:α(BN per-out-ch)= APoT signed [−9,4] shift-add、β = folded bias 加法、PReLU = pow2 shift、e4m3 cast。**全程 shift-add,符合 zero-multiplier 核**。命名叫 FMA 但裡面**不要放真乘法器**。
2. **順序**:`dealer(input-group descale) → OACC(收完 contraction) → α·(·)+β → PReLU(real/imag 各自,pow2) → e4m3 cast`。BN/PReLU 一定在**完整累加之後**(per-output-channel)。dealer + α = 你說的「兩層 scale」,對。
3. **e4m3 cast 用「下一層」的 k**(input-boundary 指數),且 cast 必須 **bit-exact 對 golden**:指數是 `floor`(已記載),mantissa rounding mode 去對 `fp8_qat` 實作(RNE?)。這是 FMA 的 per-output config 輸入(雷 ④)。
4. **複數 kreal/kimag**:decoder split-pointwise 在 **stage 3 real = 2⁻⁵、imag = 2⁻⁶ 不同**(`*_fused_act_kreal` vs `kimag`)。dealer 依餵進來的 slice 選 variant,operand 路由在 group 層,**descale variant 選擇也要在那層一起決定**(雷 ⑤)。

**小事(timing)**:LZD-hierarchical(`L = high.any ? 8+high.pos : low.pos`)+ 配對 lane 的 carry chain 是 16-bit 模式的組合關鍵路徑(LZD 跨 16 bit + 進位),timing 留意。累加用容差、golden 全程 fp32 → 符合 ②③ 的等價性論證,OK。

---

## 5. 收尾 OPEN 項要抓的 golden probe

1. 每層 **max \|32-group pre-descale partial sum\|** 與 OACC running max(關 ①;§3 已對 conv pointwise 做完,缺 out_proj / qkv in_proj / depthwise)。
2. 每層 descale 的 **group 邊界 + code ↔ HW lane-group 對映**(關 ②;group 邊界錯位 = 算錯)。
3. 每個 decoder split-pointwise stage 的 **kreal / kimag**(關 ④/⑤)。
4. **e4m3 輸出 cast 的 rounding mode**(關 ④,要 bit-exact)。
5. QK 確認 **`2^-2`(1/√d)與 `2^-k` 都在 softmax 前**套在累加器(關 ③)。

---

## 6. bit-exact 參考(Python)+ edge cases

RTL 要對得上的就是這幾段(取自 §3 的重建腳本)。

```python
# e4m3 解碼 (OCP E4M3: 1-4-3, bias 7, max 448, NaN @ 0x7F/0xFF)；HW decode table 要逐 code 對齊
def e4m3_decode(c):           # c: uint8
    s = (c >> 7) & 1; e = (c >> 3) & 0xF; m = c & 7
    sg = -1.0 if s else 1.0
    if e == 0:        return sg * (2.0**-6) * (m/8.0)      # subnormal
    if e == 0xF and m == 7:  return float('nan')           # NaN (S.1111.111)
    return sg * (2.0**(e-7)) * (1.0 + m/8.0)               # normal; e=0xF,m=6 -> 448

# 一個 lane (mode 0 W4A8)：純整數/精確,FP16 放得下 (significand <= 7 bit)
product = q * e4m3_decode(x_code)        # q in [-7,7]

# group(32 lane) -> dealer -> OACC ；descale 必須逐組、在跨組累加之前
def conv_pointwise_out(oc, pos, Q, Xcode, scale_apot, k):   # Q[oc,ic], Xcode[ic,pos]
    acc = 0.0
    for g in range(0, IC, 32):                              # 4 組 (IC=128 時)
        T_g = sum(Q[oc][ic] * e4m3_decode(Xcode[ic][pos]) for ic in range(g, g+32))  # <-- FP16 風險在這
        acc += scale_apot[oc][g//32] * T_g                 # dealer: APoT 2^a±2^b, exp in [-16,-6]
    return (2.0**k) * acc                                   # k = descale_pow2_exp (此層)
# 注意:scale_apot[oc][g] 各組不同 -> 不能把 (2**k * Σ_g) 改成 (Σ_g 先全加再一次 descale)
```

Edge cases(RTL 最容易錯的):
- **FP16 tree 溢位**:pre-descale 和理論最壞 100352 > 65504 → internal 用 >FP16 或 saturating;**不要**讓它變 Inf。
- **INT4 範圍**:q ∈ [−7,7](symmetric qmax7),**−8 不出現**;sign 在 raw_s8 已是 two's-complement int8。
- **APoT 2-term sign**:`2^a ± 2^b` 的 ± 來自 `scale_apot_residual_sign`;`scale_apot_has_second` = 0 時退化成單 term `2^a`。
- **e4m3**:NaN 在 `0x7F/0xFF`;max 448 = `0x7E`;subnormal(e=0)要算;golden roundtrip max = 0(編碼無損,decode 要剛好對上)。
- **group size = 32**(input channel 軸),自然順序;**packet top_k = 4 / prob_bits = 8 / prob_scale = 256**(softmax 路)。
- **scale-application order**:dealer(input-group)在 OACC 前;α(BN, output-channel)在 OACC 後 —— 兩層不可互換、不可合併(不同軸)。

---

## 7. 範圍:lane 本體不受影響

①~⑤ 全在 **group / post-acc 層**。lane 本體(純組合、4 mode、直接輸出 FP16、無 scale port、`pair_en` 由 `i_mode` derive)維持你定的乾淨版即可。建議在 **group 介面**預留:① tree internal 位寬 > FP16(或 saturating)、③ post-acc mode select、⑤ kreal/kimag descale variant 選擇 三個 hook。

lane I/O 骨架可在此基礎上開寫。
