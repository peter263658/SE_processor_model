# RTL FIX — FP16 32-way tree overflow:per-tensor pre-tree preshift

Date: 2026-06-07
對象:CE datapath 硬體顧問
範圍:`pe_pair` / `PEG` / `CE_core` / `dealer`(+ 2 個 TB)。**INT4 conv 路完全不動。**

---

## 0. TL;DR

CE 的 32-way **FP16 saturating tree** 對下面三類路會 **溢位(算錯,不是掉分)**:

| 路徑 | pre-descale tree partial | vs FP16 65504 |
|---|---|---|
| FP8 outlier decoder pointwise(stage 0/3/4) | 最高 **275108** | **4.2×** |
| QK^T(L-Mul m8) | 最高 **~2.1M** | **33×** |
| Score×V(top-4 packet) | 最高 **~374k** | **5.7×** |

INT4 conv pw / mha in-out proj、depthwise、out_proj(W4A16)**安全**(≤0.17×)。

**修法**:這三類的 descale 都是 **per-tensor 常數**(沿 contraction 軸不變),所以把其中的 **coarse pow2 部分搬到 tree 之前**(運算元指數位移)是**代數恆等**。在 `pe_pair` 加一個 per-PEG 的 `i_preshift`(signed 指數),折進「product 指數加法」即可——**零額外硬體**(指數加法器本來就在),`preshift=0` 時 bit-exact 等於舊版。`dealer` 改帶 **residual descale**(= 原 descale − preshift)。

> 為什麼不能只把 tree 內部加寬?因為 **單一一個乘積本身就 > 65504**(FP8 448×320=143360;QK 366×366=134284;ScoreV 255×366=93444)→ `pe_pair` 的 per-product FP16 輸出就先 `Inf` 了(L-Mul `expf≥31 → INF15`)。所以**一定要在運算元/乘積端 pre-scale**,加寬 accumulator 救不了。

---

## 1. 原理佐證

### 1.1 為什麼會爆(根因)

descale 全部延到 tree **之後**(`dealer`)。tree 累的是 **pre-descale** 乘積和。凡是「**兩個運算元都被放大進 e4m3 / q0.8 滿幅**」的路,pre-descale 乘積就爆:

- **INT4 conv(安全)**:只有 activation 單邊放大(raw e4m3 code = `e4m3(x·2^k)`,≤448);weight 是小整數 q∈[−7,7]。單乘積 ≤ 7×448 = **3136**,32-sum ≤ 11310(實測 golden)。FP16 撐得住。
- **FP8 outlier(爆)**:weight 也被放大(per-tensor `fp8_scale≈757` → e4m3 code ≤448),activation 也放大 → 單乘積 ≤ 448×448 = **200704**。雙邊放大。
- **QK(爆)**:Q、K **都**過同一個 `activation_quantizer`(k=2⁵,×32 → code ≤366,見部署碼 `complexLayers.py` `_mha_attention_forward` L2134-2137:`q=k=v=activation_quantizer(...)`)→ 乘積放大 ×1024 vs 真值。
- **Score×V(爆)**:prob 是 q0.8(`prob_scale=256` → code ≤255)× V e4m3(×32,≤366)。

### 1.2 為什麼前移合法(關鍵)

`true_out = Σ_ic descale · (w·x)`。若 `descale` **沿 contraction 軸 ic 是常數**(per-tensor / per-head 對 QK / per-tensor 對 FP8),可提到 Σ 外:`= descale · Σ_ic(w·x)`,等價於「先把 descale 套到每個乘積、再累加」。所以把 descale 的 **pow2 部分** 變成運算元/乘積的指數位移(pre-tree)與原本 post-tree 套用 **數值完全相同**。

- ✅ 可前移:**FP8 weight per-tensor descale、QK 的 Q/K 各自 2⁻ᵏ、Score×V 的 prob/256 與 V 2⁻ᵏ**。
- ❌ **不可**前移(雷②):**INT4 pointwise 的 per-GROUP(per-32-ic)descale**——每組不同,提不出 Σ。→ 這些路 `preshift=0`,維持原本 post-tree `dealer`(而且它們本來就不爆)。

### 1.3 證據(golden:`p225_p225_009_az-40_snr+0.0.wav`)

腳本:`scripts/verify_fp16_tree_headroom.py`(conv/proj,golden-exact;INT4 數字逐位重現舊 handoff §3)、`scripts/verify_fp16_tree_attention.py`(MHA 內層,calibration bound)。

```
conv  PRE  worst max|node| = 275108 (4.2x)   overflow nodes = 251543
MHA   QK 2.1M(33x) / ScoreV 374k(5.7x) / out_proj 2565(0.04x, safe)
```

---

## 2. RTL 改動(before / after,逐檔)

> ⚠️ 你的 RTL 是舊版,行文字可能略不同。**核心指令**:在 `pe_pair` 把一個 signed 指數 `i_preshift` **加到「product 的最終 biased 指數」**——四條都要加:INT4 路 + L-Mul 路 × lo + hi 兩個 slice;加在 underflow(≤0→0)/overflow(≥31→Inf)clamp **之前**。golden 參考模型 `pe_lane`(TB-only)**不要加**(它代表 preshift=0 基準)。

### 2.1 `pe_pair.sv` — 加 port

```systemverilog
// port list, 在 i_mode / 運算元之後、輸出之前加:
    input  logic signed [7:0] i_preshift,   // per-tensor PRE-tree pow2 descale.
                                            // =0 for per-GROUP INT4 (雷②);
                                            // !=0 only for per-TENSOR paths (FP8 pw, QK, Score×V).
                                            // preshift=0 => bit-exact to the old pe_pair.
```

### 2.2 `pe_pair.sv` — 四條 product 指數各加 preshift

每條 `*_expf` 是 `logic signed [31:0]`,在尾端加 `+ $signed({{24{i_preshift[7]}}, i_preshift})`(8-bit sign-extend 到 32-bit)。

```systemverilog
// (1) INT4 lo
// BEFORE:
    assign int4_expf_l = $signed({{24{expY_l[7]}}, expY_l}) + $signed({27'd0, Ll}) + 32'sd15
                         + $signed({31'd0, int4_rc_l});
// AFTER:
    assign int4_expf_l = $signed({{24{expY_l[7]}}, expY_l}) + $signed({27'd0, Ll}) + 32'sd15
                         + $signed({31'd0, int4_rc_l}) + $signed({{24{i_preshift[7]}}, i_preshift});

// (2) L-Mul lo
// BEFORE:
    assign lmul_expf_l = $signed({{24{exX_l[7]}}, exX_l}) + $signed({{24{exY_ll[7]}}, exY_ll})
                         + $signed({31'd0, carry_l}) + 32'sd15;
// AFTER:
    assign lmul_expf_l = $signed({{24{exX_l[7]}}, exX_l}) + $signed({{24{exY_ll[7]}}, exY_ll})
                         + $signed({31'd0, carry_l}) + 32'sd15 + $signed({{24{i_preshift[7]}}, i_preshift});

// (3) INT4 hi
// BEFORE:
    assign int4_expf_h = $signed({{24{yh8_e[7]}}, yh8_e}) + $signed({28'd0, Lh}) + 32'sd15;
// AFTER:
    assign int4_expf_h = $signed({{24{yh8_e[7]}}, yh8_e}) + $signed({28'd0, Lh}) + 32'sd15
                         + $signed({{24{i_preshift[7]}}, i_preshift});

// (4) L-Mul hi
// BEFORE:
    assign lmul_expf_h = $signed({{24{exX_h[7]}}, exX_h}) + $signed({{24{exY_lh[7]}}, exY_lh})
                         + $signed({31'd0, carry_h}) + 32'sd15;
// AFTER:
    assign lmul_expf_h = $signed({{24{exX_h[7]}}, exX_h}) + $signed({{24{exY_lh[7]}}, exY_lh})
                         + $signed({31'd0, carry_h}) + 32'sd15 + $signed({{24{i_preshift[7]}}, i_preshift});
```

零/NaN/Inf 特例與 clamp 都在 `*_expf` **比較之後**,不受影響。preshift 為負時把乘積往小推(目標路乘積很大,不會 underflow 到掉精度)。

### 2.3 `PEG.sv` — 加 port 並 pass 給 32 個 pe_pair

```systemverilog
// port list 加:
    input  logic signed [7:0]  i_preshift,   // 一個 PEG = 一個 contraction stream = 一個 per-tensor 值

// pe_pair 實例化加:
        .i_preshift (i_preshift),
```

### 2.4 `CE_core.sv` — per-PEG port + route

```systemverilog
// port list 加(per PEG):
    input  logic [NUM_PEG-1:0][7:0]  i_preshift,

// generate 內 PEG 實例化加:
        .i_preshift ($signed(i_preshift[g])),
```

### 2.5 `dealer.sv` — 無邏輯改動,只是語意

`i_apot` 現在帶 **residual descale = full_descale − preshift**。per-group INT4(preshift=0)時就是原本的 full per-group descale,完全不變。建議在 header 註明即可。

### 2.6 TB(保持 bit-exact)

```systemverilog
// tb_PEG_w4a8.sv:DUT 驅 0,REF(pe_lane×32)不動 => 仍 16000/0
    PEG u_dut (.i_mode(i_mode), .i_x(i_x), .i_y(i_y), .i_preshift(8'sd0), .o_accum(dut_accum));

// tb_ce_core_smoke.sv:宣告 logic [NUM_PEG-1:0][7:0] i_preshift; 全設 8'sd0;接到 CE_core
```

---

## 3. Controller / builder 要負責(RTL 只收 port)

每個 sub-op(per PEG-stream)給兩個值:`preshift` 與 `dealer` 的 **residual** code。

| 路徑 | preshift(folded in pe_pair) | dealer residual |
|---|---|---|
| INT4 conv pw / depthwise / out_proj | **0** | full per-group / per-channel descale(不變) |
| FP8 outlier pw(stage 0/3/4) | coarse pow2(本 golden 算出 **2⁻³ / 2⁻⁴**) | full_descale − preshift(APoT-6a 的 a 欄 −= preshift) |
| QK^T | **2⁻⁽ᵏ⁺ᵏ⁾ = 2⁻¹⁰**(k=5,Q 與 K 各 2⁻ᵏ) | 殘量;1/√d=2⁻² 仍走 score-scale(PPU SCALE) |
| Score×V | **2⁻⁽ˡᵒᵍ²ᵖʳᵒᵇ⁺ᵏ⁾ = 2⁻¹³**(prob/256 + V 2⁻ᵏ) | 1(全部前移,ctx 直接真值) |

選法(builder):`preshift = floor(log2(TARGET / max_pre_descale_partial))`(≤0),`TARGET=32768`(留 2× margin)。`max_pre_descale_partial` 由 calibration amax 上界即可(FP8 用 per-tensor descale + act k;QK/ScoreV 用 `activation_quantizer` 的 k=5 與 prob_scale)。residual:`a_exp_residual = a_exp_full − preshift`(preshift 負 → a 變大,確認仍在 APoT 編碼範圍)。

---

## 4. 驗證結果(兩支腳本已 mirror preshift 重跑)

**conv**(`verify_fp16_tree_headroom.py`):
```
PRE-fix worst max|node| = 275108 (4.2x)  ->  POST-fix worst = 31962 (0.49x)   overflow nodes = 0
FP8 outlier: preshift 2^-3 / 2^-4 -> 全部 <=0.49x, 0 溢位
INT4 / DW : preshift=0, 數字完全不變 (err/peak ~10 bit)
```

**MHA**(`verify_fp16_tree_attention.py`,calibration bound):
```
QK^T     preshift 2^-10  : 2148538 -> 2098  (0.03x)  ok   (單乘積 134284 -> 131)
Score×V  preshift 2^-13  :  373776 -> 45.6  (0.00x)  ok
out_proj  —              :    2565 -> 2565  (0.04x)  ok
```
→ **全部 sub-op fit FP16,0 overflow。**

---

## 5. 驗證(SV TB)

- `tb_PEG_w4a8`(`06_run_PEG_w4a8`):DUT vs pe_lane×32,DUT 驅 `preshift=0` → 仍 16000/0(preshift=0 結構上 bit-exact)。
- **`tb_peg_preshift`(`07_run_peg_preshift`,本次新增)**:驗 **preshift≠0**。原理 = scale-equivariance:`i_preshift=s` 只加到 product 指數 → `PEG(preshift=s) == ldexp(PEG(preshift=0), s)` 在 FP16 normal band 內逐位成立(L-Mul 模式也成立,因 mantissa 兩次算法相同、只差指數)。
  - A 確定性 anchor:全 lane product=1.0(acc0=32.0),掃 s∈[−12,+6],四 mode。
  - B 隨機:per-lane 正運算元(product∈[1,6],node 全程 in-band),四 mode、s 雙號,涵蓋 lo+hi slice 與 16-bit gating。
  - C 邊界:大 −preshift → product underflow → FTZ=0;大 +preshift → product=INF15 → saturating tree clamp=0x7BFF。
  - **已用 Python 模型預驗**(112000 cases / 0 mismatch)確認運算元編碼與 s 範圍 in-band、assert 可滿足;待 VCS 實機跑出 `[TB] ALL PASS`。

> ⚠️ 仍需在有 VCS license 的機器跑 `06` 與 `07` 各一次落地確認(本機無 license)。
- MHA 內層是 **calibration bound**(Q/K/V/scores/probs/ctx 沒 dump);要 golden-exact 需 instrument model forward 或 dump 中間張量。但 overflow 判定穩固(單乘積即溢位)。
- 1/√d=2⁻² 與任何 APoT 細部 residual 仍留在 score-scale / dealer 後段。
