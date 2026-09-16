# Consultant ask — skip dataflow + 整 frame 編排查核清單 (2026-06-10)

Context: RTL 端開始接 skipbuf dataflow（encoder skip-write → SKIPBUF → decoder pw_skip read，
+ STFT spectrum 暫存）與 engine 整 frame 編排。以下分「已資料驗證、請 rubber-stamp」與「不確定、
請查核」。資料根據 `hw_patterns_final/tb_pattern/`（activations_fp8 + skip_indices + layer_config）。

---

## A. 已資料驗證 — 請 rubber-stamp

**A1. skip 來源映射：`dec_s.pw_skip 輸入 = enc(5−s) 階段輸出`，s=1..5 為純 code copy（同 k）。**
驗證：`decoder.model.s.0.pointwise_{r,i}_skip__N` 的 codes == `encoder.model.(6−s).0.depthwise_{r,i}__N`
（= enc(5−s) 輸出的 dump）依 `skip_indices/decoder_stage_s.u16.bin` 檔案順序選 channel —
**51072/51072 全等**（s=1,2,5 × r/i × chunk 0/1 抽驗；k 相同：例 dec1 skip k=−6 == enc5 dw input k=−6）。
⇒ HW skip-write = 寫入時依 skip_indices 選 channel、原 code 直存（compact、檔案順序 = 存放順序）。
請確認 s=3,4 同樣成立（我們可自驗，列出僅供完整性）。

**A2. dec0 skip = enc5 輸出（= mattn input_proj 輸入 act）「重新量化」k: −7(re)/−6(im) → +1。**
驗證：對 t*=351，`mattn_input_proj__0/act_{re,im}_t0.hex`（k=−7/−6）解碼→ ÷2^1 → e4m3 RNE 重量化
== `decoder.model.0.0.pointwise_{r,i}_skip` @t=351 → **62~63/64 match**（殘差 1~3 = 「二次量化」
邊界：deploy 從 fp32 單次量化到 k=+1，我們從已量化的 e4m3@−7 codes 再量化）。
請確認此理解正確（dec0 skip 不是另一個 tensor）。

**A3. mattn 輸入 256 維 = 雙耳 concat：dim 0..127 = 耳A、128..255 = 耳B。**
驗證：dec0 skip chunk__0 對上 act dim 0..127、chunk__2 對上 dim 128..255。
請確認 concat 順序（哪耳在前；L=0..127?）。

---

## B. 不確定 — 請查核（按優先序）

**B1.（影響 skip-write RTL 設計）dec0 skip 的量化策略，HW 應採哪種？**
- (a) **pre-cast fp16 雙投**（建議）：enc5 pw 的 PPU 在 e4m3 cast 前的 fp16 值，分別 cast 到
  k=−7/−6（→UF 給 attention）與 k=+1（→SKIPBUF 給 dec0）。每個 consumer 各自單次量化，
  最接近 deploy（殘差只剩 fp16-tree vs fp32 的既有 ±1 boundary 類）。
- (b) codes 再量化：存好的 e4m3@−7 codes → exponent−8 + RNE/subnormal → e4m3@+1（二次量化，
  與 deploy 差 ~1-3/64 boundary codes，會傳進 dec0 pw_skip 的 MAC）。
- 另：為何 dec0 skip 校正出 k=+1（range ±896）而同 tensor 在 mattn 端是 k=−7（±3.5）？
  amax 校正依 consumer 不同我們理解，但差 2^8 想確認不是 dump/exporter bug。
- 若選 (a)，是否可由你們重生 dec0 pw_skip 的 per-layer golden（以 fp16-雙投後的輸入 codes 為準）？
  或我們自己用 gen_golden_conv.py 以重量化後 act 生成 HW bit-exact target（你們 cross-check）。

**B2. activations_fp8 的 `__N`（call_index 0..3）語意？**
chunk0 ↔ 耳A、chunk2 ↔ 耳B 已從資料看出。chunk 1、3 是什麼（第二段 wav？第二個 batch item？）。
哪個 chunk 對應 mha_module_io / qk golden 的 t*=351 capture？（我們的 attention chain golden 都基於
那份 capture；skip/conv golden 用 `__0`，需確認同源。）

**B3.（engine 編排）整 frame 的 freq-position 迴圈契約。**
layer_config: 所有 conv kernel=[5,1]、pad=[2,0]、dw stride=[2,1] ⇒ kernel 只在 freq 軸 ✓
（跨 frame 依賴只在 attention ✓）。現在 ROM 一個 sub-op = 一個 freq position（oc=channels）。
整 frame = 每層跑 F_out 個 position（enc: 32,16,8,4,2,1；dec 鏡像 1,2,4,8,16,32）。請確認：
- (i) encoder dw forward stride-2：output position p 的 tap 視窗 = input freq [2p−2 .. 2p+2]，
  邊緣 zero-pad（pad=2）。
- (ii) **decoder dw = ConvTranspose2d stride-2** 的 HW 迭代契約：output position q 對應的
  tap/input 對（zero-insert 等效：q 偶/奇用不同 tap 子集）。可否提供一層 transposed-dw 的
  per-position 黃金 I/O dump（如 dec1 dw：input [64,2] → output [64,4]，含每個 output position
  的 tap·input 對應），讓我們驗 dw 定址？**decoder dw 我們還沒有 golden 驗過**，這份 dump 是
  下一步 dw bring-up 的關鍵。
- (iii) encoder dw（forward）也請附一份 per-position golden（如 enc1 dw [8,32]→[8,16]）。

**B4.（engine 邊界）attention ↔ conv 的 reshape/分耳契約。**
- enc5 輸出（每耳 [128, F=1]）→ MHA 輸入 [256]：除 A3 的耳序外，是否有 freq 折疊（F=1 所以應
  只是 ch 直排）？
- attention 輸出 [256] → dec0 dw 輸入（每耳 [128,1]）：同 A3 的耳序拆半？
- decoder 每耳各跑一次（權重共享、時間多工）— 確認 decoder 兩耳之間沒有交叉項。

**B5.（T3 warm-up 驗證資料）早期 frame（t* < 32，window 未滿）的 attention golden dump。**
我們要驗 cold-start：前 31 個 frame 的 KV ring 未滿，需要 causal/window mask（現在 RTL 的
i_SM_mask 還是 all-ones，將改成 per-frame validity）。請 dump 一個小 t*（例 t*=5 或 10）的
qstar window / prescale/postscale scores / softmax top-4 packet / ctx（格式同 mha_module_io），
讓 warm-up mask 的 Softmax/top-4 行為（<32 valid、甚至 <4 valid 時 top-k 與 pow2 分母）可以
bit-exact 驗證。另確認：deploy 在 t<31 的 mask = triu/-inf + dist>window −inf（model.py 551-562）
之外沒有別的 special-case。

**B6.（低優先）SKIPBUF 容量布局確認。**
我們的 per-frame 預算：skip（雙耳、re+im、含 freq 維）Σ count_s×F_s = 704 B/slice/耳 → 2816 B
+ STFT spec（Q2.14 int16, 64 bins, re+im, 雙耳）512 B = 3328 B ≤ SKIPBUF 4 KiB ✓（zero slack 880 B）。
count_s×F_s：s0 64×1、s1 64×2、s2 32×4、s3 16×8、s4 8×16、s5 4×32。請確認 skip 在 decoder 消費
時的 freq 維 layout（我們計畫 channel-major per position：position f 的 [count_s] 連續，與
activations_fp8 dump 的 [C, F, T] C-major 一致 — 即同一 channel 的 F 個 position 連續？dump 是
[ci*n + f*T + t]，n=F*T ⇒ channel-major、channel 內 F-major。讀側 per-position 取 stride。確認
pw 在 position f 的輸入 = skip[:, f] 無誤）。

---

## 附：我們接下來不被 block 的部分（已動工）
- skip-READ path（decoder pw_skip 從 SKIPBUF 讀，golden act 預載驗 bit-exact）— 不依賴 B1。
- skip-WRITE s=1..5（純 code copy，A1 已驗）。
- STFT spec 區（Q2.14 存取 + mask_apply 餵入，用 mask_apply_golden 驗）。
- dec0 skip-write 先照 B1(a) pre-cast fp16 雙投實作，等你們確認後驗收。
