# 論文 Feature 總整理 — BCCTN 雙耳語音增強 × 零乘法器加速器（2026-07-07）

> 目的：寫論文前的完整 feature 盤點。每個步驟列出「做了什麼」與「換到什麼好處（數字）」。
> 資料來源：`19_algo_final/docs/RESULTS_ARCH_PROGRESSION.md`（架構演進權威版）、
> `ARCHITECTURE_SUMMARY_STRICT18.md`、`HANDOFF_HW_ZERO_MULTIPLIER_CORE.md`、
> `20_pesq_finetune/docs/STATUS_CHIP_PAPER_COMPARISON_POSITIONING_2026-06-28.md`、
> `演算法端_品質對齊_SPEC.md`，以及各 `NN_*` 目錄的 README / RESULTS / HANDOFF 文件。
> 英文版：`PAPER_FEATURE_SUMMARY_2026-07-07.md`。

---

## 0. 一句話定位（headline claim）

> **第一顆「雙耳、保留空間線索」的語音增強加速器**：在標準品質指標上追平/超越最強單聲道 SE 晶片
> （STOI **96.85 > 95.1，CICC'26-北大**），同時獨家保留單聲道設計做不到的 ILD/IPD 空間線索
> （MBSTOI **0.836**），並以**純數位、零乘法器的 W4A8 神經核心**實現
> （Conv/MHA/BN/PReLU/softmax 全部 shift/add/mux，無通用乘法器、無除法器、無 exp LUT），
> **全模型權重常駐單顆 256 KiB WBUF**（無 runtime DRAM 權重搬運）。
>
> 競品一律用**期刊-學校**標示（JSSC'20-台大 / CICC'26-北大 / A-SSCC'25-清華 / JSSC'24-首爾大；
> 2026-07-07 決議廢除 P 編號、剔除 MIT VLSID'25）。

---

## 1. 全局演進總表

| 階段 | 權重基準 | MAC/frame（乘法等效） | 核心乘法器 | VCTK / TIMIT MBSTOI |
|---|---:|---:|---|---|
| 原始 BCCTN（fft 512、embed 512） | 3,973,288 | ≈29 M（自身 basis） | 有 | 0.8519 / 0.8114 |
| 硬體尺寸 FP8/QAT baseline | 830,300 | ~1.67 M | 有 | 0.8399 / 0.7966 |
| 14 composite（production） | 337,876（RTL） | 1.10 M | 有 | 0.8354 / 0.7880 |
| 18 strict core（exact descale） | 337,876（INT4） | 1.10 M | **零** | 0.8355 / 0.7902 |
| 18 deploy（delayed APoT、動態 act） | 337,876（INT4） | 1.10 M | **零** | 0.8365 / 0.7912 |
| **★ 18 deploy（pow2 block-float act）** | 337,876（INT4） | 1.10 M | **零** + **無 runtime act scaling** | **0.8357 / 0.7898** |

- 參數（儲存權重基準）：**3,973,288 → 337,876（−91.5%）**，還沒算 INT4 位元組減半。
- 計算量：**≈29 M → 1.10 M 乘法等效/frame（約 20–27×，basis 不同須註明）**。
- 品質成本 vs 原始 full-size：約 −0.016 VCTK / −0.022 TIMIT MBSTOI，換到整個核心零乘法器。
- Deploy gate：VCTK ≥ 0.835（deploy 0.8357）、TIMIT ≥ 0.788（deploy 0.7898）——**兩者皆過**。
- Deploy ckpt：`18_w4a8_pow2_mixed_lmul_hw_handoff_20260530/ckpt/strict_core_pow2_blockfloat_deploy_ep3_vctk0.8357_timit0.7898.ckpt`

---

## 2. Phase 0 — 演算法基線與架構縮小（BCCTN → hardware-sized）

| Feature | 做了什麼 | 好處 |
|---|---|---|
| **BCCTN 基線**（ICASSP'24 血統） | 複數 encoder–decoder（6 級）+ 複數 MHA 瓶頸 + 每耳複數比值遮罩；loss = SNR + 10·STOI + ILD + 10·IPD；L/R 共用權重 | 雙耳增強同時**保留 ILD/IPD 空間線索**——整個計畫的 USP 起點 |
| **Depthwise-separable 複數卷積** | 全卷積 `(5,2)` → depthwise + pointwise `(5,1)` | 卷積參數/MAC 大幅下降（卷積佔 >80% FLOPs），是後續所有量化工作的架構基礎 |
| **因果化 + 滑動視窗注意力**（attn_win=31） | 因果 padding；注意力只看 31 frame 視窗 | 可低延遲串流；S×S softmax 成本受控；KV 歷史有限 → KVBUF 可以小 |
| **架構縮小** | `fft_len 512→128`、`win 400/100→128/80`、`embed_dim 512→256`、`heads 32→16` | 參數 **3.97 M → 0.83 M（−79.1%）**；品質僅 −0.012 VCTK / −0.015 TIMIT；模型開始塞得進硬體預算 |

---

## 3. Phase 1 — 硬體感知分析（01–06，不重訓、純 profiling）

| # | 做了什麼 | 關鍵數字 | 換到的決策（好處） |
|---|---|---|---|
| **01 利用率/瓶頸分析** | 256-MAC weight-stationary 陣列上逐層算 MAC/DMA cycle 與利用率。⚠️ 原始 run 是 embed-512/32-head 舊變體；2026-07-07 已用 deploy 尺寸重跑（見驗證報告） | Deploy basis（E=256/16）：total **0.823 G MACs/s**，attention 佔 **54.0%** MAC（decoder 30.8%、encoder 15.2%）；歷史數字 **77.6%** 是 E=512 變體。Encoder depthwise 利用率 13–17%（decoder DW-transpose 約 29%）；W4A8 幾乎不改平均利用率（45.1%→45.1%） | **W4A8 的價值在 SRAM/DMA 不在利用率**；主攻方向定為 attention 權重量/搬運——兩個 basis 結論相同 |
| **01-8kHz 取樣率實驗** | 16 kHz vs 8 kHz-升頻輸入對照 | 8 kHz 後 **ILD/IPD 比 noisy 更差**（ILD 4.82→5.98 dB） | **不能降到 8 kHz**：空間線索保真依賴全頻寬 → 鎖定 16 kHz |
| **02 權重位元寬掃描** | Weight-only PTQ：INT8/INT4/INT3/FP4 逐層+全網。✅ **2026-07-07 已用 n=500/seed 42 重跑、artifact 保留**（`results_rerun_2026-07-07/`）——論文引用這組數字 | Uniform INT4：SI-SNR **−0.38 dB**（低於 0.5 dB 門檻）、ILD +0.20 dB、IPD +0.56°、**DMA −50%**；INT3 崩潰（**−4.36 dB**）；單層 W3 Δ 各僅 ~0.0x dB 但 mixed 合併 −2.31 dB（省 61.5%）；tier **29 C + 1 B 精確重現** | 選定 **W4A8**；證明「逐層分析系統性樂觀、誤差跨層累積」→ 拒絕 W3/混 W3；改用 per-group/per-head scale |
| **03 動態範圍分析** | 104 個權重張量的 per-tensor/channel/group/head 統計 | **5 個 decoder pointwise（stage 0/3/4）kurtosis>5 → 留 FP8**（已驗證）；real/imag 可共用 scale（已驗證）；MHA head-group 不均勻 → per-head：**deploy 對齊的 16-head 切分下 8/12 矩陣不均勻**（2026-07-07 重切，`results_regen_2026-07-07_h16/`；報告的 10/12 是在 16-head 模型上做 32 路切分）。✅「42/42 conv 需 per-channel/group」已由 2026-07-07 修復版 analyzer 重生補回支撐（`results_regen_2026-07-07/`，最壞 var_max 0.526/0.494；group-size 平均精確重現報告表格） | 直接產出量化配方：PW per-group、DW per-channel、MHA per-head、outlier 層留 FP8、re/im 共 scale（scale 開銷 <1%） |
| **04 稀疏性分析** | 權重/激活/注意力分數多門檻稀疏統計 | 權重稀疏是隨機的（無結構，43/52 層）；16 頭全有貢獻；W=31 視窗內無恆零；PReLU 激活稀疏 ~19–62% 但是動態的（註：報告「6 dense = Q/K/V/O」是誤標——5 conv + 1 MHA） | **負面結果但省下硬體成本**：不做 zero-skip/剪枝硬體，資源集中在 W4A8 與 attention |
| **05 卷積重用/DMA 分解** | 跨層權重相似度、SVD 低秩、時間重用、DMA 佔比 | 卷積間無可共用權重；conv kernel_t=1 → 無跨 frame 重用；**MHA 權重 DMA 640 KB ≈ 全部 PW conv 的 4 倍** | 確認 **MHA 是 DMA 瓶頸**：優先 W4 量化 MHA（4×），DW 用 freq-row sliding buffer |
| **06 注意力混合精度** | 各子運算動態範圍 + 注入 fake-quant + top-k 全模型評估。✅ **2026-07-07 已用 n=20 重跑**補掉稽核發現的 n=2 缺口（n=2 舊結果保存在 `results_n2_2026-04-16/`） | **QK^T \|max\| = 1049 > e4m3 448**（n=2 probe 是 911——結論更強）→ 需 FP16 累加器；**12/16 尖銳 head 在 n=20 下確認**；top-4 的機率列 L1 誤差很低 | **top-4 稀疏注意力**是 06 唯一存活到 deploy 的建議（品質背書引 07 的 0.8340 / 14 的 0.8360）；「int16 accumulator / 全 fp16」表已被 13/14/18（FP16 累加器 + W4A8 + L-Mul m8）取代 |

---

## 4. Phase 2 — 量化 + 結構壓縮（07–14）

品質基準：VCTK noisy MBSTOI 0.7861；FP8/QAT baseline 0.8399。

| # | Feature | 做了什麼 | 好處（數字） |
|---|---|---|---|
| **07/08** | **W4A8 混合精度 QAT 配方** | INT4 權重（PW per-group-32、DW per-channel、MHA per-head）+ FP8 e4m3 激活 + fp16 累加器 + top-4；decoder PW outlier（stage 0/3/4）留 FP8 | MBSTOI 0.8340（gsz32+topk4）；權重記憶體 vs 512-dim FP8 **−82% DMA**（2.80 MB→487 KB，~5.75×）；fp16 累加與 top-4 證明「免費」（Δ<0.001） |
| **09** | **權重共用（tied）複數 MHA** ★架構創新 | `MHA_re ≡ MHA_im`：一套權重，複數輸出 = 兩個 real pass 合成 `(a−b)+i(a+b)` | MHA 參數 **526,336→263,168（−50%）**；real pass **4→2**；softmax 串行次數 4→2；品質**反而上升**（finetune 0.8411 > untied 0.8397；strict W4A8 0.8383）；全模型參數 830,300→**567,132（−31.7%）** |
| **09→11** | **Exp-free shift-top4-pow2 softmax**（`shift_topk_pow2_clamp8_nomantissa`） | max-shift 定標 → top-4 選擇 → 機率 = `2^clamp(⌊score⌋,−8,0)`（無 mantissa）→ pow2 分母移位除 | **移除 exp LUT、除法器、乘法器**；只要 top-k 選擇器+加減法器+barrel shifter+LZD；MBSTOI 0.8360（Δ−0.0013 vs 精確 softmax）；TIMIT 跨域驗證不崩（+0.0616 增益） |
| **10** | **定點 STFT/iSTFT** | STFT 前後端改 16-bit 定點（Q1.14/Q5.10）與 fp8×fp8 掃描 | fp8×fp8 STFT MBSTOI 0.8170 ≈ fp32 0.8153；16-bit 定點與浮點四位小數一致 → **前端可用便宜定點 FFT 硬體** |
| **11** | **平行 MBSTOI 評估基礎設施** | GPU 推論 + CPU metric process pool 分離 | 評估加速 **~13×**（結果 byte 相同）→ 讓後續大規模 softmax/QAT 掃描可行（方法論貢獻，可寫進 implementation details） |
| **12** | **Decoder grouped pointwise + skip 子集** | PW 拆成 x-branch/skip-branch 各自 grouped（**g42**：groups_x=4、groups_skip=2）；skip 只留 L1 排名 **top-50%** 通道（靜態索引 → ROM） | decoder PW MAC **−74.7%**、全卷積 MAC −46.5%；全模型 **1.67→1.44 M MAC/frame（−14%）**；參數 567,132→**485,332（−14.4%）**；skip 頻寬減半；品質 0.8352/0.7919（在雜訊範圍內） |
| **13** | **注意力資料路徑收窄 + L-Mul m4 可行性** | (a) 注意力 scratch 全 fp16；(b) QK^T 與 Score×V 改 **L-Mul**（對數乘法：指數相加+mantissa 相加，無完整乘法器） | fp16 累加 0.8352/0.7914（免費）；L-Mul m4 QAT 後 **0.8364/0.7925（比 baseline 好）**；乘法器能量 ~**−70%**（MAC 數不變）；score-domain 量化被系統性排除（雙重離散化） |
| **14** | **Grouped QKV（qkv_groups=4）+ top-4 q0.8 稀疏封包** ★production stack | Q/K/V 投影只留對角 64×64 block（30-epoch QAT）；softmax 機率打包成 4-entry q0.8 封包（8-bit 機率+5-bit 局部索引） | QKV MAC/參數 **−75%**（RTL 196,608→49,152）；全模型 **1.39→1.10 M MAC/frame（−21%）**；64 lanes@5 MHz duty **1.02→0.81（不達標→達標）**；QKV 權重 96 KB→24 KB；機率列 **512→52 bits（−89.8%）**；composite **0.8354 / 0.7880 過雙 gate**；RTL 權重合計 **337,876（vs 09 −59.3%）** |

---

## 5. Phase 3 — 全 L-Mul 探索與品質恢復（15–17，探索性、未進 deploy）

| # | 做了什麼 | 結果 | 對論文的意義 |
|---|---|---|---|
| **15 all-L-Mul 評估** | 把 L-Mul m4 推到**所有** Conv/ConvTranspose/Linear/MHA 乘法 | 最佳 QAT/finetune 後 VCTK 0.8264–0.8281，距 gate 仍差 **~0.009** | 界定 L-Mul 適用邊界：**attention + FP8 outlier 用 L-Mul、其餘用 INT4 shift-add** 才是可部署組合 |
| **16 KD/heatmap 恢復** | 逐層誤差 heatmap、輸出級知識蒸餾（teacher=production）、bias 校正 | heatmap 熱點=最深 decoder PW；KD 最佳恢復到 **0.8288**（仍差 ~0.007）；bias 校正無效 | 消融證據：all-L-Mul 缺口不是累加器/封包/top-k 造成（fp32 累加器無改善），支持 deploy 的混合乘法器設計 |
| **17 L-Mul-aware STE** | 設計三種 L-Mul 反向傳播變體（identity / autograd_lmul ≈0.675× 梯度阻尼 / 手動 grad_scale），forward bit-exact | Smoke 驗證通過，**未正式啟動**（備援手段） | 未來工作素材；顯示恢復瓶頸可能在 STE 梯度路徑 |

---

## 6. Phase 4 — 零乘法器嚴格核心（18，deploy 模型）

拓撲/MAC 數與 14 相同（337,876 / 1.10 M），以下全部是「把每個乘法變成 shift/add/mux」：

| Feature | 做了什麼 | 好處 |
|---|---|---|
| **APoT 權重 scale**（`2^a ± 2^b`，exp∈[−16,−1]） | per-group/per-channel/per-head scale 從實數乘法改成 2 shift + 1 add | scale 乘法器歸零；0.8350/0.7892（無損） |
| **Conv INT4 shift-add** | INT4 code × e4m3 激活 = 對 significand 做 CSD shift-add（\|q\|≤7 ⇒ ≤2 個加減），descale 延後到累加器後一次做 | 卷積乘法器歸零；與逐權重 dequant 代數等價（寬累加下 bit-exact） |
| **L-Mul m8（QK/Score×V）+ m4（12 個 FP8 outlier 層）** | 浮點乘 → sign-XOR + 指數相加 + 一個 (m+2)-bit mantissa 加法 | 注意力/outlier 乘法器歸零（m4=6-bit 加法器、m8=10-bit 加法器） |
| **BN 摺疊 → APoT[−9,4]** | BN 摺成 `a_q·x + b`（帶符號 APoT），conv bias 併入 BN bias | BN 乘法器 → 2 shift + 1 add |
| **PReLU pow2 斜率**（`±2^k`，k∈[−8,0]） | 斜率量化成帶號 2 的冪 | PReLU 乘法器 → sign-mux + 1 shift |
| **Delayed 2-term APoT descale**（outlier 層，[−16,−6]） | 12 個 FP8 outlier 層的 per-tensor descale 從精確 fp32 乘法改成延遲 APoT shift-add，並在此算術下做 5-epoch 恢復 finetune | 最後一顆乘法器移除；**0.8365/0.7912，反超用乘法器的 14 composite（0.8354/0.7880）** |
| **★ pow2 block-float 激活** | 激活 = e4m3 raw code + **per-tensor pow2 共享指數**（離線校準 `k=⌊log2(448/amax)⌋`，53 個邊界）；`2^-k` descale **融合**進 post-acc 權重 descale 當指數偏移 | **完全移除 runtime amax/scale/descale**；`LMul(code·2^k,w)==2^k·LMul(code,w)` 精確成立 → **eager==fused bit-exact（48 個 outlier L-Mul 呼叫 relL2=0）** → 軟體 MBSTOI 就是晶片數字 |
| **在部署算術下訓練**（QAT/恢復 finetune） | 最終模型是在「和硬體 bit-exact 相同」的算術（L-Mul、delayed APoT、pow2 block-float）下 finetune 出來的 | 品質 gate 在真實硬體路徑上成立，不是近似模擬；**deploy 0.8357 / 0.7898** |

**核心宣稱**：神經核心（encoder+MHA+decoder 的 Conv/MHA/BN/PReLU/softmax）**零通用乘法器、零除法器、零 exp LUT**。範圍外（獨立 DSP 前後端）：STFT/iSTFT、window、波形域 mask apply。

---

## 7. Phase 5 — 硬體交付與 RTL（18_handoff / 19_algo_final）

| Feature | 內容 | 狀態/好處 |
|---|---|---|
| **交付包 `hw_patterns_final/`** | `tb_pattern/`（打包 INT4/FP8 權重碼、APoT/pow2 scale 碼、fused descale、逐層 block-float 激活、BN/PReLU 常數、skip 索引；manifest：63 權重層、12 FP8 outlier、54 INT4、missing=0）+ `sram_images/`（`wbuf_unified_init.bin` + memory map）+ golden `.hex` | RTL 的 single source of truth；`18_hw_handoff` 保留 ckpt 與完整重生工具鏈 |
| **全模型 on-chip 常駐** | WBUF 256 KiB 單埠：權重 240,336 B（INT4 223,312 + FP8 17,024）+ 常數 ~10 KB = **251,677 B（95.7% 滿）** | **無 runtime DRAM 權重搬運**——由 −91.5% 參數 + INT4 + tied MHA + grouped QKV 共同達成；徹底解掉最初的 DMA 瓶頸 |
| **CE 資料路徑** | `PEG×4 →（fused APoT descale）dealer×4 → OACC（2-bank 複數累加）→ PPU_fma`；64 複數 lane（128 real）；同一套可組態 add/shift slice 覆蓋 m4 / W4A8 / m8 / W4A16 四模式 | 任一模式都沒有專用乘法器；複數乘 = 4 PEG 流 + OACC 加減合成 |
| **PPU 共用單元** | descale+BN+PReLU+requant-to-e4m3 集中在一個 post-MAC 單元。⚠️ RTL 實作是**單一複數 PPU**（一對 re+im；吞吐足夠：每 ⌈ic/32⌉ cycle 出一個複數輸出）——「~8-lane pool」是早期演算法端的估算；論文以 RTL 為準 | 面積效率：attention（最大計算 block——deploy basis 54.0%、舊 E=512 basis ~78%）沒有 BN/PReLU，專用 lane 會閒置 |
| **緩衝設計** | KVBUF 32 KiB（dual-port ring，tied MHA 兩個 pass ⇒ 2× K/V 歷史）、UFBUF 8 KiB、SKIPBUF 4 KiB（top-50% skip） | 記憶體階層凍結（2026-06-03 SRAM spec） |
| **RTL 驗證進度**（2026-07-07 依硬體端更新） | bit-exact golden ladder 全綠：PEG 16,000 向量 × 4 模式 0 錯；逐層 golden；MHA 鏈式；**全 frame e2e + boot e2e bit-exact**；T6 真 TSMC SRAM macro 已換；multi-mode tapeout build；SpyGlass 乾淨 | **實測 frame latency 28,831 cy = 1.442 ms（warm）/ 31,554 cy = 1.578 ms（cold）@ 20 MHz**（約 5 ms 預算的 29–32%）。唯一缺口：**合成/實測 area 與 power（comparison table 缺口）** |

---

## 8. Phase 6 — 論文品質定位（20_pesq_finetune）

| Feature | 做了什麼 | 結果/主張 |
|---|---|---|
| **VoiceBank+DEMAND 對齊評估** | 824 檔、標準協定、binaural render（az0/az70）、deploy 模型 zero-shot | **STOI 0.9685（96.85）> 95.1（CICC'26-北大）> 91.44（JSSC'24-首爾大）** —— 在所有有報 STOI 的晶片對手中最高；PESQ nb 2.89→3.01 / wb 2.00→2.27；SI-SDR 17.9→17.1（健康） |
| **PESQ↔空間線索 tradeoff 消融**（劑量依賴、已證明） | 從 deploy 熱啟動、PESQ loss 權重遞增 | w=0：MBSTOI 0.834/ILD 3.33/IPD 16.0 → w18：0.823/3.44/17.7 → w30：0.773/3.98/21.6。**PESQ 權重越高，空間線索單調惡化** → 「量化不傷線索、PESQ finetune 才傷」；把弱項轉成設計意圖證據 |
| **PESQ 誠實框架** | wb/nb 分開標示；**絕不**拿我方 nb 對 CICC'26-北大的 wb 3.34；引用 BCCTN 原論文不報 PESQ（其設計目標是空間線索保真、非 PESQ——論文可直接用的完整論述範本見 `FEATURES_en.md` E6） | 高 PESQ 變體（mix_ep19 nb 3.324）SI-SDR 崩到 11.9、MBSTOI 掉到 0.802 → **報 deploy，不報灌水點** |
| **量化不犧牲空間保真**（同測資同 metric A/B） | 拿官方 BCCTN FP32 ckpt（11M 參數）與我方 W4A8 deploy 在相同 64 檔 VCTK + 相同 metric 對比 | **IPD 18.0° vs FP32 17.3°（打平）、ILD 3.76 vs 4.59 dB** → 可發表主張：「W4A8 零乘法器核心的空間線索保真與 FP32 參考相當」；並證明原論文 0.4 dB/5° 是作者私有測資特性，**絕對 ILD/IPD 不可跨測資比** |
| **對手定位表**（2026-07-07 起改用「期刊-學校」標示、廢除 P 編號；MIT VLSID'25 已剔除——非頂級 venue） | JSSC'20-台大（Lee）/ **CICC'26-北大（Zhi；VB+DEMAND，PESQ wb 3.34/STOI 95.1，D-CIM）** / A-SSCC'25-清華（Wu；visual-aided）/ JSSC'24-首爾大（Park；multiplier-less 單聲道） | **四個**對手**全是單聲道、零空間線索** → MBSTOI/ILD/IPD 無人競爭；硬體差異化：純數位零乘法器（vs CICC'26-北大的 D-CIM 更好移植；vs JSSC'24-首爾大我們是更難的 binaural transformer） |

---

## 9. 論文可直接使用的 claim 清單

1. **首創雙耳保線索 SE 加速器**（4 個對比晶片——JSSC'20-台大、CICC'26-北大、A-SSCC'25-清華、JSSC'24-首爾大——全為單聲道）。
2. **零乘法器神經核心**：Conv/MHA/BN/PReLU/softmax 全 shift/add/mux；無乘法器、除法器、exp LUT。
3. **全模型權重常駐晶片**：337,876 權重（INT4+FP8）+ 常數塞進單顆 256 KiB WBUF（95.7%）。
4. **靜態校準、零 runtime 激活縮放**：pow2 block-float + 指數偏移融合；eager==fused bit-exact（relL2=0）→ **回報的 MBSTOI 就是晶片數字**。
5. **STOI 96.85 > 95.1（CICC'26-北大）> 91.44（JSSC'24-首爾大）**（北大晶片為同資料集直接比）；MBSTOI 0.836 + ILD/IPD 保真為獨家指標。
6. **壓縮數字**：參數 −91.5%（3.97 M→338 K）、MAC −20~27×（29 M→1.10 M/frame）、注意力機率傳輸 −89.8%（512→52 bits/row）、MHA 權重 DMA 512 KB→24 KB（QKV）。
7. **tied complex MHA**：−50% MHA 參數/softmax 串行、品質反升（0.8411 > 0.8397）——架構級創新點。
8. **量化不犧牲空間保真**：W4A8 vs FP32 同測資 IPD 18.0° vs 17.3°。
9. **PESQ↔空間線索 tradeoff 消融**：劑量依賴證據，支持「設計選擇保留空間線索」的敘事。
10. **在部署算術下訓練**：QAT/恢復 finetune 直接在 bit-exact 硬體算術下進行，deploy 點反超乘法器版 production（0.8365/0.7912 vs 0.8354/0.7880，pow2 block-float 版 0.8357/0.7898）。

---

## 10. 寫作注意事項（caveats，寫論文時務必對齊）

1. **參數基準不要混用**：01/02 分析用 512-dim/32-head 舊變體（3.97 M）；03/04/06 用 256-dim；07 的 `unified_analysis_report` 算出 2.54 M 是另一種計數基準。**論文一律引用 `RESULTS_ARCH_PROGRESSION.md` 的曲線**（3,973,288 → 830,300 → 567,132 → 485,332 → 337,876 RTL）。各目錄的 `model_size.md` 是過期樣板，勿引用。
2. **WBUF 容量**：早期分析文件寫 64 KB/32 KB 目標；**凍結版是 256 KiB 全常駐**（2026-06-03 SRAM spec 為準，HANDOFF §4/§9 的 streaming 敘述已過期）。
3. **MAC 數字 basis**：原始 29 M/frame 是 hop-100/fft-512 自身 basis 的 profile 值，與 deploy 的 1.10 M（hop-80/fft-128）**不可直接同軸**，畫圖需註明。
4. **group size**：03 分析建議 gsz=256，但 deploy 實際用 **gsz=32**（07 掃描的 sweet spot）。2026-07-07 已裁定：03 的 `results_0410/summary.md`/CSV 中 BN/PReLU 列是修 bug 前舊產物、趨勢反向；限定 conv+MHA 權重（n=56）後正確趨勢為 per-tensor 7.63e-4 → gsz16 2.76e-4（最佳）→ gsz32 2.93e-4 → gsz256 3.36e-4。**絕不可引「gsz256 = 0.28×最佳」**。
5. **窗函數不一致——2026-07-07 已解決**：訓練與 RTL 都是 **Hamming**（RTL bit-exact 對過 deploy，T4）；`FUSIONS.md` 的「Hann」是文件錯誤、已加註 dated erratum。
6. **ILD/IPD 絕對值不可跨測資比**（原論文 0.4 dB/5° 為其私有測資）；論文報「相對 noisy 的改善量」與「同測資 FP32 vs W4A8 保真」。
7. **per-SNR 完整表——2026-07-07 已鎖定**：`18_..._hw_handoff_20260530/results/apot_descale_eval/ft_pow2_ep3_full/per_snr_locked_paper_table.csv`（VCTK+TIMIT × 8 SNR × noisy/enh/Δ × 5 指標；per-SNR 平均重現 0.8357/0.7898；canonical 以 `*_averages.txt` 為準）。VCTK/TIMIT 的 per-SNR STOI/ESTOI/SI-SDR 若需要仍要補跑。
8. **晶片 power/energy/area/node 待 RTL bring-up** 後才能填 comparison table。
9. **15–17 的 all-L-Mul 不在 deploy 內**（探索性）；deploy 的 L-Mul 只用在 QK/Score×V（m8）與 12 個 FP8 outlier 層（m4），寫作時勿混淆。
10. **PESQ-leaning finetune 全部掉出 MBSTOI gate** → 最終回報模型 = deploy（低/零 PESQ 壓力點）。
11. **「attention 佔 77.6% MAC」是 embed-512 舊變體、不是 deploy 模型。** Deploy basis（E=256/16，2026-07-07 重跑）：total 0.823 G MACs/s、attention 54.0 / decoder 30.8 / encoder 15.2。「attention 是最大 block」兩個 basis 都成立；引哪個 basis 都要標明。⚠️ 這個重跑值是**第三種 MAC 口徑**（complex→4-real 慣例、pre-09 untied 架構；÷200 fps ≈ 4.1 M/frame）——與 deploy 的 1.10 M multiply-equivalent/frame **不可換算、不可互除**；只能拿來引用百分比分佈。完整稽核：`RESULTS_ANALYSIS_VERIFICATION_01_06_2026-07-07.md`。
12. **06 分析已於 2026-07-07 用 n=20 重跑**——QK^T 1049>448、12/16 尖銳 head 皆確認；引用 n=20 數字。其 IPD ~1.4 rad 仍是飽和的 raw-STFT 指標——絕不可與 15–21° 的 eval 數字並列。top-4 品質引 07/14。
13. **05 低秩「+18–27%」節省只在 95% energy 門檻成立**（99% 全部轉負）；05 的「num_heads=32」是文件錯（實際 16，不影響數字）；MHA 640 KB 少算複數外圍 proj（全算約 768 KB）。
14. **04 的「6 dense = Q/K/V/O projections」是誤標**——6 層中 5 層是 conv；稀疏性結論本身已驗證、可安全引用。
15. **論文用圖**：2026-07-07 已用修正後資料重生於 `paper_figures/`（SVG + Visio 用 EMF + PNG）；各圖資料源與 caption 必註 caveat 見 `paper_figures/CAPTIONS.md`。
