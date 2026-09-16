# RESULTS: 01–06 分析驗證（論文引用前的正確性稽核）

Date: 2026-07-07
Base candidate: `18_w4a8_pow2_mixed_lmul_hw_handoff_20260530/ckpt/strict_core_pow2_blockfloat_deploy_ep3_vctk0.8357_timit0.7898.ckpt`（deploy 參照）
Handoff: [PAPER_FEATURE_SUMMARY_2026-07-07.md](PAPER_FEATURE_SUMMARY_2026-07-07.md)（本報告是其 Phase 1 章節的稽核附件）；相關：[19_algo_final/docs/RESPONSE_OPEN_ITEMS_PAPER_NUMBERS_2026-07-07.md](19_algo_final/docs/RESPONSE_OPEN_ITEMS_PAPER_NUMBERS_2026-07-07.md)

## 0. TL;DR

| Dir | 驗證結論 | 論文可用性 | 關鍵動作 |
|---|---|---|---|
| 01 utilization | 數字忠於腳本（逐一重現），**但 profiler 寫死 embed 512/32-head，非 deploy 模型** | ⚠️ 定性結論可用；絕對數字須換 basis | ⭐⭐ 已用 E=256/16 重跑：**total 0.823 G；attention 54.0 / decoder 30.8 / encoder 15.2**；W4A8 utilization 45.1→45.1（結論不變） |
| 02 bitwidth | 內部算術一致、方法正確（symmetric abs-max、正確 deploy config），**但 raw eval 輸出未留存（results/ 被 gitignore）** | ⚠️ 數字不可重現；引用需附說明或重跑 | tier 數矛盾（29+1 vs「30 全過 W3」）、「500 pairs」與腳本預設矛盾 — 引用時避開這兩點 |
| 03 dynamic range | **report 正確、`results_0410/summary.md` 與部分 CSV 是修 bug 前舊產物**（BN/PReLU 列 MSE 趨勢反向） | ✅ report 的趨勢與 5-outlier/per-head/re-im 結論可用 | group-size 圖已改用「conv+MHA 權重限定（n=56）」重算：per-tensor 7.63e-4 → gsz16 2.76e-4（最佳）→ gsz256 3.36e-4；**勿引 summary.md 的 gsz256=0.28×** |
| 04 sparsity | 數字全部重現、結論站得住；root TODO 的「仍有 bug」晚於結果 6 天，但查無殘留計算 bug | ✅ 定性結論可用（加註） | 「6 dense = Q/K/V/O」是錯的（實為 5 conv + 1 MHA）；authoritative dir = `results/`（非 results_test/） |
| 05 conv reuse | 算術全對；`num_heads=32` 是文件錯（實際 16，不影響任何數字）；TODO「05 x」指的是 model-eval 待辦，非 reuse 分析 | ✅ 可用（低秩數字限 @95% energy） | 低秩 +18–27% 節省**只在 95% energy 門檻**成立、99% 全部轉負；MHA 640 KB 少算複數外圍 proj（全算 ≈768 KB），定性不變 |
| 06 attn precision | 數字忠於 CSV，**但全部（動態範圍/靈敏度/entropy/top-k 全模型）都是 n=2**；20-sample run 從未執行 | ⚠️ 只能當 early feasibility probe 引用 | IPD 1.455 rad 是飽和無效指標（勿與 15–21° 並列）；summary.md「V 範圍較小」與自身資料相反；int16 accumulator 建議已被 13/14/18 的 FP16+L-Mul 取代（只有 top-4 存活到 deploy） |

## 1. 跨目錄系統性問題

1. **模型變體不一致**：01/02 分析 512-dim/32-head；03/04/05/06 分析 256-dim（03 的 `config/model.yaml` 仍是 512 舊檔，權重實測 256）。論文引用一律標 basis。
2. **`results/` 被 gitignore**：01/02 報告引用的 raw 檔已不在磁碟（01 可由純解析式腳本重現，02 不可）。
3. **各目錄 `model_size.md`/`readme.md`/`hardware_handover_report.md` 是共用樣板**，內容（3.97M、INT8、window 32）與該目錄實驗無關，一律勿引。
4. **06 的 `analysis_results/unified_analysis_report.txt`（2.5M params）是另一支舊工具的輸出**，與 06 報告無關，勿引。

## 2. 01 — deploy basis 重跑（本次新產出）

方法：`utilization_profiler.py` 為純解析式（ckpt 載入後即棄用），將模組層級 `EMBED_DIM/NUM_HEADS` 換為 256/16 後重建 layer table（先重現 512 版全部數字再切換，確認 apples-to-apples）。逐層明細：`paper_figures/data/util_rows_e256.csv`。

| 量 | E=512/32（舊報告） | **E=256/16（deploy basis）** |
|---|---|---|
| Total MACs（1 s 音訊） | 1,687,427,200 | **823,171,200** |
| Attention / Decoder / Encoder | 77.6 / 15.0 / 7.4 % | **54.0 / 30.8 / 15.2 %** |
| 平均 utilization FP8 → W4A8 | 45.3 → 45.4 % | **45.1 → 45.1 %** |
| 子運算 utilization（不變的定性結論） | — | input_proj 91.4、QKV proj 92.3、out_proj 88.9（compute-bound）；QK^T 49.6、Score×V 33.0（act-DMA bound） |

其他修正：報告「encoder/decoder depthwise 皆 13–17%」有誤——decoder depthwise-transpose ≈ **29.4%**（E 無關）；「MHA 512→256 KB」是 E=256 值、與同報告其它 E=512 數字不同 basis（數值本身對 deploy 是對的）。utilization 定義為 compute/(compute+max(DMA)) 之**無重疊下界**，論文引用時應註明。複數係數在 QK^T/Score×V 用 ×8、projection 用 ×4（影響 <3% MACs），引用比例時無妨。

## 3. 03 — group-size 矛盾的裁定（機制證明）

- `summary.md` 的表 = `group_size_sensitivity.csv` 全 100 列的欄平均；其中 45 列（全部 BN、PReLU、19 個小 conv）呈現「群越大 MSE 每倍砍半」的非物理行為 → 舊版 `quantize_w4_mse` 的 padding bug（現行腳本 docstring 明言已修，但 `results_0410` 從未重生）。
- 限定 real conv+MHA 權重（n=56）後趨勢翻正：**per-tensor 7.628e-4 → gsz16 2.757e-4（最佳）→ gsz32 2.933e-4 → gsz256 3.360e-4**（`paper_figures/make_paper_figures.py` fig04 內即時重算）。
- 誠實寫法：「更細的 group 單調降低量化 MSE，16–256 全部在最佳值 ~1.3× 內；deploy 採 QAT 掃描選出的 gsz=32」。**不可**寫「256 最優」。

## 4. 06 — 引用邊界

- **全部結果基於 2 個測試樣本**（單一 `audio_pairs` 餵所有子分析；報告只對 sensitivity 承認 n=2，實則動態範圍/entropy/top-k 全模型皆同）。20-sample run 無任何輸出存在。
- **top-4 的品質背書請改引後續大規模驗證**：07（750-pair paper-style, topk4 0.8340）、14（packet c3 0.8360）。06 只證「可行性方向」。
- IPD 1.455 rad（83.4°）為全 TF-bin 未加權 raw-STFT 指標，對 k 完全不敏感（1.453–1.456 常數）→ 飽和、無資訊，**絕不可**與 eval pipeline 的 15–21° 並列。
- 06 建議中**只有 top-4 存活到 deploy**；「QK^T accumulator int16 / 全 fp16」已被 13/14/18 的 FP16 accumulator + W4A8 + L-Mul m8 取代——引用 06 表格當「最終設計」會與 18 自相矛盾。

## 5. 修正後的論文安全引用集（Phase 1 分析）

| 主張 | 引用值 | 來源 |
|---|---|---|
| Attention 是最大 compute block | 54.0%（deploy basis；512-variant 77.6% 僅作歷史動機） | 本報告 §2 |
| W4A8 不是為 utilization、是為 SRAM/DMA | 45.1→45.1%（無變化）+ MHA 權重減半 | 本報告 §2 + 02 |
| INT4 是最低穩健 uniform 位寬 | −0.56 dB / ILD +0.21 / IPD +0.64°；INT3 −5.79 dB 崩潰；逐層樂觀（29 層各自過、合併 −3.08 dB） | 02（註明 raw 未存檔） |
| 5 個 FP8 outlier 層 | kurtosis 12.33/9.36/8.46/7.33/6.38（decoder PW 0/3/4） | 03 `dynamic_range_table.csv`（已驗證） |
| Per-head scale 必要 | 10/12 head-group 非均勻（>1e-3） | 03 `per_head_variance.csv` |
| 更細 group 降 MSE、deploy 用 32 | 7.63e-4 → 2.76e-4（gsz16）→ 2.93e-4（gsz32） | 本報告 §3 |
| 稀疏不可硬體化 | 43/52 層 random-unstructured；結構化僅最末 decoder 小層；無可剪 head | 04 `results/`（已驗證） |
| MHA 是 DMA 瓶頸 | 640 KB（複數外圍 proj 全算 ≈768 KB）≈ 4× 全部 PW conv | 05（已驗證，附註計法） |
| QK^T 超出 e4m3 | ±911.5 > 448（n=2 probe；deploy 對策 = fp16 accumulation） | 06（標 n=2） |
| Top-4 幾乎免費 | 06 為 n=2 方向性證據；品質背書引 07 的 0.8340 / 14 的 0.8360 | 06→07/14 |

## 6. Reproduction

```bash
# 01 deploy-basis 重跑（純解析、無 GPU）
cd /raid/R12K41024/01_utilization_analysis
python - <<'EOF'
import importlib.util
spec = importlib.util.spec_from_file_location("up", "utilization_profiler.py")
up = importlib.util.module_from_spec(spec); spec.loader.exec_module(up)
up.MODEL_CFG.update(embed_dim=256, num_heads=16); up.EMBED_DIM=256; up.NUM_HEADS=16
rows = up.add_w4_column(up.build_layer_table())
EOF

# 03 group-size 重算（conv+MHA 限定）與全部論文圖
cd /raid/R12K41024/paper_figures && python make_paper_figures.py
```

建議（非阻塞）：(a) 用現行已修復的 `dynamic_range_analyzer.py` 重生 `results_0410`，一併補回缺失的 `per_channel_variance.csv`（「42/42」主張目前無 artifact 支撐）；(b) 06 以 n=20 重跑一次（1 GPU、短時）讓動態範圍/entropy 數字可安心引用；(c) 02 重跑一次並把 `results/` 移出 gitignore 或另存，使品質數字可重現。

---

## 7. Addendum（同日執行完成）：三個重跑全部收工（GPU 2 一張卡序列）

| 重跑 | Artifacts | 新數字 vs 舊報告 | 判定 |
|---|---|---|---|
| **06 n=20**（舊 n=2 備份於 `results_n2_2026-04-16/`） | `06_attn_mised_precision/results/`（覆蓋為 n=20） | QK^T \|max\| **1049 > 448**（n=2: 911）；Q ±51 / K ±71 / V ±104；**12/16 sharp heads 不變**；top-4 L1 曲線同型；IPD ~1.42 rad 仍飽和（指標本身無效，非數據問題） | ✅ 結論全部保持且更強；引用 n=20 |
| **03 regen**（修復版 analyzer，num_heads=32 同原設定） | `03_dynamic_range/results_regen_2026-07-07/` | 全表 group-size 平均**精確重現報告表格**（1.314e-3 → gsz16 9.572e-4 (0.73×) → gsz256 1.202e-3，單調）→ 證明報告出自修復版、舊 `results_0410` 才是壞快照；**`per_channel_variance.csv` 重生：42/42 conv var_max>1e-4**（最壞 0.526/0.494 = 報告值） | ✅ 03 報告現在完全有 artifact 支撐；引用 regen dir |
| **02 n=500 seed 42**（ckpt = `07_.../epoch=134-step=130680.ckpt`） | `02_weights_bitwidth/results_rerun_2026-07-07/`（sensitivity_matrix、mixed_precision_summary.csv、fp8_baseline.json 全保留） | **結論級全部重現**：tier **29 C + 1 B** 精確一致（先前的 29+1 vs「30 全過」矛盾以 artifact 裁定為 29+1）；省 61.5%/50%/62.5% 精確一致；Δ（vs FP8 baseline 7.429）：INT8 −0.05、**INT4 −0.38 dB / ILD +0.20 / IPD +0.56°**、INT3 **−4.36 dB 崩潰**、mixed **−2.31 dB**（逐層樂觀性重現：單層 W3 Δ 全部 ~0.0x dB）。絕對值與舊報告不同（舊 baseline 10.49 vs 新 7.43）＝評估子集不同；舊絕對值仍不可重現 | ✅ 論文引用**新 n=500 數字**（有 artifact）；舊報告數字僅作歷史參考 |

**圖已同步**：fig02（02 新 Δ）、fig04（03 regen 官方表）、fig06/fig07（06 n=20）皆改為直接讀上述 artifact 重生（`paper_figures/`，同步包已更新）。

**補充（同日，deploy 對應性檢查後）**：
- fig01 改名 `fig01_mac_breakdown_baseline_e256`（原 `_deploy` 誤導）：它是 deploy「尺寸」（E=256/16）但 **pre-09 架構**（untied/dense/未 grouped）的 baseline 分佈——動機圖，非 deploy 網路的分佈。
- fig05 改用 **deploy 對齊的 16-head 切分**（`results_regen_2026-07-07_h16/`）：**8/12** 矩陣不均勻（>1e-3，最壞 output_proj.imag 5.6e-3）；報告原本的 10/12 是在 16-head 模型上做 32 路切分。per-head scale 的結論在 deploy 粒度下成立。
- `paper_figures/CAPTIONS.md` v2 新增 **model/stage 欄**（BASELINE-PROFILE / FP8-QAT-BASELINE / DEPLOY），明確標出每張圖算在哪個模型上：fig10/11/12 = DEPLOY 本體，fig01 = baseline profile，fig02–09 = FP8-QAT 前身（分析階段）。
