# RESPONSE: 顧問 Open Items（§10）— 論文引用數字/資料位置/引用文獻

Date: 2026-07-07
Base candidate: `18_w4a8_pow2_mixed_lmul_hw_handoff_20260530/ckpt/strict_core_pow2_blockfloat_deploy_ep3_vctk0.8357_timit0.7898.ckpt`
Handoff: 顧問 §10 Open Items（2026-07-07 由 chat 轉交，無 repo 內前導文件）；相關前導：[RESPONSE_FP16_PPU_EVAL_2026-06-10.md](RESPONSE_FP16_PPU_EVAL_2026-06-10.md)、[RESULTS_ARCH_PROGRESSION.md](RESULTS_ARCH_PROGRESSION.md)

## 0. TL;DR

| # | Open item | 回答 | 狀態 |
|---|---|---|---|
| 1 | dec-PW MAC −62% vs −74.7% | 兩者皆對、不同 feature 範圍：**−62.16% = 只做 grouping（P1 g42）**；**−74.74% = grouping + top-50% skip subset（P2 g42+s050_topk，= deploy 組態）**。論文引 **−74.7%（dec-PW local）/ −46.5%（conv-total）**，可註明分解 | 已解 |
| 2 | MBSTOI 引用對 | 同意顧問建議：**本文引 deploy 0.8357 / 0.7898**，footnote HW-chain-certified **0.8356 / 0.7897**（Δ ≤ ±0.0001，含 FP8 preshift 重烤後全 fp16 鏈） | 已解 |
| 3 | per-SNR 表未鎖定 | **已鎖定**：`per_snr_locked_paper_table.csv`（16 rows，VCTK+TIMIT × 8 SNR，noisy/enh/Δ × 5 指標），平均值 sanity 重現 0.8357 | ⭐⭐ 完成 |
| 4 | E-series raw data 位置 | **在本 repo**：`20_pesq_finetune/results/{pesq_eval,voicebank_demand}/`（45+ run dirs，含 per-SNR 明細）。缺口：per-run 指令列/ckpt hash 未系統性記錄 → 投稿前補 manifest | 部分解 |
| 5 | P1–P5 citations | 見 §5 全清單。**注意：SPEC 與 STATUS 兩文件 P3/P4/P5 編號互相矛盾**；建議鎖 STATUS（較新）編號；MIT VLSID'25 無 PDF、STATUS 表未收錄，需決定是否納入 | 待決定編號 |

---

## 1. dec-PW MAC reduction：−62% vs −74.7% 的 counting basis

來源：[12_decoder_grouped_pointwise/RESULTS_P1_GROUPED_2026-05-17.md](../../12_decoder_grouped_pointwise/RESULTS_P1_GROUPED_2026-05-17.md)、[RESULTS_P2_SKIP_2026-05-18.md](../../12_decoder_grouped_pointwise/RESULTS_P2_SKIP_2026-05-18.md)。

| Stack | dec_pw MAC | Δ vs baseline | full_conv MAC | Δ vs baseline |
|---|---:|---:|---:|---:|
| baseline（split, ungrouped） | 37,698,560 | — | 60,559,680 | — |
| g42（P1，只 grouping） | 14,264,320 | **−62.16%** | 37,125,440 | −38.70% |
| **g42 + s050_topk（P2，= deploy）** | **9,526,528** | **−74.74%** | **32,387,648** | **−46.52%** |

- **Basis**：module-call MAC，hook 只掛 `Conv2d`/`ConvTranspose2d`——**不含** MHA `nn.Linear`、attention score、STFT（P1 doc §5、P2 doc §3 附註）。複數 4-real convention 不影響 ratio。
- PROG（[RESULTS_ARCH_PROGRESSION.md](RESULTS_ARCH_PROGRESSION.md)）的「decoder PW −62%」列只對應 **P1 grouping**，skip subset 列在下一 row 沒重引累計值——不是矛盾，是拆列。
- **論文引用建議**：deploy 組態 = g42+s050_topk，引 **−74.7%（dec-PW local）**；若行文分解則寫「grouping −62.2%，再加 top-50% skip subset 至 −74.7%」。全模型 normalized basis（含 MHA）對應 **1.67→1.44 M/frame（−14%）**，是另一個 basis，勿與 conv-only 數字同軸並列。

## 2. MBSTOI 引用對：deploy vs HW-chain-certified

同意顧問建議，採：

- **本文**：deploy **VCTK 0.8357 / TIMIT 0.7898**（`results/apot_descale_eval/ft_pow2_ep3_full/{vctk,timit}_paper_style_results_averages.txt`）。
- **Footnote**：整條 fp16 HW 鏈（conv PPU fp16 tree+dealer+BN+PReLU+cast、decoder INT4、FP8-outlier L-Mul 在 fp16 tree + **重烤後 preshift**）認證值 **0.8356 / 0.7897**，Δ ≤ ±0.0001（[RESPONSE_FP16_PPU_EVAL_2026-06-10.md](RESPONSE_FP16_PPU_EVAL_2026-06-10.md) §3；dec4 Inf witness 消失）。
- 這個 footnote 直接強化 claim 4（「回報的 MBSTOI 就是晶片數字」）：軟體 deploy 值與 bit-accurate HW 鏈值差 ≤ 1e-4。
- 相鄰變體佐證（同 doc §2）：full fp16-PPU 0.8357/0.7897、BN-fp32/tree-fp16 0.8357/0.7899——全部 ±0.0001 噪聲級。

## 3. per-SNR 表：已鎖定

**新 artifact**：`18_w4a8_pow2_mixed_lmul_hw_handoff_20260530/results/apot_descale_eval/ft_pow2_ep3_full/per_snr_locked_paper_table.csv`

- 內容：**VCTK + TIMIT × SNR {−6,−3,0,3,6,9,12,15} dB**，每列 {noisy, enhanced, Δ} × {MBSTOI, PESQ, SegSNR, ILD_err_dB, IPD_err_deg}（SPEC §C 格式）。
- 來源：既有 `ft_pow2_ep3_full/{vctk,timit}_paper_style_results.csv`（deploy 全量 eval，非重跑）。
- Sanity：per-SNR 等權平均重現 VCTK **0.8357**；TIMIT 等權平均 0.7897 vs 官方 averages 0.7898（等權 vs 檔數加權的捨入邊界；**canonical 引用以 averages txt 為準**，per-SNR 表供逐點圖/Δ 用）。
- 尚缺（若要完整對 SPEC §C）：VCTK/TIMIT 的 per-SNR **STOI/ESTOI/SI-SDR** 不在既有輸出內（只有 VB+DEMAND run 有）——需要的話要補跑 eval（估 1 GPU、數小時），請顧問確認是否需要。

## 4. E-series raw data（20_pesq_finetune）

資料**在本 repo**（19_algo_final 是 frozen handoff、本來就不含；請查 20）：

| 內容 | 路徑 |
|---|---|
| VCTK/TIMIT PESQ-ft eval（45+ runs：baseline_deploy、w6…w42、mix、anneal、per-SNR 明細） | `20_pesq_finetune/results/pesq_eval/<run>/{vctk,timit}/snr_*dB/average_results.txt` + `*_paper_style_results.csv` |
| VoiceBank+DEMAND eval（az0/az70、deploy/finetune 系列） | `20_pesq_finetune/results/voicebank_demand/<run>/` |
| eval 入口 | `20_pesq_finetune/code/eval.py`、`code/eval_voicebank_demand.py`、`scripts/` |
| 結論文件 | [20_pesq_finetune/docs/STATUS_CHIP_PAPER_COMPARISON_POSITIONING_2026-06-28.md](../../20_pesq_finetune/docs/STATUS_CHIP_PAPER_COMPARISON_POSITIONING_2026-06-28.md) |

**已知缺口（投稿前補）**：per-run 的精確指令列、Hydra config snapshot、ckpt md5 未系統性記錄在 run dir 內。建議產一份 `E_SERIES_MANIFEST.md`（run → ckpt → cmd → 數字表格），演算法端可在下一輪補齊。

## 5. P1–P5 citations（自 `comparison_table/` PDF 首頁抽取）

| # (STATUS 編號) | Citation | 數字（給 comparison table） |
|---|---|---|
| P1 | Y.-C. Lee, T.-S. Chi, C.-H. Yang, "A 2.17-mW Acoustic DSP Processor With CNN-FFT Accelerators for Intelligent Hearing Assistive Devices," **IEEE JSSC, vol. 55, no. 8, Aug. 2020**（首頁 p. 2247） | 40nm, 2.17 mW; WAR +41%; IEEE/NOISEX-92 |
| P2 ⭐ | Q. Zhi, W. Ren, S. He, M. Li, Y. He, L. Ye, T. Jia (Peking Univ.), "A 22nm 15.2 µJ/Inference Speech Enhancement Accelerator with Contextual-Aware KV-Cache Management and Reconfigurable D-CIM Engine," **CICC 2026**（venue 依 STATUS doc；PDF 無 venue header，camera-ready 前需補頁碼確認） | 22nm, 1.9 mW, 15.2 µJ; VB+DEMAND PESQ_wb 3.34 / STOI 95.1; D-CIM |
| P3 | N. Wu, S. Wang, P. Wang, H. Li, K. Liu, X. Liu, Y. Guo, Z. Wang, H. Jiang (Tsinghua/Fudan/NTU-SG), "A 3.64 µJ/frame Visual-Aided Speech Enhancement Processor Tolerating −10dB SNR Input for Hearing Aids in 22nm CMOS," **IEEE A-SSCC 2025, Session 20, Paper 20.1** | 22nm, 3.64 µJ/frame; TMSV(+visual) PESQ 2.13 @ −10 dB |
| P4 | S. Park, S. Lee, J. Park, H.-S. Choi, K. Lee, D. Jeon (SNU), "A Real-Time Speech Enhancement Processor for Hearing Aids in 28-nm CMOS," **IEEE JSSC**（ISSCC'23 延伸；PDF 為 JSSC 版，卷期頁碼 early-access 未印，camera-ready 前補） | 28nm, 0.74 mW; CHiME-2 PESQ 2.73(nb) / STOI 91.44 / SDR 14.0; multiplier-less; SepConv+GRU 56k |
| （P5?） | **MIT VLSID'25**（CHiME2 8 kHz, PESQ 2.79）：**`comparison_table/` 無 PDF、STATUS 表未收錄** | 待決定 |

**編號衝突（需鎖定）**：[演算法端_品質對齊_SPEC.md](../../演算法端_品質對齊_SPEC.md) §1 的編號是 P3=SNU JSSC'24、P4=MIT VLSID'25、P5=Tsinghua；[STATUS_CHIP_PAPER_COMPARISON_POSITIONING_2026-06-28.md](../../20_pesq_finetune/docs/STATUS_CHIP_PAPER_COMPARISON_POSITIONING_2026-06-28.md) §1 是 P3=Wu(Tsinghua)、P4=Park(SNU)、無 MIT。**建議**：以 STATUS（較新、含完整定位論述）為 canonical 編號；MIT VLSID'25 是否進 comparison table 請顧問決定——要進的話演算法端需先取得 PDF 核對數字。

## 6. 相關通告（非顧問所問，但影響共同引用的數字）

演算法端本輪對 01–06 分析做了 paper 前驗證，其中一項影響雙方共用文件：**01 utilization 的「attention 佔 77.6% MAC」是 `embed_dim=512/32-head` 舊模型的 profile**（profiler 寫死 512）；用同一 profiler 公式換到 deploy 尺寸（256/16）時分佈為 **attention 54.0 / decoder 30.8 / encoder 15.2（total ≈0.82 G）**。「attention 是最大 block」的結論不變，但引用絕對比例時需標明 basis。PROG 文件與 HANDOFF 內引用 77.6% 之處均同此註。完整驗證報告見演算法端後續 RESULTS 文件。

## 7. Reproduction

```bash
# per-SNR locked table（已產出，重生用）
cd /raid/R12K41024/18_w4a8_pow2_mixed_lmul_hw_handoff_20260530
# 來源 CSV: results/apot_descale_eval/ft_pow2_ep3_full/{vctk,timit}_paper_style_results.csv
# 輸出:     results/apot_descale_eval/ft_pow2_ep3_full/per_snr_locked_paper_table.csv

# citations 抽取
cd /raid/R12K41024/comparison_table && python -c "from pypdf import PdfReader; ..."
```
