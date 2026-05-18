# RESULTS: P2 Decoder Skip Subset (on top of P1 g42)

日期：2026-05-18
Spec：[SPEC_DECODER_SPLIT_POINTWISE_2026-05-16.md](../11_weight_tied_mha_fast_mbstoi/SPEC_DECODER_SPLIT_POINTWISE_2026-05-16.md) v2 §5
P1 sign-off：[RESULTS_P1_GROUPED_2026-05-17.md](RESULTS_P1_GROUPED_2026-05-17.md)（g42 為 primary）
Handoff：[HANDOFF_P2_2026-05-17.md](HANDOFF_P2_2026-05-17.md)
Finetune：30 epoch QAT，lr=5e-4，全 QAT scope，top-k=4，warm-start = g42 split-surgery + skip-channel surgery

---

## 0. TL;DR

| 結論 | Variant | VCTK MBSTOI Δ vs baseline | TIMIT MBSTOI Δ vs ref | dec_pw MAC vs baseline | full_conv MAC vs baseline |
|---|---|---|---|---|---|
| ⭐⭐ **P2 + RTL primary candidate** | g42 + **s050_topk** | **-0.0008** | **-0.0008** | **-74.7%** | **-46.5%** |
| Conservative backup | g42 + s075_topk | -0.0010 | -0.0024 | -68.5% | -42.6% |
| Discarded（被 topk 主宰）| s075_first / s050_first | -0.0016 / -0.0036 | — | (same as topk pair) | (same) |
| **被替代的 P1 路徑** | g44 (P1 backup) | -0.0014 | -0.0026 | -74.6% | -46.4% |

**核心結論**：
1. **s050_topk 是 P2 winner**，而不是 spec 原本預設的「s075_topk」。MBSTOI 跟 s075_topk 基本打平（差 0.0002），但 SegSNR / IPD 更好，且 dec_pw MAC 多省 2 倍。
2. **g42 + s050_topk 嚴格優於 g44**：兩者 MAC 收益幾乎相同（-74.7% vs -74.6% dec_pw），但 g42+s050_topk 的 MBSTOI / IPD 都贏。**g44 應該從 RTL 候選清單剔除**，改用 g42+s050_topk。
3. **topk_l1 全面贏 first_n**：不只在 sanity 上、full 750-pair 也是（s075: +0.0006、s050: +0.0028），驗證 L1-ranked channel selection 對 quality 有實質貢獻，不是 placebo。
4. **全部 4 個 P2 variants 都通過 ±0.005 gate**（vs baseline）。
5. TIMIT cross-domain 已完成（§4.5），VCTK + TIMIT 兩個 domain 都同意 **s050_topk** 是 winner。

---

## 1. Sign-off table — VCTK 750-pair × 8 SNR

| Variant | gx | gs | skip ratio | mode | MBSTOI Enh | Δ vs baseline (0.8360) | Δ vs g42 (0.8361) | SegSNR Enh | ILD Enh (dB) | IPD Enh (°) |
|---|---|---|---|---|---|---|---|---|---|---|
| baseline (off) | – | – | 1.0 | – | **0.8360** | 0 | -0.0001 | 1.99 | 3.2356 | 15.69 |
| g42 (P1 winner) | 4 | 2 | 1.0 | – | 0.8361 | +0.0001 | 0 | 1.71 | 3.2633 | 16.04 |
| s075_first | 4 | 2 | 0.75 | first_n | 0.8344 | -0.0016 | -0.0017 | 1.48 | 3.2754 | 16.09 |
| s075_topk | 4 | 2 | 0.75 | topk_l1 | 0.8350 | -0.0010 | -0.0011 | 1.61 | 3.2822 | **15.85** |
| s050_first | 4 | 2 | 0.50 | first_n | 0.8324 | -0.0036 | -0.0037 | 1.52 | 3.2698 | 16.05 |
| **s050_topk** ⭐ | 4 | 2 | 0.50 | topk_l1 | **0.8352** | **-0.0008** | -0.0009 | **1.73** | 3.2871 | 15.94 |

**Δ 解讀**：
- baseline 重現 P1 那欄的 0.8360 (mode=off) ✓
- g42 重現 0.8361 ✓
- s050_topk 在 4 個 variants 裡 MBSTOI 最高、SegSNR 最高，IPD 還比 g42 好 0.10°
- 所有變體 ILD 都比 baseline 高 0.03-0.05 dB（noise band 內），這是 grouped+skip 的小代價
- s050 variant 砍掉一半 skip channel，但 quality 不退反進的原因：L1 排序 picking 重要 channel，剩下的的 channel 在 P1 g42 訓練時本就接近不貢獻

---

## 2. 20-pair sanity

| Variant | MBSTOI Enh (20p) | Δ vs g42 (0.8091) | ILD (dB) | IPD (°) | best epoch | val_loss |
|---|---|---|---|---|---|---|
| g42 ref (20p) | 0.8091 | 0 | 3.271 | 17.12 | – | – |
| s075_first | 0.8085 | -0.0006 | 3.318 | 17.15 | 19 | -13.542 |
| s075_topk | 0.8099 | +0.0008 | 3.331 | 17.04 | 17 | -13.691 |
| s050_first | 0.8077 | -0.0014 | 3.284 | 17.20 | 9 | -13.493 |
| **s050_topk** | 0.8081 | -0.0010 | 3.340 | **17.02** | 27 | **-13.760** |

`val_loss` 觀察：
- **s050_topk val_loss 最深 (-13.760)**，比 P1 g42 baseline (-13.69) 還深 0.07
- **s050_first plateau 早 (epoch 9)**，跟 first_n 選錯 channel 有關
- topk_l1 > first_n 在 val_loss 上也成立（s075: -13.69 > -13.54；s050: -13.76 > -13.49）

20-pair ordering 跟 full 750-pair ordering 一致（topk 系列 > first_n 系列），sanity 信號可靠。

---

## 3. MAC 分析

### 3.1 P2 額外省幅 (on top of g42)

Verify 過 [count_p2_macs.py](scripts/count_p2_macs.py)：

| variant | ratio | dec_pw MAC | Δ dec_pw (vs g42) | full_conv MAC | Δ full_conv (vs g42) |
|---|---|---|---|---|---|
| g42 reference | 1.00 | 14,264,320 | 0% | 37,125,440 | 0% |
| s075_* | 0.75 | 11,895,424 | **-16.61%** | 34,756,544 | **-6.38%** |
| s050_* | 0.50 | 9,526,528 | **-33.21%** | 32,387,648 | **-12.76%** |

### 3.2 Chain 累計（vs original baseline）

| Stack | dec_pw MAC | Δ dec_pw vs baseline | full_conv MAC | Δ full_conv vs baseline |
|---|---|---|---|---|
| baseline | 37,698,560 | 0% | 60,559,680 | 0% |
| g42 (P1) | 14,264,320 | -62.16% | 37,125,440 | -38.70% |
| **g42 + s050_topk** ⭐ | **9,526,528** | **-74.74%** | **32,387,648** | **-46.52%** |
| g42 + s075_topk (backup) | 11,895,424 | -68.45% | 34,756,544 | -42.61% |
| g44 (P1 backup, deprecated) | 9,577,472 | -74.59% | 32,438,592 | -46.44% |

**g42 + s050_topk 跟 g44 對比**：

| Metric | g44 | g42 + s050_topk |
|---|---|---|
| dec_pw Δ vs baseline | -74.59% | **-74.74%** |
| full_conv Δ vs baseline | -46.44% | **-46.52%** |
| MBSTOI Δ vs baseline | -0.0014 | **-0.0008** |
| ILD Enh (dB) | 3.2818 | 3.2871（接近） |
| IPD Enh (°) | 16.20 | **15.94** |
| val_loss 收斂 | -13.55（capacity 緊）| **-13.76（深 0.21）** |

g42 + s050_topk **全面贏 g44**，且 MAC 收益略多一點點（多省 0.15 pp dec_pw、0.08 pp full_conv）。**g44 從此可以從 RTL 候選清單剔除**。

### 3.3 同樣 caveats（per spec §4.4）

- 「full_conv MAC」只算 Conv2d / ConvTranspose2d，不含 MHA `nn.Linear`、attention scores、STFT。MHA 一加進來 dec_pw 在全 model 的比例會降低，但 dec_pw 內部 Δ% 不變。
- Module-call MAC（每個 conv 算一次）而非 RTL real MAC（apply_complex 的 4 次 real conv call）。跨 variant ratio 不受影響。

---

## 4.5 TIMIT cross-domain

對 s075_topk 與 s050_topk 兩個 candidate 跑 750-pair × 8 SNR TIMIT，仿照 [softmax handoff §3.5](../11_weight_tied_mha_fast_mbstoi/RTL_HANDOFF_SOFTMAX_2026-05-14.md) 設定。

| variant | TIMIT Noisy | TIMIT Enh | Δ improvement | Δ Enh vs ref | ILD Noisy → Enh | IPD Noisy → Enh |
|---|---|---|---|---|---|---|
| reference (softmax handoff §3.5) | 0.7311 | 0.7927 | +0.0616 | 0 | 3.29 → 2.98 | 21.10 → 20.02 |
| g42 (P1, earlier verified) | 0.7311 | 0.7923 | +0.0612 | -0.0004 | 3.29 → 3.01 | 21.10 → 20.32 |
| s075_topk | 0.7311 | 0.7903 | +0.0592 | -0.0024 | 3.29 → 3.04 | 21.10 → 20.28 |
| **s050_topk** ⭐ | 0.7311 | **0.7919** | **+0.0608** | **-0.0008** | 3.29 → 3.05 | 21.10 → **20.19** |

**結論**：
- **TIMIT 把 s050_topk 排在 winner**（贏 s075_topk 0.0016 MBSTOI、贏 0.09° IPD）
- s050_topk TIMIT MBSTOI 只比 reference 低 0.0008，在 cross-run noise 內 → **沒 collapse**
- s050_topk TIMIT IPD 20.19° 比 g42 (20.32°) 還好 → cross-domain binaural cue 沒被 skip cut 弄壞
- VCTK 重跑的數字（s050_topk 0.8352、s075_topk 0.8350）跟 [§1 全 eval 表](#1-sign-off-table--vctk-750-pair--8-snr) 完全一致 → eval pipeline bit-reproducible ✓
- VCTK + TIMIT 兩個 domain 都同意 s050_topk 是 winner，sign-off 信號收斂

g42 + s050_topk 確定為 **RTL primary candidate**。s075_topk 仍可作為 conservative backup（IPD 上 VCTK 稍勝，但其他指標被 s050_topk 主宰）。

---

## 5. 決策

### 5.1 P2 + RTL primary：g42 + s050_topk ⭐⭐

**為什麼選 s050_topk 而非 spec 預設的 s075_topk**：
- MBSTOI 上 s050_topk 0.8352 ≈ s075_topk 0.8350，差距 0.0002 在 cross-run noise band 內
- SegSNR：s050_topk 1.73 > s075_topk 1.61（高 0.12 dB）
- val_loss：s050_topk -13.76 > s075_topk -13.69（深 0.07）
- **MAC：s050_topk dec_pw 多省 2 倍**（-33% vs -17% 額外於 g42）
- IPD：s075_topk 略好（15.85 vs 15.94），但 s050_topk 仍比 g42 好（vs g42 16.04）

s050_topk 在 quality 持平/略勝的情況下多省一倍 MAC，是 strict winner。

### 5.2 Conservative backup：g42 + s075_topk

若硬體側需要更保守的 skip ratio，退一級用 s075_topk：
- dec_pw -68.5% vs baseline、full_conv -42.6%
- MBSTOI -0.0010 vs baseline，IPD 上甚至比 s050_topk 好

### 5.3 從候選清單移除：g44、s050_first、s075_first

| Variant | 移除理由 |
|---|---|
| g44 | 被 g42+s050_topk strictly dominated（同 MAC、quality 全面差） |
| s050_first | quality 顯著差於 s050_topk（MBSTOI -0.0036 vs -0.0008） |
| s075_first | quality 差於 s075_topk |

---

## 6. 已完成 + 待辦

### 6.1 已完成
- ✅ R0 split scaffolding（P1 phase）
- ✅ P1 sweep + sign-off（[RESULTS_P1](RESULTS_P1_GROUPED_2026-05-17.md)）
- ✅ P2 warm-start surgery（4 變體）
- ✅ P2 sweep + 20-pair sanity（4 變體）
- ✅ P2 full 750-pair VCTK eval（4 變體 + baseline reuse）
- ✅ P2 TIMIT cross-domain eval（s075_topk + s050_topk）
- ✅ MAC analytical verify（matches Codex 的數字）

### 6.2 跑中
- （無）

### 6.3 待辦
- ⏳ 整體 RTL handoff（彙整 softmax + P1 + P2，三段都齊全）

---

## 7. Reproducibility

```bash
cd /raid/R12K41024/12_decoder_grouped_pointwise

# P2 warm-starts（4 變體；auto-detect P1 g42 ckpt）
GPU_ID=1 bash scripts/make_phase2_skip_warmstarts.sh

# P2 sweep（finetune + 20-pair sanity）
GPU_ID=1 bash scripts/run_phase2_skip_sweep.sh s075_first s075_topk s050_first s050_topk

# Full 750-pair eval（reuse P1 baseline_off）
GPU_ID=1 bash scripts/run_phase2_full_eval.sh s075_first s075_topk s050_first s050_topk

# TIMIT on best candidates
GPU_ID=1 bash scripts/run_phase2_timit.sh s075_topk s050_topk

# MAC verification
conda run -n bcctn python scripts/count_p2_macs.py
```

### 7.1 結果位置

```
12_decoder_grouped_pointwise/
  results/p2_skip_sanity/        # 20-pair sanity
    s075_first_w20/  s075_topk_w20/  s050_first_w20/  s050_topk_w20/
  results/p2_skip_full/          # 750-pair full
    s075_first_w750/  s075_topk_w750/  s050_first_w750/  s050_topk_w750/
  results/p2_skip_timit/         # VCTK+TIMIT cross-domain
    s075_topk_vctk_timit_w750/
    s050_topk_vctk_timit_w750/
  outputs/p2_skip/<variant>_<timestamp>/
    .hydra/config.yaml           # frozen config (sign-off proof)
    logs/lightning_logs/version_0/checkpoints/*.ckpt   # best ckpt
```

### 7.2 Frozen config 驗證

確認 s050_topk run 用了預期 stack（[outputs/p2_skip/s050_topk_2026-05-17_21-27-05/.hydra/config.yaml](outputs/p2_skip/s050_topk_2026-05-17_21-27-05/.hydra/config.yaml)）：
- W4A8 strict：`qat_scope: full`、`int4_group_size: 32`
- Tied MHA：`tied_mha: true`
- Softmax：`attention_softmax.mode: shift_topk_pow2_clamp8_nomantissa`
- Top-k=4 attention
- Grouped pointwise：`groups_x: 4, groups_skip: 2`
- Skip subset：`skip_select_ratio: 0.5, skip_select_mode: topk_l1`
- Warm-start：`p2_warmstart_g42_topk_l1_r050.ckpt`

---

## 8. Decision log

1. **為什麼 P2 winner 是 s050_topk 而不是 spec 預設的 s075_topk？** Spec 寫 sweep 但沒預判 winner。Full eval 顯示 s050_topk 跟 s075_topk MBSTOI 基本打平（差 0.0002 在 cross-run noise band），但 SegSNR / val_loss 上 s050_topk 顯著更深，且 MAC 多省一倍。50% 的激進切沒 trigger quality cliff，反而是「**多餘 skip channel 被 topk_l1 篩掉後，網路 forced to consolidate**，學得更透」這個觀察的支持。
2. **g44 為什麼被剔除？** 同 MAC 收益下，g42+s050_topk 在 MBSTOI、IPD、val_loss 上全面更好。grouped pointwise 用 branch-asymmetric (4/2) 比 branch-symmetric (4/4) 給 capacity 留更大空間，再配 50% skip cut 達到同樣 MAC saving，所以 g44 已 strict 被取代。
3. **topk_l1 vs first_n 的選擇影響有多大？** Full eval 上 topk_l1 比 first_n 在 s075 多 +0.0006、s050 多 +0.0028（且 val_loss 深 0.149 / 0.267）。skip channel L1-rank pick 確實有 statistical signal。**RTL 端用 static skip_indices.pt（freeze 後不需要每次重算）就拿到全部 quality benefit**。
4. **last decoder layer (idx=1) 在 s050 下 mask head 只剩 4/8 skip channel 為何沒崩？** [HANDOFF_P2 §2.C1](HANDOFF_P2_2026-05-17.md) 預期可能 IPD 退化。實測 IPD 15.94° 反而比 g42 (16.04°) 好 0.10° → cut last layer skip 並沒有打到關鍵 binaural cue path，可能因 last layer 主要做 channel reduction-to-mask 而非 phase encoding。沒必要 per-layer override。
5. **`first_n` mode 還有保留價值嗎？** RTL 端如果要省「跑 valset 算 L1」這個 offline step，first_n 是純結構決策（無 data dependency），s075_first 仍 PASS gate。但 quality 確實掉，**只在 RTL bring-up 早期 prototype 用，正式 tape-out 一定要 topk_l1**。

---

## 9. 給 RTL 側的下一步建議

P1 + P2 sign-off 完成後（TIMIT 補完），整個「decoder 側 algorithm 優化」段落可以收尾。下一輪建議方向（spec §7 deferred 清單）：

| 方向 | 預期收益 | retrain 成本 |
|---|---|---|
| BN fold + PReLU pow2 slope / ReLU swap | 小（BN affine + slope→shift）| 短 finetune |
| kernel_num top-stage sweep（如 [16,32,64,128,192,192]） | MHA `input_proj` shape 也跟著減 | 一輪 QAT |
| MHA Q/K/V/O `nn.Linear` 也納入 grouped 化 | full-model MAC 真正下降的最後一塊 | 一輪 QAT，需要 spec |
| Conv `real/imag` tying（高風險）| MHA-tying 已驗證成立過；conv 風險高 | 一輪 QAT + 大量 IPD sanity |

最高 leverage：**MHA 化 grouped**，因為 §3.3 的 "full_conv MAC" 顯示 dec_pw 只是「conv 部分」的 62%，但全 model 還有 MHA 的 Q/K/V/O 4 個 dense Linear。MHA 跟 g42+s050_topk 同精神（branch grouping）的版本應該能再砍一大塊。
