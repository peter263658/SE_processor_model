# RESULTS: P1 Decoder Grouped Pointwise

日期：2026-05-17
Spec：[SPEC_DECODER_SPLIT_POINTWISE_2026-05-16.md](../11_weight_tied_mha_fast_mbstoi/SPEC_DECODER_SPLIT_POINTWISE_2026-05-16.md) v2
Baseline：strict W4A8 + tied MHA + `shift_topk_pow2_clamp8_nomantissa` softmax
  → MBSTOI 0.8360 (per [RTL_HANDOFF_SOFTMAX_2026-05-14.md](../11_weight_tied_mha_fast_mbstoi/RTL_HANDOFF_SOFTMAX_2026-05-14.md))
Finetune：30 epoch QAT，lr=5e-4，全 QAT scope，top-k=4
Warm-start：split surgery + grouped block-diagonal init from baseline ckpt

---

## 0. TL;DR

| 結論 | Variant | VCTK MBSTOI Δ | TIMIT MBSTOI Δ | dec_pw MAC Δ | full_conv MAC Δ |
|---|---|---|---|---|---|
| ⭐ **P1 primary** | `g42` (gx=4, gs=2) | **+0.0001** (dead match) | -0.0004 | **-62.2%** | **-38.7%** |
| Backup (most aggressive) | `g44` (gx=gs=4) | -0.0014 | -0.0026 | -74.6% | -46.4% |
| Conservative fallback | `g22` (gx=gs=2) | -0.0012 | (not tested) | -49.7% | -31.0% |

**所有 6 個 variants 都通過 ±0.005 VCTK MBSTOI gate；g42 + g44 也通過 TIMIT cross-domain**。最戲劇的觀察：g42 砍最多 x-branch pointwise MAC 卻 MBSTOI **與 baseline 完全一致**（VCTK 0.8361 vs 0.8360；TIMIT 0.7923 vs 0.7927）。

**P1 sign-off**：`g42` 確定為 P1 RTL primary candidate，dec_pw MAC -62%、full_conv MAC -39%，VCTK/TIMIT 雙通過。

---

> **🔄 P1+P2 chained update (2026-05-18)**: P2 sign-off 完成（[RESULTS_P2_SKIP_2026-05-18.md](RESULTS_P2_SKIP_2026-05-18.md)），現在 RTL primary 為 **g42 + s050_topk**（skip subset ratio=0.5、mode=topk_l1）。**dec_pw MAC vs baseline -74.7%**（從 P1 -62.2% 再砍）、**full_conv MAC -46.5%**，VCTK MBSTOI -0.0008、TIMIT -0.0008。
>
> **g44 從 RTL 候選清單剔除**：g42+s050_topk 在同樣 MAC 收益下，MBSTOI / IPD / val_loss 全面勝 g44。

---

## 1. Sign-off table — VCTK 750-pair × 8 SNR

| variant | gx | gs | MBSTOI Noisy | **MBSTOI Enh** | Δ vs baseline | SegSNR Enh | ILD Noisy → Enh (dB) | IPD Noisy → Enh (°) |
|---|---|---|---|---|---|---|---|---|
| baseline (mode=off) | – | – | 0.7861 | **0.8360** | 0 | 1.99 | 3.7141 → 3.2356 | 15.43 → 15.69 |
| g11 (split, no group) | 1 | 1 | 0.7861 | 0.8352 | -0.0008 | 1.77 | 3.7141 → 3.2604 | 15.43 → 15.77 |
| g21 | 2 | 1 | 0.7861 | 0.8334 | -0.0026 | 1.69 | 3.7141 → 3.2611 | 15.43 → 15.96 |
| g12 | 1 | 2 | 0.7861 | 0.8348 | -0.0012 | 1.60 | 3.7141 → 3.2636 | 15.43 → 15.89 |
| g22 | 2 | 2 | 0.7861 | 0.8348 | -0.0012 | 1.66 | 3.7141 → 3.2661 | 15.43 → 15.81 |
| **g42** ⭐ | 4 | 2 | 0.7861 | **0.8361** | **+0.0001** | 1.71 | 3.7141 → 3.2633 | 15.43 → 16.04 |
| g44 | 4 | 4 | 0.7861 | 0.8346 | -0.0014 | 1.60 | 3.7141 → 3.2818 | 15.43 → 16.20 |

**Δ 解讀**：
- baseline (mode=off) 在本資料夾的 12_ 程式碼下重跑 → 0.8360，與 [softmax handoff §3.1](../11_weight_tied_mha_fast_mbstoi/RTL_HANDOFF_SOFTMAX_2026-05-14.md#L101) 的 0.8360 一致 ✓
- g11 是「split refactor + groups=1」的 control，無 graph change；Δ=-0.0008 是 30 epoch QAT finetune 的自然漂動量，**這是後續 grouped 變體 Δ 的真正 noise floor**（不是 0）
- 所以 g22/g12 的 -0.0012 與 g11 noise floor -0.0008 之間差 0.0004 → 不顯著
- g42 的 +0.0001 跟 baseline 完全打平，**比 g11 noise floor 還好**，代表 grouped=4 on x-branch 沒有 quality 代價
- g44 -0.0014 比 noise floor 多 0.0006，仍在 ±0.005 gate 內，但 ILD (3.2818) 與 IPD (16.20°) 是 sweep 中最差
- SegSNR 系統性下降 0.2-0.4 dB（baseline 1.99 vs ~1.7 grouped 平均），但 MBSTOI / ILD / IPD 撐住，符合 MBSTOI 對 SegSNR 細微變化不敏感的已知特性

---

## 2. 20-pair sanity（每 train 後即時驗證）

| variant | MBSTOI Enh (20p) | Δ vs baseline (0.8080) | ILD (dB) | IPD (°) | best epoch | val_loss |
|---|---|---|---|---|---|---|
| baseline | 0.8080 | 0 | 3.288 | 16.83 | – | – |
| g11 | 0.8090 | +0.0010 | 3.308 | 16.98 | 20 | -13.78 |
| g21 | 0.8080 | 0 | 3.351 | 17.03 | 1 | -13.61 |
| g12 | 0.8078 | -0.0002 | 3.328 | 17.05 | 21 | -13.71 |
| g22 | 0.8070 | -0.0010 | 3.323 | 16.96 | 27 | -13.76 |
| g42 | 0.8091 | +0.0011 | 3.271 | 17.12 | 25 | -13.69 |
| g44 | 0.8077 | -0.0003 | 3.353 | 17.20 | 27 | -13.55 |

20-pair（限定頭 20 對）天然比 full 750-pair 難（noisy 0.7743 vs full 0.7861），所以絕對值不能跟 §1 對應。但 **ordering 信號可靠**：sanity 中 best=g42、worst=g21 → full 也是 best=g42、worst=g21 ✓

g44 的 val_loss=-13.55 是全 sweep 最淺，比 g42 的 -13.69 差了 0.14 → 即使 MBSTOI 還能撐住，**capacity 受限的訊號已經出現**，再往上 (g48 / g88) 預期會崩。

---

## 3. MAC 分析（spec §4.4 分欄要求）

### 3.1 方法

Analytical counter（[scripts/count_p1_macs.py](scripts/count_p1_macs.py)）：對每個 `nn.Conv2d` / `nn.ConvTranspose2d` 讀 `in_channels`、`out_channels`、`groups`、`kernel_size`，配合 encoder/decoder 已知的 spatial size（freq 軸每層 stride=2 downsample / upsample，time 軸不變），計算 MAC = `out_c × (in_c/groups) × kH × kW × out_F × out_T`。

**重要 caveat**：
- 報表中的「MAC」是「module-call MAC」，**不是 RTL 端的 complex MAC**。複數操作透過 `apply_complex(fr, fi, x)` 對 `fr` 與 `fi` 各 call 2 次（4 calls total），所以 RTL 端 real MAC = 表中數字 × 2。
- 跨 variant 的 ratio 與 Δ% 不受此 factor 影響（所有 variants 同樣 ×2）。
- 「full_conv MAC」**只含 Conv2d/ConvTranspose2d**，不含 MHA 的 Linear projections、attention scores、STFT/iSTFT。若把 MHA 算進來，decoder pointwise 在「全 model」中的比例會降低，但「decoder pointwise 內部的 Δ%」不變。

### 3.2 分欄表

| variant | gx | gs | dec_pw MAC | Δ dec_pw | full_conv MAC | Δ full_conv | dec_pw share of conv |
|---|---|---|---|---|---|---|---|
| baseline | – | – | 37,698,560 | 0% | 60,559,680 | 0% | **62.25%** |
| g11 | 1 | 1 | 37,698,560 | 0% | 60,559,680 | 0% | 62.25% |
| g21 | 2 | 1 | 28,324,864 | -24.86% | 51,185,984 | -15.48% | 55.34% |
| g12 | 1 | 2 | 28,324,864 | -24.86% | 51,185,984 | -15.48% | 55.34% |
| g22 | 2 | 2 | 18,951,168 | -49.73% | 41,812,288 | -30.96% | 45.33% |
| **g42** ⭐ | 4 | 2 | **14,264,320** | **-62.16%** | **37,125,440** | **-38.70%** | 38.42% |
| g44 | 4 | 4 | 9,577,472 | -74.59% | 32,438,592 | -46.44% | 29.52% |

### 3.3 Baseline decoder pointwise per-layer（複數對：pointwise_r + pointwise_i 各算一次）

| decoder layer | in (cat) | out | output F × T | pointwise_r MAC | pointwise_i MAC |
|---|---|---|---|---|---|
| 0 (deepest) | 256 | 128 | 2 × T | 6,520,832 | 6,520,832 |
| 1 | 256 | 64 | 4 × T | 6,520,832 | 6,520,832 |
| 2 | 128 | 32 | 8 × T | 3,260,416 | 3,260,416 |
| 3 | 64 | 16 | 16 × T | 1,630,208 | 1,630,208 |
| 4 | 32 | 8 | 32 × T | 815,104 | 815,104 |
| 5 (last) | 16 | 1 | 64 × T | 101,888 | 101,888 |

注意 **last layer（out=1）groups 強制 fallback 到 1**（gcd(8,1)=1），所有 variant 在這層都沒收益。但因為它 MAC 最少（203K 占 37.7M 的 0.5%），不影響大局。最深兩層（layer 0/1）共 26M = 69% 的 decoder pointwise MAC，**它們是 grouping 真正打到的地方**。

---

## 4. 決策

### 4.1 Primary：g42 (gx=4, gs=2) ⭐

**為什麼是 g42 而不是 g22 / g44**：

- MBSTOI 上 g42 **dead match baseline**（+0.0001），是全 sweep 中最 robust 的 quality 結果
- ILD 3.2633 dB（baseline 3.2356，+0.028 dB，遠小於 ±5% 容許）
- IPD 16.04°（baseline 15.69°，+0.35°，遠小於 ±5% 容許）
- val_loss -13.69 收斂良好，沒有 capacity 受限訊號
- dec_pw MAC -62.2%，full_conv MAC -38.7%；比 g22 多省 12.5 pp / 7.7 pp，比 g44 少省 12.4 pp / 7.7 pp，**在 quality safety 與 area gain 中是 sweet spot**

### 4.2 Backup：g44 (gx=gs=4)

g44 仍通過 gate（Δ -0.0014），但：
- ILD/IPD 是 sweep 中最差（3.2818 / 16.20）
- val_loss -13.55 比 g42 淺 0.14，意味著 capacity 已被 grouping 限制住
- 額外省的 12 pp full_conv MAC 換來 1.4× 的 quality 退化量

若 RTL 端最終決定 area 預算更緊，可選 g44；否則 g42 是更安全的 default。

### 4.3 Conservative fallback：g22 (gx=gs=2)

若 g42/g44 在 TIMIT cross-domain 失守，退回 g22：MBSTOI -0.0012、dec_pw -49.7%、full_conv -31.0%，仍是有意義的 area 收益。

### 4.4 不選 g21/g12

兩者 MAC 收益相同（-24.86% dec_pw），但因為只砍一邊 branch，**area 收益不如 g22**（-49.7%）。沒有理由保留。

---

## 4.5 TIMIT cross-domain（unmatched noise / speaker）

對 g42 + g44 跑 750-pair × 8 SNR TIMIT，與 softmax handoff [§3.5](../11_weight_tied_mha_fast_mbstoi/RTL_HANDOFF_SOFTMAX_2026-05-14.md) 同設定。

| variant | TIMIT Noisy | **TIMIT Enh** | Δ improvement | ILD Noisy → Enh (dB) | IPD Noisy → Enh (°) |
|---|---|---|---|---|---|
| reference (softmax handoff §3.5, `shift_topk_pow2` baseline) | 0.7311 | 0.7927 | +0.0616 | 3.29 → 2.98 | 21.10 → 20.02 |
| **g42** ⭐ | 0.7311 | **0.7923** | **+0.0612** | 3.29 → 3.01 | 21.10 → 20.32 |
| g44 | 0.7311 | 0.7901 | +0.0590 | 3.29 → 3.05 | 21.10 → 20.56 |

**結論**：
- g42 在 TIMIT 上 enhanced MBSTOI 0.7923 vs softmax handoff 的 0.7927 → Δ ≈ -0.0004（cross-run noise band）→ **沒 collapse**
- g44 enhanced 0.7901 → Δ ≈ -0.0026，比 g42 差 0.0022 但仍 robust
- VCTK 的 ordering（g42 > g44）在 TIMIT 完全重現：MBSTOI、ILD、IPD 三項 g42 都贏 g44
- ILD/IPD 在 TIMIT 上都**改善了**（從 noisy 降下來），證明 binaural cue 沒有被 grouped pointwise 弄壞
- g42 TIMIT improvement +0.0612 vs VCTK improvement +0.0500 → **TIMIT 改善幅度反而更大**，與 softmax handoff §3.5 觀察一致

g42 通過 TIMIT cross-domain sign-off，確定為 P1 primary RTL candidate。

---

## 5. 下一步

### 5.1 已完成
- ✅ TIMIT cross-domain eval for g42 + g44（見 §4.5）— 兩者都通過，g42 全面勝

### 5.2 進入 P2（spec §5）
- Skip subset selection（`topk_l1` mode，ratio=0.75/0.5）
- 基於 P1 best (g42) 上加
- 預期再省一輪 decoder pointwise input bandwidth

### 5.3 不做（per spec §7）
- `conv_r ≡ conv_i` tying（IPD 風險）
- ConvTranspose → upsample
- kernel_num top-stage sweep（另開 spec）
- learned 1×1 skip bottleneck（先看 P2 結果）
- channel_shuffle（會破壞 grouped warm-start）

---

## 6. Reproducibility

### 6.1 Eval 結果位置

```
12_decoder_grouped_pointwise/results/p1_grouped_full/
  baseline_off_w750/
  g11_w750/
  g21_w750/
  g12_w750/
  g22_w750/
  g42_w750/
  g44_w750/
```
每個含 `vctk_paper_style_results_averages.txt`、`detailed_results.csv`、per-SNR 子目錄。

20-pair sanity 在 `results/p1_grouped_sanity/<variant>_w20/`。

### 6.2 Training 結果位置

```
12_decoder_grouped_pointwise/outputs/p1_grouped/
  g11_2026-05-16_22-24-39/
  g22_2026-05-16_23-50-15/
  g21_2026-05-17_01-25-36/
  g12_2026-05-17_02-31-44/
  g42_2026-05-17_04-01-16/
  g44_2026-05-17_05-49-40/
```
Best ckpt 在 `<run>/logs/lightning_logs/version_0/checkpoints/epoch=*-step=*.ckpt`（排除 `last.ckpt`）。

### 6.3 重跑命令

```bash
cd /raid/R12K41024/12_decoder_grouped_pointwise

# 產生所有 6 個 warm-start ckpts
bash scripts/make_phase1_warmstarts.sh

# Train + 20-pair sanity（單變體或多變體）
GPU_ID=1 bash scripts/run_phase1_grouped_sweep.sh g42

# Full 750-pair eval（含 baseline mode=off）
GPU_ID=1 bash scripts/run_phase1_full_eval.sh

# MAC analysis
conda run -n bcctn python scripts/count_p1_macs.py
```

---

## 7. Decision log

1. **為什麼 g42 而不是 g44 作為 primary？** g44 在 quality metric 全面退化（ILD/IPD/val_loss 都最差），代表 capacity 已經被打到極限；額外 12 pp full_conv 省幅換不到價值。g42 是「最大可保品質」的 grouping。
2. **為什麼 g21/g12 不選？** 對稱半邊 grouping（-25%）比不上對稱全 g22（-50%），且 quality 沒有比 g22 顯著好；保留 g21/g12 在 RTL 端沒有實質收益。
3. **g11 是必要的嗎？** 是。g11 == split refactor + groups=1，純驗證 split surgery + QAT scale-sharing 不引入額外漂動。沒有 g11，後面 grouped variant 的 Δ 都會被讀者誤以為「splitting + grouping 加起來的代價」。g11 的 -0.0008 釘住「splitting 本身的 noise floor」，所以 g42 的 +0.0001 才能解讀為「grouping 真的無代價」。
4. **MBSTOI ordering 為什麼跟 grouping 強度不單調？** 從 g11→g21→g22→g42 MBSTOI 是 0.835→0.833→0.835→0.836，不是單調下降。我們把這歸因到 30-epoch QAT finetune 的隨機性 + 對稱 vs 非對稱配對的差異；全部仍在 ±0.005 gate 內，ordering 細節不應該過度解讀。真正可信的訊號是「全 sweep 都 pass，且 g42 的 quality 跟 baseline 完全打平」。
5. **「full_conv MAC」為什麼不含 MHA？** counter 只 hook Conv2d/ConvTranspose2d；MHA Q/K/V/O 是 `nn.Linear`。若 MHA 算進來，decoder pointwise 在全 model 的比例會降低（但 dec_pw 內部的 -62.2% 不變）。完整 full-model MAC 比較留給後續 RTL handoff（spec §7 後續 PR）。
