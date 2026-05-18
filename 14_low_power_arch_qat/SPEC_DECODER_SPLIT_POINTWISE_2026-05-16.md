# SPEC: Decoder Grouped Pointwise (+ Skip Reduction)

日期：2026-05-16（v2：把 exact_split 從 phase 降為 scaffolding，重心放在 grouped pointwise）
作者：planner (Claude)
實作對象：Codex (or any implementor)
源頭 baseline：`11_weight_tied_mha_fast_mbstoi/`（strict W4A8 + top-k=4 + tied MHA + shift_topk_pow2_clamp8_nomantissa）
目標資料夾：**`12_decoder_grouped_pointwise/`**（從 `11_*` 完整複製；僅在新資料夾改 code，不動 11_）

---

## 0. TL;DR

打 decoder 第一個 hotspot：cat 後 pointwise 1×1。
**重心是 algorithm 改動（grouped + skip subset），不是 refactor**。

| Stage | 名稱 | 性質 | Retrain | 預期收益 |
|---|---|---|---|---|
| **R0** | RTL/dataflow equivalence note + scaffolding | 純重構 / unit test | 無 | 為後續 phase 鋪路；本身不是實驗 |
| **P1** | `decoder_grouped_pointwise` （branch-asymmetric groups）| 改 compute graph | 30-ep finetune | pointwise eligible-layer MAC 降 g 倍 |
| **P2** | `skip_select_ratio`（static subset；先用 `topk_l1`）| 砍 skip channel | 30-ep finetune | skip activation bandwidth 降 + decoder pointwise input 同時降 |

**deferred、不在本 spec：** conv_r/conv_i tying、ConvTranspose→upsample、kernel_num top-stage sweep、learned 1×1 skip bottleneck、`learned_gate` skip selection、`channel_shuffle`、BN fold、PReLU pow2、ReLU swap。每一條都另開 spec。

---

## 1. 背景

詳見 [RTL_HANDOFF_SOFTMAX_2026-05-14.md](RTL_HANDOFF_SOFTMAX_2026-05-14.md) 為 softmax 替換結果。本 spec 是 follow-up，處理 encoder/decoder 側剩餘的 conv 算術。

關鍵 facts（已驗證）：

1. Decoder forward 在 [model.py:222-226](DCNN/models/model.py#L222-L226) 對每層做 `torch.cat([x, encoder_skip], dim=1)`，cat 後第一個 op 是 pointwise 1×1。
2. Decoder block：[ComplexDepthwiseSeparableConvTranspose2d](DCNN/utils/complexPyTorch/complexLayers.py#L1118) = `pointwise(1×1) → depthwise_transpose(5×1) → BN → PReLU`。
3. **per-layer channel 表**（下表是 PyTorch complex tensor 的 channel 軸；real/imag 不佔用 channel 軸，而是 complex dtype 的兩個分量）。實際 `DCNN` 內部會把 `config.model.kernel_num=[16,32,64,128,256,256]` prepend 成 `[2,16,32,64,128,256,256]` 再傳給 `Decoder`，所以 decoder 有 6 個 block，最後一層輸出 mask head channel = 1。

| decoder iter k | idx | in_x | in_skip | cat in | out | g≤? OK |
|---|---|---|---|---|---|---|
| 0 | 6 | 128 | 128 | 256 | 128 | 128 |
| 1 | 5 | 128 | 128 | 256 | 64 | 64 |
| 2 | 4 | 64 | 64 | 128 | 32 | 32 |
| 3 | 3 | 32 | 32 | 64 | 16 | 16 |
| 4 | 2 | 16 | 16 | 32 | 8 | 8 |
| 5 | 1 | 8 | 8 | 16 | 1 | 1 |

`in_x = in_skip = kernel_num_internal[idx]//2`，`cat = kernel_num_internal[idx]`，`out = kernel_num_internal[idx-1]//2`。`g≤?` 是該層允許的 `max groups`（限制 = `gcd(in_per_branch, out)`）。current P1 sweep 的 `g≤4` 對前五層有效；最後一層因 `out=1` 永遠 fallback 到 `groups=1`。

4. BN/PReLU 目前不被 QAT 動到 ([fp8_qat.py:132-133](DCNN/utils/fp8_qat.py#L132-L133))；本 spec 也不動它們，留給後續獨立 axis 的 PR。

---

## 2. 倉庫設定

### 2.1 新資料夾

```bash
cp -r /raid/R12K41024/11_weight_tied_mha_fast_mbstoi /raid/R12K41024/12_decoder_grouped_pointwise
cd /raid/R12K41024/12_decoder_grouped_pointwise
```

所有後續編輯只動 `12_decoder_grouped_pointwise/`，不修改 11_。

### 2.2 Config flag

於 `12_decoder_grouped_pointwise/config/config.yaml` 的 `model:` block 新增：

```yaml
  # Decoder pointwise refactor (this spec)
  # Stages:
  #   off          — original cat([x, skip]) pointwise path (legacy)
  #   split        — scaffolding only: bit-equivalent split (Wx*x + Ws*skip), groups=1
  #   grouped      — split + per-branch grouped pointwise (P1)
  decoder_grouped_pointwise:
    mode: "off"
    groups_x: 1            # global; clipped per-layer to gcd(in_x, out)
    groups_skip: 1         # global; clipped per-layer to gcd(in_skip, out)
    # Optional: not used in P1 (will likely break warm-start init)
    channel_shuffle: false
    # Skip subset selection (P2)
    skip_select_ratio: 1.0     # ∈ (0, 1]; static subset
    skip_select_mode: "topk_l1"   # {"topk_l1", "first_n"}; learned_gate deferred
    skip_indices_path: null    # required when skip_select_ratio < 1.0 and mode=topk_l1
```

---

## 3. R0 — Scaffolding（不是 phase，只是 RTL/dataflow note + 入口程式）

> **本節只負責「讓 grouped 與 skip-subset 有可改的入口」與「驗證 channel split 順序正確」。不做 full eval、不做 20-pair sign-off、不出 RESULTS_R0 文件。**

### 3.1 RTL/dataflow equivalence note

```text
pointwise(cat([x, skip], dim=1)) == pointwise_x(x) + pointwise_skip(skip)

where:
  pointwise_x.weight   = pointwise.weight[:, :C_x, :, :]
  pointwise_skip.weight = pointwise.weight[:, C_x:, :, :]
  C_x = C_skip = kernel_num_internal[idx] // 2
```

複數情況下對 `conv_r` 與 `conv_i` 各做一次同樣的切片。bias 在現行實作 `pointwise.*.bias=False`，不需處理（bias 留在 depthwise）。

### 3.2 程式改動

**檔 A：[DCNN/utils/complexPyTorch/complexLayers.py](DCNN/utils/complexPyTorch/complexLayers.py)**

新增類別 `ComplexDepthwiseSeparableConvTranspose2dSplit`，支援 `groups_x` / `groups_skip`：

```python
class ComplexDepthwiseSeparableConvTranspose2dSplit(Module):
    """
    forward(x, skip) ==
        ComplexDepthwiseSeparableConvTranspose2d.forward(cat([x, skip], dim=1))
    when groups_x = groups_skip = 1 and weights initialized from a baseline ckpt
    by splitting pointwise.weight along input-channel dim.

    Args:
        in_x_channels   : x branch real-format input channel count
        in_skip_channels: skip branch real-format input channel count
        out_channels    : real-format output channel count
        groups_x        : groups for pointwise_*_x   (assert in_x %g==0  and out %g==0)
        groups_skip     : groups for pointwise_*_skip
    """
    def __init__(self, in_x_channels, in_skip_channels, out_channels,
                 kernel_size, stride=1, padding=0, output_padding=0,
                 dilation=1, bias=True,
                 groups_x=1, groups_skip=1):
        super().__init__()
        assert in_x_channels    % groups_x    == 0 and out_channels % groups_x    == 0
        assert in_skip_channels % groups_skip == 0 and out_channels % groups_skip == 0

        self.pointwise_r_x    = Conv2d(in_x_channels,    out_channels, kernel_size=1,
                                       bias=False, groups=groups_x)
        self.pointwise_i_x    = Conv2d(in_x_channels,    out_channels, kernel_size=1,
                                       bias=False, groups=groups_x)
        self.pointwise_r_skip = Conv2d(in_skip_channels, out_channels, kernel_size=1,
                                       bias=False, groups=groups_skip)
        self.pointwise_i_skip = Conv2d(in_skip_channels, out_channels, kernel_size=1,
                                       bias=False, groups=groups_skip)

        self.depthwise_r = ConvTranspose2d(out_channels, out_channels, kernel_size, stride,
                                           padding, output_padding, groups=out_channels,
                                           bias=bias, dilation=dilation)
        self.depthwise_i = ConvTranspose2d(out_channels, out_channels, kernel_size, stride,
                                           padding, output_padding, groups=out_channels,
                                           bias=bias, dilation=dilation)

    def forward(self, x, skip):
        pw_x    = apply_complex(self.pointwise_r_x,    self.pointwise_i_x,    x)
        pw_skip = apply_complex(self.pointwise_r_skip, self.pointwise_i_skip, skip)
        pw_out  = pw_x + pw_skip
        out     = apply_complex(self.depthwise_r, self.depthwise_i, pw_out)
        return out
```

不要動原 `ComplexDepthwiseSeparableConvTranspose2d`，保留作為 `mode=off` 路徑與 ground truth。

**檔 B：[DCNN/models/model.py](DCNN/models/model.py)** — `Decoder` class

- `__init__` 讀 `decoder_grouped_pointwise.mode`：
  - `"off"`：行為與目前完全相同（保留 legacy path）
  - `"split"` / `"grouped"`：每層用 `ComplexDepthwiseSeparableConvTranspose2dSplit`
- 每層的 `groups_x_eff = min(global_groups_x, gcd(in_x, out))`，`groups_skip` 同理。**這是必要的 per-layer fallback**，避免 `g=4` sweep 在 future kernel_num 改動時炸掉（current kernel_num 所有層都支援 `g ≤ 4`，沒衝突；但 `g=8` sweep 會在 idx=4 fallback 成 4）。print 每層生效的 groups 方便 debug。
- 因 split 版 forward 需要兩個輸入，把 BN / PReLU 從 `nn.Sequential` 拆出，自定 `_DecoderBlockSplit` wrapper：
  ```python
  class _DecoderBlockSplit(nn.Module):
      def __init__(self, split_conv, bn=None, prelu=None):
          super().__init__()
          self.split_conv = split_conv
          self.bn = bn
          self.prelu = prelu

      def forward(self, x, skip):
          x = self.split_conv(x, skip)
          if self.bn   is not None: x = self.bn(x)
          if self.prelu is not None: x = self.prelu(x)
          return x
  ```
- `Decoder.forward`：
  ```python
  def forward(self, x, encoder_out):
      for idx, block in enumerate(self.model):
          skip = encoder_out[-1 - idx]
          if self.mode != "off":
              x = block(x, skip)
          else:
              x = torch.cat([x, skip], 1)
              x = block(x)
      return x
  ```

**檔 C：`scripts/convert_ckpt_to_grouped.py`**（新建）

兩個動作合一：split surgery + grouped block-diagonal warm-start。

1. **Split surgery**（`groups_x=1, groups_skip=1` 時為止）：
   - 載入 baseline ckpt（`epoch=45-step=44528.ckpt`）
   - 對每個 decoder layer 的 `pointwise_r.weight [Out, In_x+In_skip, 1, 1]` 沿 dim=1 切兩段 `[:, :In_x]` 與 `[:, In_x:]`，寫入新 key `pointwise_r_x.weight` 與 `pointwise_r_skip.weight`
   - `pointwise_i.weight` 同理
   - 其他 key（depthwise / BN / PReLU / encoder / MHA）原樣 copy

2. **Grouped warm-start init**（`groups > 1` 時）：
   - 起點是上一步 split 後的 `pointwise_*_x.weight`，shape `[Out, In_x, 1, 1]`
   - 對 `groups = g`：
     - reshape 成 `[g, Out/g, g, In_x/g, 1, 1]`
     - 取對角 block：`new_w[gi, :, :, :, :, :] = old_w[gi, :, gi, :, :, :]`
     - reshape 回 `[Out, In_x/g, 1, 1]`（注意 PyTorch grouped Conv2d 的 weight shape 是 `[Out, In/groups, kH, kW]`）
   - `pointwise_*_skip` 同理
   - 物理意義：假設「跨 group cross-channel coupling 一開始忽略」，比 random init 收斂快很多。**這是 P1 finetune 能在 30 epoch 內收斂的關鍵**。

cmd 介面：
```bash
python scripts/convert_ckpt_to_grouped.py \
    --in       <baseline.ckpt> \
    --out      <split_or_grouped.ckpt> \
    --groups_x 2 --groups_skip 2
# groups_x = groups_skip = 1 等於純 split surgery
```

### 3.3 R0 Sanity check（不算 sign-off，只是 channel-order 正確性）

**檔 D：`verify_decoder_split_equivalence.py`**（新建）

只跑一次，確保 `groups_x = groups_skip = 1` 模式下與原 forward bit-equivalent：
1. 建 dense model（`mode="off"`）與 split model（`mode="split", groups=1`），分別 load 對應 ckpt。
2. 對 4 個 random batch + 2 個真實 audio batch 跑 forward。
3. 印出每層 max abs diff，要求 final wav max diff < 1e-5。
4. 印 `[decoder_split exact, groups=1] PASS` / `FAIL`。

PASS 即可往 P1 走，**不需 20-pair MBSTOI**（bit-equivalent 沒有跑 metric 的意義）。

---

## 4. P1 — Decoder Grouped Pointwise（branch-asymmetric）

**目標**：在 split 入口下，分別對 `pointwise_x` 與 `pointwise_skip` 套 `groups`，sweep quality vs MAC trade-off。

### 4.1 Sweep matrix

每個變體都從 R0 的 split ckpt + grouped warm-start 開始 finetune。

| 變體 | groups_x | groups_skip | 備註 |
|---|---|---|---|
| `g11` | 1 | 1 | == split exact，作為 finetune-noise reference（量化「不改 graph」純 finetune 漂動量）|
| `g21` | 2 | 1 | x 砍半，skip 全保留 |
| `g12` | 1 | 2 | skip 砍半，x 全保留（對比 g21，看哪邊更敏感）|
| `g22` | 2 | 2 | 對稱 g=2 |
| `g42` | 4 | 2 | x 更激進 |
| `g44` | 4 | 4 | 對稱 g=4 |

**`channel_shuffle = false`**（per §0 deferred 與 §2.2 註解）。

### 4.2 Finetune setting

每變體：
- start ckpt：R0 surgery + warm-start 對應 `groups_x/skip`
- `training.n_epochs = 30`
- `training.learning_rate = 5e-4`
- `training.qat_scope = full`
- `training.top_k_attention = 4`
- `model.attention_softmax.mode = shift_topk_pow2_clamp8_nomantissa`
- `model.decoder_grouped_pointwise.mode = grouped`
- `model.decoder_grouped_pointwise.groups_x / groups_skip` 對應

寫成 `scripts/run_phase1_grouped_sweep.sh`，依序跑所有 variant。

### 4.3 Sign-off

對每個變體跑 §6 eval 模板：
- VCTK 750-pair × 8 SNR：MBSTOI 不退化 > 0.005 vs baseline 0.8360
- ILD/IPD 不退化 > 5%
- TIMIT cross-domain：對 **best 1-2 個變體**跑（不是全部都跑，TIMIT 在 RTL 候選確定後再跑）

結果寫進 `RESULTS_P1_GROUPED_YYYY-MM-DD.md`，照 [RTL_HANDOFF_SOFTMAX_2026-05-14.md §3.1](RTL_HANDOFF_SOFTMAX_2026-05-14.md#L101) 表格格式。

### 4.4 MAC 收益 reporting

不要只說「pointwise 降 g 倍」。請報「decoder pointwise eligible layer 的 MAC 降幅」：
- 算每層原本 pointwise MAC：`C_in × C_out × F × T`
- 算 grouped 後 MAC：`(C_in/g) × C_out × F × T`（注意 PyTorch grouped Conv2d 算法）
- 加總所有 decoder pointwise 層的「`pointwise_x + pointwise_skip`」總 MAC，跟 baseline 比
- depthwise / BN / PReLU / activation movement 不變，不要算進去

報表中分開列「全 model MAC 降幅」與「decoder pointwise MAC 降幅」兩個數字，避免被誤讀。

---

## 5. P2 — Skip Select Ratio

**前提**：P1 已選出 best `(groups_x, groups_skip)`；P2 在此基礎上加 skip channel subset。

### 5.1 兩種 select mode（先做 static，避開 learned_gate）

| mode | 行為 | 何時 freeze |
|---|---|---|
| `first_n` | 取前 `round(C_skip * skip_select_ratio)` 個 channel | inference-time（sanity check 用，不期望好）|
| `topk_l1` | 從 baseline encoder 對 validation set 跑一次，每層 encoder output 每 channel 的 L1 norm 排序，取 top-k channel index，**固定下來**寫入 `skip_indices.pt` | pre-compute 一次，後續 freeze |

**`learned_gate` deferred**：複雜度高、training graph 不必然省 compute，先驗證「skip 真的有冗餘」再決定要不要這個機制。

### 5.2 程式

於 `Decoder.forward`：
```python
skip = encoder_out[-1 - idx]
if self.skip_select_ratio < 1.0:
    skip = skip[:, self.skip_indices[idx]]  # pre-computed
```
`split_conv` 的 `in_skip_channels` 也對應改成 `C_skip_keep`，所以 P2 變體必須重新 build 模型（不能共用 P1 ckpt 直接 load，要走 surgery）。

P2 implementation note：在 grouped P1 candidate 上，skip index 必須 **group-aware**，每個 skip group 保留相同數量的 channel，否則會破壞 `groups_skip` 的 warm-start shape。以 g42 為例：
- `s075` keep = `[96, 96, 48, 24, 12, 6]`
- `s050` keep = `[64, 64, 32, 16, 8, 4]`

Scripts：
- `scripts/precompute_skip_indices.py`：用 P1 best ckpt 跑 valset，對每層 encoder output 沿 (B, F, T) 取 L1 mean，group-wise top-k，存 `skip_indices.pt`。
- `scripts/convert_ckpt_to_skip_subset.py`：從 P1 split/grouped ckpt 依 fixed indices 裁 `pointwise_{r,i}_skip` weights，產 P2 warm-start ckpt。
- `scripts/make_phase2_skip_warmstarts.sh`：批次產生 P2 indices + warm-start ckpt。
- `scripts/run_phase2_skip_sweep.sh`：train + 20-pair sanity。
- `scripts/run_phase2_full_eval.sh` / `scripts/run_phase2_timit.sh`：full VCTK / TIMIT eval。

### 5.3 Sweep

| 變體 | mode | ratio | groups |
|---|---|---|---|
| `s100` | -- | 1.0 | P1 best (reference) |
| `s075_first` | first_n | 0.75 | P1 best |
| `s075_topk` | topk_l1 | 0.75 | P1 best |
| `s050_first` | first_n | 0.5 | P1 best（sanity，可選跑）|
| `s050_topk` | topk_l1 | 0.5 | P1 best |

### 5.4 Sign-off

同 P1。寫進 `RESULTS_P2_SKIP_YYYY-MM-DD.md`。

---

## 6. Eval / Verify 模板

### 6.1 20-pair sanity（每個 train 結束都跑）

```bash
CUDA_VISIBLE_DEVICES=0 METRIC_WORKERS=64 conda run -n bcctn python eval.py \
    --model_checkpoint <variant_ckpt> \
    --vctk_test_dir  /raid/R12K41024/BCCTN/Dataset/noisy_testset \
    --vctk_clean_dir /raid/R12K41024/BCCTN/Dataset/clean_testset \
    --output_dir results/sanity_<variant>_w20 \
    --paper_style_eval --batch_size 8 --limit_pairs 20 --metric_workers 64 \
    --hydra_overrides model.decoder_grouped_pointwise.mode=grouped \
                      model.decoder_grouped_pointwise.groups_x=<gx> \
                      model.decoder_grouped_pointwise.groups_skip=<gs> \
                      model.attention_softmax.mode=shift_topk_pow2_clamp8_nomantissa \
                      training.qat_scope=full training.top_k_attention=4
```

### 6.2 Full VCTK 750-pair

同 6.1，去掉 `--limit_pairs`，`--output_dir` 改 `results/<variant>_w64`。

### 6.3 TIMIT cross-domain（只對 best 1-2 candidates 跑）

仿照 [RTL_HANDOFF_SOFTMAX_2026-05-14.md §6.3](RTL_HANDOFF_SOFTMAX_2026-05-14.md#L407)，加上 grouped/skip 對應 hydra_overrides。

### 6.4 預估時間（224-core / 1 H100）

- 30-epoch QAT finetune：~3-4 小時
- VCTK 750-pair eval：~6 分鐘
- TIMIT + VCTK：~17 分鐘
- P1 (6 variants finetune + eval)：~24 小時 + 1 小時 eval
- P2 (5 variants)：~20 小時 + 50 分 eval
- TIMIT for 2 best candidates：~35 分鐘

---

## 7. 不要做（deferred，獨立 spec）

- `conv_r ≡ conv_i` real/imag tying（IPD 風險）
- ConvTranspose → upsample + separable conv
- Encoder-decoder mirror weight sharing
- **kernel_num top-stage sweep**（會動 MHA `input_proj` shape；另開 spec + checkpoint surgery）
- **Learned 1×1 skip bottleneck**（待 P2 驗證 skip 真有冗餘再決定）
- **`learned_gate` skip selection**
- **`channel_shuffle`**（會破壞 grouped warm-start init）
- **BN fold / PReLU pow2 / ReLU swap**（另一條 axis）

---

## 8. 文件 / commit 規範

- 每 phase 結束在新資料夾根目錄產一個 `RESULTS_P<n>_<name>_YYYY-MM-DD.md`，照 [RTL_HANDOFF_SOFTMAX_2026-05-14.md §3.1](RTL_HANDOFF_SOFTMAX_2026-05-14.md#L101) 表格格式。
- R0 完成可單獨 commit；P1/P2 各自 commit。
- commit msg 前綴 `[12_decoder_grouped]`。
- 兩 phase 結論彙整為 `RTL_HANDOFF_DECODER_YYYY-MM-DD.md`，作為 RTL 端下一個交付。

---

## 9. 風險與回退

| 風險 | 偵測 | 回退 |
|---|---|---|
| R0 split 出現 numerical drift | `verify_decoder_split_equivalence.py` max diff > 1e-5 | 檢查 channel 切片順序：cat 是 `[x, skip]` → `[:, :C_x]` 是 x、`[:, C_x:]` 是 skip。`C_x = kernel_num_internal[idx]//2` |
| P1 `groups=4` ILD 明顯退化 | RESULTS 表 ILD Enhanced > 3.5 dB | 鎖 `groups ≤ 2`；或只在 deepest 2 層用 `groups>1`（per-layer override，需擴 config）|
| P1 warm-start 仍收斂慢 | 30 epoch val_loss 沒下到接近 baseline | 拉到 50 epoch；或 warm-start 換成「整個 group 取 mean」而非「對角 block」|
| P2 `topk_l1` 沒 baseline 好 | MBSTOI 比 P1 best 掉 > 0.005 | 退回 `ratio=1.0`；不採用 P2 |
| TIMIT 退化 > VCTK | TIMIT MBSTOI Δ > 0.01 | 該變體標 ❌；考慮回退到上一個保守變體 |

---

## 10. 驗收 checklist

- [ ] `12_decoder_grouped_pointwise/` 從 `11_*` 完整複製，11_ 沒被改
- [ ] R0：`verify_decoder_split_equivalence.py` PASS，final wav max diff < 1e-5
- [ ] R0：**不跑** 20-pair sign-off（bit-equivalent，無意義）
- [ ] P1：6 個 variant 全跑、`RESULTS_P1_GROUPED` 寫齊（含 decoder pointwise MAC 降幅單獨欄）
- [ ] P1：best 1-2 個 candidate 跑 TIMIT
- [ ] P2：skip indices pre-compute 完成；5 個 variant 全跑、`RESULTS_P2_SKIP` 寫齊
- [ ] P2：best candidate 跑 TIMIT
- [ ] 最終 `RTL_HANDOFF_DECODER_*.md` 整理出一個 RTL primary candidate `(groups_x, groups_skip, skip_select_ratio)` 配對
