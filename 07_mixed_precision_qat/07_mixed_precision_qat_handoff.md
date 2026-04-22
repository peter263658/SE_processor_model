# 07 — Mixed-Precision W4A8 QAT Fine-tune 工作交接清單

**日期**：2026-04-17  
**交接人**：peter263658  
**對應資料夾**：待建立（建議命名 `BCCTN_sep_conv_causal_slide_win_mixed_precision_256dim/`）  
**Pre-trained checkpoint**：`BCCTN_sep_conv_causal_slide_win_quant_256dim/outputs/` 下最新的 epoch=134 checkpoint

---

## 一、背景摘要

已完成 6 個硬體感知分析（01–06），以下是跨報告交叉比對後的核心結論：

| 結論 | 來源報告 |
|------|---------|
| MHA 佔 77.6% MACs，是全模型最大瓶頸 | 01 |
| Uniform INT4 品質損失 −0.56 dB（可接受），DMA 減半 | 02 |
| 5 個 decoder PW conv 有 outlier（kurtosis > 5），需保留 FP8 | 03 |
| MHA 9/12 head-group weight 不均勻，需 per-head scaling | 03 |
| Weight sparsity 全為隨機非結構化，硬體投入無效益 | 04 |
| Cross-layer weight sharing 無機會（cos sim < 0.6） | 05 |
| MHA DMA 640 KB 是搬運瓶頸（比所有 conv 合計大 4×） | 05 |
| Top-k=4 稀疏注意力品質幾乎無損，計算量省 87% | 06 |
| QK^T 累加器動態範圍 1392，需 fp16（不能用 fp8/int8） | 06 |
| 現有 FP8 QAT 中間運算在 FP32，硬體部署需降至 fp16 | 06 |

---

## 二、本次 QAT 的目標

從現有 FP8 QAT pre-trained model 做 fine-tune，實現以下混合精度配置：

### 2.1 目標精度配置表

| 模組 | Weight 精度 | Activation 精度 | 累加器精度 | Scale 粒度 | 備註 |
|------|-----------|----------------|-----------|-----------|------|
| Encoder DW conv (×12) | **INT4** | FP8 | fp16 | per-channel | Cout 少，per-channel 必要 |
| Encoder PW conv (×12) | **INT4** | FP8 | fp16 | per-group (gsz=256) | 正常分布 |
| Decoder DW conv (×12) | **INT4** | FP8 | fp16 | per-channel | 同 encoder |
| Decoder PW conv stage 0,3,4 (×5) | **FP8 保留** | FP8 | fp16 | per-channel | kurtosis > 5，有 outlier |
| Decoder PW conv 其他 (×7) | **INT4** | FP8 | fp16 | per-group (gsz=256) | 正常分布 |
| MHA Wq/Wk/Wv/Wo (re+im) | **INT4** | FP8 | fp16 | **per-head** | 9/12 head-group 非均勻 |
| MHA input_proj/output_proj | **INT4** | FP8 | fp16 | per-head | 保守起見用 per-head |
| **QK^T 累加** | — | — | **fp16** | — | max=911 > fp8 max=448 |
| **Softmax** | — | fp16 | — | — | [0,1] 範圍，fp16 足夠 |
| **Score×V** | — | fp16 | fp16 | — | 稍保守 |
| **Top-k masking** | — | — | — | — | k=4, softmax 後歸零+重歸一化 |
| BN / PReLU | FP32 保留 | — | — | — | 參數極少，不量化 |

### 2.2 目標品質指標（500 對 VCTK 測試集）

| 指標 | FP8 Baseline | 目標（INT4 QAT 後） | 容忍閾值 |
|------|-------------|-------------------|---------|
| SI-SNR | 10.492 dB | ≥ 10.0 dB | ΔSI-SNR < 0.5 dB |
| ILD error | 4.006 dB | ≤ 4.5 dB | ΔILD < 0.5 dB |
| IPD error | 23.97° | ≤ 29.0° | ΔIPD < 5° |

---

## 三、實作步驟（依序）

### Step 0：建立新資料夾

```bash
# 從現有 quant 版本複製
cp -r BCCTN_sep_conv_causal_slide_win_quant_256dim/ \
      BCCTN_sep_conv_causal_slide_win_mixed_precision_256dim/
cd BCCTN_sep_conv_causal_slide_win_mixed_precision_256dim/
```

### Step 1：重寫 Fake Quantization 模組

**現有問題**：`DCNN/utils/fp8_qat.py` 只支援 per-tensor FP8 fake quant。

**需要新增**：
1. **INT4 symmetric fake quant**（per-group / per-channel / per-head）：
   ```python
   def int4_fake_quant(x, group_size=256, num_bits=4):
       qmax = 2**(num_bits-1) - 1  # = 7
       # 按 group 切分
       x_grouped = x.reshape(-1, group_size)
       scale = x_grouped.abs().amax(dim=-1, keepdim=True) / qmax
       x_q = (x_grouped / scale).round().clamp(-qmax, qmax) * scale
       return x_q.reshape_as(x)
   ```

2. **Per-head scale for MHA**：
   - `in_proj_weight` shape = `(3*embed_dim, embed_dim)` = `(768, 256)`
   - 按 head 分：每 `head_dim=16` rows 一組，共 48 組（3×16 heads）
   - 每組獨立計算 scale

3. **FP16 累加器模擬**：
   - 在 matmul 前把 input cast 到 fp16，累加也用 fp16
   - 特別是 QK^T 和 Score×V

4. **Layer-wise 精度配置**：
   - 用 config dict 控制每層的 weight/activation 精度
   - 5 個 decoder outlier 層（dec PW stage 0,3,4）保持 FP8

**參考現有 code**：
- 現有 fake quant 邏輯：`DCNN/utils/fp8_qat.py`（FP8Parametrization 類）
- 層名對應：見 03 報告的層名規律表

### Step 2：在 Attention 中加入 Top-k Masking

**修改檔案**：`DCNN/utils/complexPyTorch/complexLayers.py` 的 `ComplexMultiheadAttention`

**問題**：PyTorch 原生 `nn.MultiheadAttention` 把 QK^T → softmax → Score×V 封裝在內部，無法在 softmax 後直接插入 top-k。

**解決方案**（二擇一）：

**方案 A（推薦）：自行實作 attention forward**
```python
def custom_attention_forward(Q, K, V, attn_mask, k=4):
    # QK^T
    attn_scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(head_dim)
    
    # 加上 causal sliding window mask
    attn_scores = attn_scores + attn_mask  # -inf for masked positions
    
    # Softmax
    attn_weights = F.softmax(attn_scores, dim=-1)
    
    # Top-k masking (可在 training 中用 straight-through 或直接 hard mask)
    if k < attn_weights.size(-1):
        topk_val, topk_idx = torch.topk(attn_weights, k=k, dim=-1)
        mask = torch.zeros_like(attn_weights)
        mask.scatter_(-1, topk_idx, 1.0)
        attn_weights = attn_weights * mask
        attn_weights = attn_weights / (attn_weights.sum(dim=-1, keepdim=True) + 1e-8)
    
    # Score × V
    output = torch.matmul(attn_weights, V)
    return output
```

**方案 B：用 PyTorch 的 scaled_dot_product_attention + custom mask**
- 較難插入 top-k，不推薦

**注意事項**：
- Top-k 的 backward 是 sparse 的，PyTorch autograd 會自動處理（scatter 有梯度）
- 報告 06 指出 4 個 flat heads（head 2,3,5,14）entropy 較高，可考慮 per-head k（sharp=4, flat=8）
- 初期建議統一 k=4，簡化實作

### Step 3：修改 Trainer 整合新量化

**修改檔案**：`DCNN/trainer.py`

```python
# 替換原有的 enable_fp8_qat 呼叫
from DCNN.utils.mixed_precision_qat import enable_mixed_precision_qat

enable_mixed_precision_qat(
    model,
    config={
        'encoder_dw': {'weight_bits': 4, 'weight_scale': 'per_channel'},
        'encoder_pw': {'weight_bits': 4, 'weight_scale': 'per_group', 'group_size': 256},
        'decoder_dw': {'weight_bits': 4, 'weight_scale': 'per_channel'},
        'decoder_pw_outlier': {'weight_bits': 8, 'weight_format': 'fp8_e4m3'},  # stage 0,3,4
        'decoder_pw_normal': {'weight_bits': 4, 'weight_scale': 'per_group', 'group_size': 256},
        'mha': {'weight_bits': 4, 'weight_scale': 'per_head', 'num_heads': 16},
    },
    activation_format='fp8_e4m3',
    accumulator_format='fp16',
    top_k=4,
)
```

### Step 4：Fine-tune 訓練

```bash
# 從 FP8 pre-trained checkpoint 啟動
CUDA_VISIBLE_DEVICES=0 python train.py \
    training.resume_from=/raid/R12K41024/BCCTN_sep_conv_causal_slide_win_quant_256dim/outputs/2026-03-30/15-18-17/logs/lightning_logs/version_0/checkpoints/epoch=134-step=130680.ckpt \
    training.n_epochs=30 \
    training.learning_rate=0.0001 \
    training.mixed_precision_qat=true \
    training.top_k_attention=4
```

**訓練建議**：
- Learning rate 降為原來的 1/10（0.0001），避免破壞已學到的特徵
- 訓練 20–30 epochs 應足夠（fine-tune 不需要從頭訓練）
- 監控 ILD 和 IPD loss 是否穩定（這是空間感保真的關鍵指標）

### Step 5：全模型聯合評估

```bash
CUDA_VISIBLE_DEVICES=0 python eval.py \
    --model_checkpoint "outputs/.../checkpoints/best.ckpt" \
    --vctk_test_dir   "../Dataset/noisy_testset" \
    --vctk_clean_dir  "../Dataset/clean_testset" \
    --output_dir      "./results" \
    --paper_style_eval \
    --batch_size 8
```

**必須驗證的項目**：

| 驗證項 | 方法 | 通過標準 |
|--------|------|---------|
| SI-SNR | 500 對 VCTK | ΔSI-SNR < 0.5 dB vs FP8 baseline |
| ILD error | 500 對 VCTK | ΔILD < 0.5 dB |
| IPD error | 500 對 VCTK | ΔIPD < 5° |
| Top-k stability | 對比 k=4 vs k=8 vs full | 確認 flat heads 不退化 |
| 累積量化誤差 | 對比 per-layer vs 全模型聯合 | 報告 02 已警告此風險 |

---

## 四、已知風險與注意事項

### 風險 1：累積量化誤差（高風險）
- 報告 02 證明：30 層個別 W3 時 ΔSNR < 0.18 dB，但 29 層同時 W3 時 ΔSNR = −3.08 dB
- **INT4 聯合量化的累積誤差必須用全模型評估確認，不能只看 per-layer 數據**

### 風險 2：Top-k 樣本量不足（中風險）
- 報告 06 的 top-k 評估僅用 2 個測試樣本
- Fine-tune 前建議先用 20+ 樣本確認 top-k=4 的 PTQ 效果穩定

### 風險 3：自行實作 Attention Forward 的正確性（中風險）
- 把 PyTorch 原生 MHA 拆開自寫需小心：
  - Sliding window mask 的 causal 方向
  - Complex attention 的 4-pass（re×re, re×im, im×re, im×im）
  - 多頭的 reshape 順序
- **建議先在 FP32 下驗證自寫 attention 與原生 MHA 輸出一致**，再加入量化

### 風險 4：Per-head scale 的梯度（低風險）
- INT4 fake quant 的 scale 是 derived（從 weight max 計算），非 learnable parameter
- STE（straight-through estimator）梯度會直接穿過量化 → 標準做法，通常無問題

---

## 五、現有分析報告與程式碼索引

| 資料夾 | 報告檔 | 核心結論 |
|--------|--------|---------|
| `01_utilization_analysis/` | `01_utilization_analysis_report.md` | MAC 利用率不均：DW 13-17%、MHA proj 92.6%、Score×V 33% |
| `02_weights_bitwidth/` | `02_weights_bitwidth_analysis_summary.md` | Uniform INT4 可接受（−0.56 dB）；Mixed W3 不行（−3.08 dB） |
| `03_dynamic_range/` | `03_dynamic_range_discussion_report.md` | 5 層 decoder outlier 需 FP8；MHA 需 per-head scale；gsz=256 夠用 |
| `04_sparsity_analysis/` | `04_sparsity_analysis_report.md` | 無結構化稀疏，硬體投入無效益 |
| `05_conv_reuse/` | `05_conv_reuse_analysis_report.md` | MHA DMA 640KB 是主瓶頸；MHA Out proj 可低秩分解 |
| `06_attn_mised_precision/` | `06_attn_mised_precision_analysis_report.md` | Top-k=4 幾乎無損；QK^T 需 fp16 累加器 |

**現有 FP8 QAT 程式碼**（最重要的參考）：
| 檔案 | 說明 |
|------|------|
| `BCCTN_sep_conv_causal_slide_win_quant_256dim/DCNN/utils/fp8_qat.py` | FP8 fake quant 實作（需擴展為 INT4） |
| `BCCTN_sep_conv_causal_slide_win_quant_256dim/DCNN/utils/complexPyTorch/complexLayers.py` | ComplexMultiheadAttention（需加 top-k） |
| `BCCTN_sep_conv_causal_slide_win_quant_256dim/DCNN/models/model.py` | MultiAttnBlock（attention mask 邏輯） |
| `BCCTN_sep_conv_causal_slide_win_quant_256dim/DCNN/trainer.py` | QAT 啟用點 |

---

## 六、成功標準

本次 QAT fine-tune **完成**的定義：

1. Mixed-precision fake quant 模組完成並通過單元測試
2. Top-k=4 attention 實作完成並驗證（FP32 下與原生 MHA 輸出一致）
3. Fine-tune 完成（20-30 epochs from FP8 pretrained checkpoint）
4. 全模型聯合評估通過品質閾值（SI-SNR/ILD/IPD）
5. 輸出最終的 per-layer 精度配置表和對應的 checkpoint

**如果品質不達標的 fallback 策略**：
- 先嘗試放寬 decoder outlier 層數（不只 stage 0,3,4，擴展到更多層保持 FP8）
- 再嘗試 top-k=8（犧牲部分計算量省益，換回品質）
- 最後嘗試 gsz=32（更細的 scale 粒度，增加 overhead 但提升精度）
