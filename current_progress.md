# Current Doing
1. run kernel_size=(self.kernel_size, 1) instead of kernel_size=(self.kernel_size, 2)
2. change the fft from 256 to 128 apply to sep conv, but first check the result of the decrease kernel size 

# current training step
1. fft = 256 with kernel = 1 (causal attention)
2. fft = 128 with kernel = 1 (causal attention)

# TODO — Hardware-Aware Analysis (2026-04-10 顧問討論)

## T1（立即做，投入低、回報高）— profiling 腳本，不需 re-training
- [ ] **方向 1：Utilization & Pipeline 瓶頸分析** (`01_utilization_analysis/`)
  - 逐層計算 MAC cycles vs DMA cycles，算 MAC utilization
  - 重點：depthwise vs pointwise 的 MAC idle 差異
  - 已有數據：Conv ~0.3ms, Attn ~2.9ms, DMA:compute ≈ 5.8:1
- [ ] **方向 3：Weight Dynamic Range 分析** (`03_dynamic_range/`)
  - Dump 每層權重 histogram，計算 min/max/std/kurtosis
  - Conv 做 per-output-channel 分析，MHA 做 per-head 分析
  - 結果決定 scalar factor granularity 和 FP vs INT 選擇

## T2（核心實驗，需要 re-training）
- [ ] **方向 2：Weight Bit-width 探索** (`02_weights_bitwidth/`)
  - PTQ sweep: W8/W4/W3 × FP/INT 組合
  - 觀察 per-layer SNR degradation + 整體 MBSTOI/ILD/IPD
  - Mixed-precision: Conv depthwise 保持 8-bit，pointwise + MHA 壓 4-bit

## T3（進階優化，取決於前面分析結果）
- [ ] **方向 4：Zero/Sparsity 結構分析** (`04_sparsity_analysis/`)
  - 多 threshold (|w|<1e-3, 1e-2, 1e-1) 統計 zero ratio
  - 區分 structured vs random sparsity
  - 對 attention score (softmax 輸出) 也分析
- [ ] **方向 6：Attention Mix-Precision / Dense 分析** (`06_attn_mised_precision/`)
  - QK^T 高精度累加，Score×V 低精度
  - 分析 attention score density（W=31 短視窗下是否趨近零）
  - 考慮 top-k sparse attention

## T4（最後考慮，邊際收益小）
- [ ] **方向 5：Conv Weight Reuse 優化** (`05_conv_reuse/`)
  - 跨層 weight similarity 分析
  - Temporal reuse（相鄰幀 activation 重疊 cache）

## W4A8 關鍵結論
- W4 + per-group scalar (group_size=32): DMA 量砍半，scale overhead ~6%
- 需 re-training 驗證 ILD/IPD 品質