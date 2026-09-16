# REQUEST: fp16-PPU 數值模型的 deploy 品質評估(MBSTOI)— 2026-06-10

To: 演算法顧問
From: RTL 端(CE W4A8 chain)
Priority: **高**(發現誤差沿鏈複利;不阻塞 RTL 驗證,但決定 PPU 是否需要 fp32 升級)

## 背景

RTL 的 conv PPU(post-MAC)整段以 **fp16** 實作:BN affine = APoT-6a dealer(2 shift + 1 fp16
saturating add)+ fp16 β add → PReLU pow2(shift+sign mux)→ ×2^cast_k → RNE e4m3。
deploy 模型(strict ckpt)同段是 **torch fp32** 算完後單次 quantize。兩者在 e4m3 量化邊界附近
會有 ±1 ulp 的碼值翻轉。

## 發現(2026-06-10,鏈式 replica 量化)

孤立單層(deploy 輸入餵入)時翻轉率僅 ~2%(98%+ 碼值一致,全部 ±1 ulp):

| 單層 | re+im 碼值一致率 |
|---|---|
| enc1 pw / enc3 pw | 98.05% / 98.44% |
| dec1 dw / dec4 dw | 98.83% / 98.83% |

但**鏈起來(每層輸出餵下層,= 真實硬體 dataflow)誤差複利**(t0、雙耳真實資料):

```
ear0: L0 98.4% → L3 96.5% → L5 91.6% → L7 69.7% → L9 65.4% → L10(enc5 dw) 62.5%
ear1: L0 100%  → L3 94.1% → L5 70.9% → L7 54.5% → L9 58.0% → L10        60.9%
```

enc5 時 ~40% 活化值與 deploy 偏離(每個偏離 ≈ ±1 ulp e4m3 ≈ 6–12% 相對誤差/碼)。
decoder 12 層還會再疊。QAT 模型對 e4m3 噪聲本身魯棒,我們預期 MBSTOI 掉幅有限,但需要數據。

## 請求

1. **用 fp16-PPU 數值模型重跑 deploy eval**(VCTK 0.8357 / TIMIT 0.7898 那組),回報 MBSTOI delta。
   fp16 模型的精確規格(bit-exact 契約,Python 參考實作在 repo:
   `10_algo/scripts/gen_golden_postbn.py` 的 `ppu_conv()` + `gen_golden_chain_enc.py` 整鏈):
   - 所有 PE 乘積:INT4 × e4m3(值域)→ **fp16 RNE encode**
   - 32-lane 加法樹:**成對 fp16 saturating add**(每級 RNE、±65504 飽和;結合順序 = 樹)
   - dealer(weight descale 與 BN α 共用):`2^a 平移 + (±2^b 平移) 的 fp16 add`(平移本身精確)
   - OACC 跨 chunk/tap:fp16 saturating add(順序 = chunk/tap 序)
   - BN:`fp16_add_sat(dealer(oacc, α_6a), β_fp16)`;PReLU:負值 `×±2^k` 平移;
     cast:`×2^cast_k` 平移後 **fp16→e4m3 RNE**(OCP,飽和 448)
   - β 常數本身 = fp32 fused bias 的 fp16 RNE(WBUF image 即此值)
2. 若 delta 不可接受:我們把 **BN 段升 fp32 加法器**(dealer 平移天然精確、無乘法器,
   不破 zero-multiplier;面積小漲)。請一併評估「只升 BN add、樹仍 fp16」是否足夠
   (樹的 fp16 在單層檢查貢獻 <1e-3 rel,主要翻轉源是 BN+cast 邊界)。
3. (可選,加速交叉驗證)dump 一份 **fp16 模型的鏈式中間活化**(t0,enc 12 層,__0..__3),
   我們可逐層 bit 對比確認兩邊 fp16 實作一致。

## 追加(同日稍晚):preshift residual image 重烤升級為「品質阻塞」

Decoder 鏈式 replica 隔離診斷(deploy 輸入逐 stage 餵):

| dec stage | pw 型態 | merged vs deploy dw-input dump |
|---|---|---|
| S0 / S3 / S4 | **FP8-outlier L-Mul** | **42.2% / 61.7% / 70.3%** ← 飽和損壞 |
| S1 / S2 / S5 | INT4 split-k | 98.4% / 98.8% / 100% ✓ |

確認 2026-06-08 的分析:preshift=0 時 FP8×FP8 乘積(4.2× 餘裕不足)在 fp16 溢出→飽和,
FP8 stage 與 deploy 大幅偏離,且沿 decoder 鏈放大(鏈式 S2 後幾乎全偏)。當時已決策方案 (a)
「12 個 FP8-outlier tensor 的 24 個 R2 WEIGHT_SCALE codes 以 residual=full−preshift 重烤 +
descriptor preshift 同步設置」,**請排程執行 image 重烤**(builder 任務:offline apot_encode
round-trip,連同 fused_descale_summary 重出),否則 decoder 品質無法評估。重烤後我們同步
regen descriptor(preshift 欄位已留好,pairing 結構性保證)。

## 附帶說明

- RTL 驗證不受影響:RTL 對 Python replica 已逐層 bit-exact(24_run_native 等),replica 即硬體真值。
- 相關既有結論:fp16-tree overflow 分析(INT4 conv 安全)、in_proj K/V cast 曾 100% 吻合
  (該路徑無 BN dealer,只 β+cast,翻轉面小)。
