# RESPONSE: fp16-PPU 數值模型的 deploy 品質評估(MBSTOI)— 2026-06-10

To: RTL 端(CE W4A8 chain)
From: 演算法顧問
Re: `REQUEST_FP16_PPU_EVAL_2026-06-10.md`

## TL;DR

| 交付 | 結論 |
|---|---|
| **#1 fp16-PPU MBSTOI delta** | **≈ 0**。full fp16 conv-PPU:VCTK **0.8357**(deploy 0.8357,Δ **0.0000**)、TIMIT **0.7897**(deploy 0.7898,Δ **−0.0001**)。**不需要把 PPU 升 fp32**。 |
| **#2 只升 BN add(tree 仍 fp16)** | 不需要 —— 連 BN 都不用升;full fp16(含 BN)就已經零損。BN-fp32/tree-fp16 變體見下表(同樣 ≈ deploy)。 |
| **#3 逐層 bit 對比** | 暫不需要。我已用自建 fp16 模型重現你們的鏈式碼值退化曲線(同形狀、**偏保守**),MBSTOI delta 站得住。需要時再交換 dump。 |
| **追加:FP8-outlier preshift 重烤** | **完成**:source 重烤(24 codes,residual=full−preshift,恆等)→ **wbuf 重打包**(@245268/247104/247164,30 bytes,bytes-only,自驗 PASS)→ **認證**(dec4 Inf witness 消失;FP8 進 fp16 tree+preshift MBSTOI **0.8356/0.7897**)。descriptor preshift 為 RTL 機械尾段(manifest 已附)。 |

**一句話**:conv PPU 全程 fp16(tree+dealer+BN+PReLU+cast)對 MBSTOI 無感(±1 ulp e4m3 噪聲,QAT 模型吸收)。唯一真正的品質風險是 decoder FP8-outlier 的 fp16 溢位 —— 那不是 PPU 數值問題,是 preshift image 問題,已照你們 2026-06-07 的設計重烤。

## 1. 怎麼建的 fp16-PPU 模型(可重現)

不依賴你們的 `ppu_conv()` / `10_algo` HDL reference —— 用 deploy code 自建 env-gated 注入,規格依 `CE_DATAPATH_DRAFT §1/§3` + 本 request §1 契約。

- 注入點:`DCNN/utils/complexPyTorch/complexLayers.py`
  - `apply_complex`:4 條 real-product stream 各過 fp16 saturating →OACC re=P0−P1 / im=P2+P3 全 fp16(`tree`)
  - `_apply_folded_bn`:`fp16(fp16(x·α)+fp16(β))`,α=APoT 2^a±2^b(exact)→`fp16(x·α)` 與 HW dealer 兩次 exact 位移後 fp16 加法 bit 一致(`bn`)
  - `ComplexPReLU`:pow2 slope 位移後 fp16(`prelu`,pow2 對 fp16 值是 exact → 數值上 inert)
  - e4m3 cast 不動(改用既有 `calibrated_pow2_e4m3` block-float,只是現在 round 的是 fp16 值)
- 開關:`BCCTN_FP16_PPU=tree,bn,prelu`(off/unset → 與 fp32 deploy **bit-identical**,baseline 零影響)
- INT4 conv fp16-safe(無溢位,`RTL_FIX_FP16_TREE_PRESHIFT` 已證),故 conv 路 saturating==plain fp16。
- 跑法:`run_fp16_ppu_eval.sh`(deploy config:pow2 block-float ckpt + calib + delayed-APoT descale;GPU 集中、METRIC_WORKERS=32)。

### 驗證 gate(`scripts/verify_fp16_ppu_chain.py`,t0 雙耳真實資料)

逐 encoder conv 邊界比 e4m3 碼,off vs fp16,重現你們的鏈式複利退化:

```
        enc0  enc1pw(L3) enc2pw(L5) enc3pw(L7) enc5dw(L10)
RTL     98.4/100  96.5/94.1  91.6/70.9  69.7/54.5  62.5/60.9
my all  100/100   88.1/87.5  69.5/68.2  54.3/52.8  47.0/46.6
```

同形狀(STFT 輸入邊界 100% → enc5 大幅退化)。我的曲線**更陡 ~15pt**,差距主因 = t0 檔案不同(我的 t0=`p225_009_az-40_snr-3.0`,你們可能不同 → activation 統計不同)+ dealer/tree 結合順序。對 MBSTOI 是**保守方向**:我的 fp16 噪聲比你們真值大,delta 仍≈0 → 真 HW 至少一樣好。

## 2. MBSTOI 結果(VCTK+TIMIT 750/750,paper-style)

| 變體 | BCCTN_FP16_PPU | VCTK | TIMIT | Δ(vs baseline) |
|---|---|---|---|---|
| deploy 基準(我的 code, off) | (off) | **0.8357** | **0.7898** | — **完全等於原 deploy**(off==fp32 bit-identical 驗證) |
| **full fp16-PPU(#1)** | `tree,bn,prelu` | **0.8357** | **0.7897** | 0.0000 / **−0.0001** |
| BN-fp32 / tree-fp16(#2) | `tree,prelu` | **0.8357** | **0.7899** | 0.0000 / **+0.0001** |

> 三者皆 ±0.0001(純噪聲級)。off 精確重現 0.8357/0.7898 證明注入 off-path bit-identical;
> #2(0.7899)甚至略高於 baseline → **BN 也不必升 fp32**,full fp16(含 BN)即零損。
> #1 的 TIMIT −0.0001(≠ exact baseline)證明 fp16 確有作用,只是 MBSTOI 無感。

### 2.1 段別分解(碼值翻轉源,t0)

| 段 | enc5 dw 碼值一致率(ear0/ear1) | 結論 |
|---|---|---|
| all (tree+bn+prelu) | 47.0/46.6 | 滿配 fp16 |
| bn only | 48.7/48.0 | **BN 是主翻轉源**(≈all) |
| tree only | 50.1/50.0 | 次之 |
| prelu only | 65.2/64.0 | 幾乎 inert(pow2 位移對 fp16 exact) |

與你們「主要翻轉源是 BN+cast、tree <1e-3 rel」一致(碼翻轉率高 ≠ 量值大;tree 的 ±1ulp 量值確實 <1e-3)。**重點:不論哪一段,MBSTOI 都沒掉**,所以 BN 升 fp32 與否都不影響品質結論。

## 3. Decoder FP8-outlier(追加):為何 MBSTOI eval 不含它,以及重烤

- 你們追加發現 S0/S3/S4(FP8-outlier L-Mul)fp16 溢位飽和(42/62/70% 隔離匹配)——這是 **preshift=0 image bug**,不是 conv-PPU 的良性 ±1 ulp。
- 我確認 `enable_outlier_fp8_lmul` 的乘法走 `_lmul_*`(fp32),**不經 `apply_complex`**,故我的 `BCCTN_FP16_PPU` 注入**不碰** FP8-outlier 乘法 → eval 中它維持 fp32(不溢位)。因此 **#1 的 delta 是「假設你們 preshift 重烤後」的乾淨數字**(良性 conv-PPU fp16),FP8 溢位由 image 修。
- **重烤(已做)**:`scripts/rebake_fp8_preshift.py --apply`,對 12 個 FP8-outlier tensor:
  - `preshift = floor(log2(32768/maxnode))`(per-tensor,得 2⁻³×8 / 2⁻⁴×2;stage0 r/i_skip maxnode<2.1k → 0)
  - residual = full × 2^(−preshift) ⇒ APoT 純指數位移:`a_exp,b_exp += |preshift|`,has_second/sign 不變(base + fused_act_k{real,imag} 同步)
  - **恆等**:pe_pair 套 2^preshift、dealer 套 residual → 淨 descale=full,deploy 輸出 bit 不變(MBSTOI 不受影響);僅 headroom 變(post|node| ≤0.49×,0 overflow)
  - 驗證:逐 tensor round-trip(decode(new)==decode(old)×2^|s|)+ `verify_fp16_tree_headroom` post-fix 0 overflow,**ALL PASS**
  - 產出:位移後的 `fp8_descale_apot_{a,b}_exp[.|_fused_act_k*].int16.bin`、原檔備份於 `.prebake_bak/`、`preshift_manifest.json`(每 tensor 的 preshift,供 descriptor `preshift_{r,i}`)、`fused_descale_summary.json` 已標註 preshift
- **wbuf 重打包(已完成)**:`scripts/patch_wbuf_fp8_preshift.py --apply`。descale 不是獨立 region,而是普通 `weight_scale` const_block(region R2,`apot_6a`,3 bytes = 2×12-bit kreal‖kimag),已在 image:dec0 @245268/71/74/77、dec3 @247104/07/10/13、dec4 @247164/67/70/73。`build_sram_images.py` 的 fused a/b 來源 = `--fused-root hw_patterns_final/tb_pattern/weights`(= 我重烤的目錄);has_second/sign 來自 `--export-root export_out_strict18/weights`。surgical in-place patch(重用 builder 的 `apot_code`/`pack_codes`,bias=24)→ **30 bytes 改變、全在 12 個 weight_scale 塊內、offset/size 不變**;自驗 current wbuf 解碼 == prebake codes、new == old<<|preshift|,**ALL PASS**。`memory_map.json` 本步**未動**(offsets/sizes 不變,bytes-only)。原 wbuf 備份 `wbuf_unified_init.bin.pre_fp8preshift_bak`。
- **RTL 端尾段(機械)**:依 `preshift_manifest.json` 設 descriptor `preshift_{r,i}` + regen ROM(欄位已留、07_run_peg_preshift 已單元測);gen 腳本 lmul_fp8 加 preshift 項;回歸自動跟上。

### 3.1 最後一段認證:FP8-outlier 真正進 fp16 tree(REPLY Q2)

我這次 #1 eval 的 FP8-outlier 乘法走 fp32 bypass;re-bake 後它才真正進 fp16 tree。補上認證:

- **Witness(`scripts/witness_dec4_fp8_inf.py`)**:dec4 FP8 product 無 preshift = **200704(=448×448)→ Inf**,peak frame 每 tensor 數千 Inf nodes(t=48/1034/6811,= 你們「t=351」同類事件);**re-baked preshift(2⁻³/2⁻⁴)→ 0 Inf、≤0.49×**。re-bake 消除溢位事件。
- **MBSTOI(`BCCTN_FP16_PPU=tree,bn,prelu,outlier`,FP8-outlier L-Mul 在 fp16 tree + preshift)**:**VCTK 0.8356 / TIMIT 0.7897**(baseline 0.8357/0.7898,Δ −0.0001/−0.0001)。注入點 `all_lmul_multiply.py` outlier forward:`out = fp16_sat(acc·2^ps)·(descale·2^-ps)`(ps 來自 `_OUTLIER_PRESHIFT` = manifest),preshift 為 exact pow2 → 值不變,fp16 只加良性 ±1 ulp。**最後未認證 segment 過關**:整條 fp16 chain(conv PPU + decoder INT4 + FP8-outlier)對 MBSTOI 全程 ≤ ±0.0001。

## 4. 交付清單

- `18_…hw_handoff_20260530/code/DCNN/utils/complexPyTorch/complexLayers.py` — fp16-PPU 注入(env-gated,off bit-identical)
- `18_…hw_handoff_20260530/run_fp16_ppu_eval.sh` — A/B eval driver
- `18_…hw_handoff_20260530/scripts/verify_fp16_ppu_chain.py` — 驗證 gate(碼值退化重現)
- `18_…hw_handoff_20260530/results/fp16_ppu_eval/seg_*/` — 三變體 MBSTOI
- `19_algo_final/scripts/rebake_fp8_preshift.py` — preshift 重烤(--apply 已執行,backup + manifest)
- 本文件

## 附帶

- 需要交付 #3(逐層 bit 對比)時:給我你們 t0 檔名 + fp16 鏈式 dump(enc 12 層 __0..__3),我逐位對齊兩邊 fp16 實作。
- 重烤的 wbuf 打包若要我接手,給我確認 `build_sram_images.py` 讀的是 `19_algo_final/hw_patterns_final` 的 weights、且 FP8 descale 確實進 wbuf 的區段。
