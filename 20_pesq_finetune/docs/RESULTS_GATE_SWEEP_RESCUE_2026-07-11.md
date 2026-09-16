# RESULTS: Gate-holding PESQ sweep + rescue — 終局判決(NO_GATE_PASS)與量測 Pareto

Date: 2026-07-11
Base candidate: deploy ep3 `18_w4a8_pow2_mixed_lmul/outputs/d1c3_pow2_finetune/d1c3_apotclip16_attnm8_outlierm4_15ep/ft/logs/lightning_logs/version_0/checkpoints/epoch=2-step=726.ckpt`(canonical VCTK MBSTOI 0.8357 / TIMIT 0.7898)
Handoff: [HANDOFF_GATE_SWEEP_CONTINUATION_2026-07-10.md](HANDOFF_GATE_SWEEP_CONTINUATION_2026-07-10.md)(§4 決策樹之 3 = 本文結論)

## 0. TL;DR

Gate = canonical VCTK MBSTOI >= 0.835 AND TIMIT MBSTOI >= 0.788(dir-18 `run_apot_descale_eval.sh`, `ACT_FP8_MODE=calibrated_pow2_e4m3` + deploy calib json, VCTK_ONLY=0, LIMIT=750)。

| Variant | 內容 | canonical vMBSTOI / tMBSTOI | canonical v_wb / t_wb | Verdict |
|---|---|---|---|---|
| gate sweep(R0–R4, 27 ckpts) | 5 條配方 × 每 epoch 篩選,7 顆候選 canonical | 最佳 0.8344 / 0.7910(sisdr_v2e5) | 1.6754 / 1.7252 | **NO_GATE_PASS**(VCTK 全滅,最近差 0.0006) |
| rescue(§4(a):sisdr_v2e5 + STOI_W=12 ×3ep) | 3 顆全 canonical | 0.8301→0.8290→0.8251(單調惡化) | v_wb 1.6995→1.7020→1.6575 | **RESCUE_NO_PASS** |
| ⭐⭐ 建議操作點 1:deploy ep3(不變) | gate + 空間線索主角 | 0.8357 / 0.7898(唯一雙 gate PASS) | 1.5932 / 1.6558 | 晶片主模型 |
| ⭐⭐ 建議操作點 2:`sw_w6cont_s12_v4e1` | PESQ-boosted 副點(兩點故事) | 0.8308 / 0.7886(VCTK FAIL −0.0042;TIMIT PASS +0.0006) | **1.7092 / 1.7531**(vs deploy +0.1160/+0.0973) | ILD 3.3363 仍優於 noisy;MBSTOI 僅 −0.0049/−0.0012 vs deploy |

**一句話:在此架構上「守 gate」與「wb PESQ > ~1.71」不可兼得(量測 Pareto,非猜測);與 SNU 的 Δ 品質差距在任何頻寬、任何 gate-holding 點都是 ~−0.2,Tier-2 正式蓋棺。論文走兩點故事 + 空間線索主打。**

## 1. Setup

- Sweep 設計、gate 定義、序列化單工協定:見 [HANDOFF_GATE_SWEEP_CONTINUATION_2026-07-10.md](HANDOFF_GATE_SWEEP_CONTINUATION_2026-07-10.md) §1–2、§9。
- 執行紀錄:trainer 2026-07-10 18:58–21:29(最終 banked R0 4/4、R1 6/6、R2 6/6、R3 5/5、R4 6/6;R1–R4 全部 attempt-1 一次過,R0 的 3/4 顆承接自 16:57–18:26 舊 chain、於 19:07 補齊第 4 顆);screener 22:47–次日 10:13(27 顆 scaled screen + 7 顆 canonical);rescue 07-11 13:14–14:42。**單工協定下 0 次外部 kill**(對照 07-10 傍晚並行時期的連環 kill;並行觸發說未被否證,操作規則維持)。
- Screener 候選規則含本輪新增的 vMBSTOI top-3 保底名額(`run_gate_sweep_screen.sh:159-162`,動機見 §3.1 offset 翻向)。

## 2. Results

### 2.1 Canonical gate 判決(7 候選,`logs/gate_sweep/canonical_summary.tsv`)

| stag | vMBSTOI | tMBSTOI | v_wb | t_wb | v_ILD | verdict |
|---|---|---|---|---|---|---|
| w6ep4_anchor | 0.8313 | 0.7875 | 1.7101 | 1.7475 | 3.4046 | FAIL(雙) |
| sw_w6cont_s12_v4e1 | 0.8308 | 0.7886 | 1.7092 | 1.7531 | 3.3363 | FAIL(V) |
| sw_w4_lr5e6_v1e0 | 0.8306 | 0.7887 | 1.6815 | 1.7096 | 3.4632 | FAIL(V) |
| sw_w6cont_s12_v3e0 | 0.8337 | 0.7902 | 1.6887 | 1.7397 | 3.3212 | FAIL(V −0.0013) |
| sw_w2_lr1e5_v1e2 | 0.8317 | 0.7861 | 1.6659 | 1.6926 | 3.4295 | FAIL(雙) |
| sw_w4_sisdr_v2e4 | 0.8335 | 0.7905 | 1.6795 | 1.7319 | 3.3109 | FAIL(V −0.0015) |
| sw_w4_sisdr_v2e5 | 0.8344 | 0.7910 | 1.6754 | 1.7252 | 3.3198 | FAIL(V **−0.0006**) |

TIMIT gate 5/7 過;**卡點唯一在 VCTK 0.835**。scaled screen 全表(27 顆)見 `logs/gate_sweep/sweep_summary.txt`(scaled vMBSTOI 極值 = sisdr_v2e5 0.8360)。

### 2.2 Rescue(§4(a))判決(`logs/gate_sweep/rescue_summary.tsv`)

種子 = sisdr_v2e5(canonical 0.8344/0.7910),配方 = R4 不變(PESQ 4、sisdr_tied、lr 5e-6)僅 STOI_W 10→12,續訓 3 ep,每顆直接 canonical:

| ckpt | vMBSTOI | tMBSTOI | v_wb | v_ILD | verdict |
|---|---|---|---|---|---|
| 種子(參考) | 0.8344 | 0.7910 | 1.6754 | 3.3198 | — |
| rescue e0 | 0.8301 | 0.7872 | 1.6995 | 3.4045 | FAIL |
| rescue e1 | 0.8290 | 0.7842 | 1.7020 | 3.5082 | FAIL |
| rescue e2 | 0.8251 | 0.7825 | 1.6575 | 3.6419 | FAIL |

單調遠離 gate,ILD 同步劣化;STOI_W=12 在此種子上無法移動平均值(機制見 §3.2)。

### 2.3 Offset 定律:finetuned 權重的 canonical 一律低於 scaled

deploy ep3 量得的「scaled 比 canonical 低 0.0016/0.0019」**不可外插**。本輪 7 顆候選 canonical−scaled 全為負:VCTK −0.0015~−0.0030、TIMIT −0.0006~−0.0025(例:anchor −0.0024/−0.0013、s12_v4e1 −0.0020/−0.0015、w4_v1e0 −0.0030/−0.0006)。**deploy ep3 的正向偏移是特例;PESQ-finetuned 權重一律負向。** 後果:scaled screen 門檻需上修 ~+0.003 才對應 canonical gate;本輪 screener 的 0.8325/0.7855 門檻偏鬆(僅浪費 canonical 名額,不漏真陽性)。

### 2.4 vs SNU(Δ PESQ, SNR −6..9, VCTK;SNU 復現數據見 [RESULTS_MONO_SE_BASELINES 2026-07-09](../../21_mono_se_baselines/docs/RESULTS_MONO_SE_BASELINES_2026-07-09.md))

| model | Δwb | Δnb | gate |
|---|---|---|---|
| SNU repro(mono FP32) | +0.5618 | +0.6980 | — |
| deploy ep3 | +0.2628 | n/a(舊 CSV 無 nb 欄) | PASS |
| s12_v4e1(canonical) | +0.3679 | +0.5009 | FAIL(V) |
| w24 ep16(scaled) | **+0.5932(贏 +0.0314)** | +0.6136(**輸 −0.0844**) | 全滅(0.8188/0.7748) |
| w24 ep19(scaled) | **+0.6376(贏 +0.0758)** | +0.6552(**輸 −0.0427**) | 全滅 |

**修正先前說法:「w24 贏 SNU PESQ」僅在 wb 成立,nb 仍輸。** 機制:torch_pesq 訓練 loss 是 wideband(先前驗證),wb 增益部分是同卷效應;SNU 以 SI-SDR 訓練,增益跨頻帶平均。結論:**任何操作點、任何頻寬,gate-holding 模型對 SNU Δ 均輸 ~0.19–0.20;Tier-2(HANDOFF §1)蓋棺。** 論文 Δ 品質欄誠實列出,主打欄為空間線索(ILD 5.71 惡化 vs 3.31 改善)。w24 兩顆為 dir-20 scaled 數字(未跑 canonical);nb 差距 −0.04~−0.08 不受 ~0.002 模式偏移影響,wb +0.0314(ep16)在偏移量級邊緣、引用時需附註。

### 2.5 附帶發現:JSSC'20(NTU)ILD < JSSC'24(SNU)的機制

16/16 測試點(2 集 × 8 SNR)同向:SNU ILD 誤差每點高於 NTU(每點 +0.113~+0.434 dB;集均 VCTK +0.237 / TIMIT +0.291),同時 SNU Δwb 每點優於 NTU;noisy 參考逐位相同(非 harness artifact)。機制:ILD = L−R(dB),bilateral mono 推論下**兩耳共同的增益成分在 ILD 相消,只有 mask 的耳間分歧變成 ILD 誤差**。SNU 的 FGRU/TGRU band-split mask 銳利跟蹤各耳局部內容(故品質贏),耳間分歧大;NTU 的 conv+FC mask 鈍、近共模(故品質輸、ILD 傷少)。反證:15 dB 時 NTU 抑噪量反而較大(SegSNR_impr 3.26 vs 2.13)但 ILD 仍傷較少 → 傷 ILD 的是 mask 耳間差異性,非抑噪能量。**敘事價值:mono 範式內「增強越好 → 空間破壞越大」單調成立;binaural 打破此 trade-off(降噪與 ILD 同時改善)。**(量級提醒:SNU−NTU 差 ~0.3 dB 為二階;範式效應 ~1.8× 才是主角。)

### 2.6 w24 ep19(PESQ-ceiling ablation)vs mono SNU/NTU:PESQ 與 ILD 並列 — appended 2026-07-11

§2.4 的延伸:把合法 PESQ 天花板 w24 ep19(W4A8 零乘法 stack,dir-20 scaled)與 mono FP32 復現同表,PESQ 與 ILD 並列(全 −6..15 平均;Δ = enhanced − noisy;d69 = SNR −6..9)。

VCTK:

| model | PESQ wb(Δ) | PESQ nb(Δ) | d69 Δwb / Δnb | ILD err(vs noisy 3.7141) | MBSTOI |
|---|---|---|---|---|---|
| w24 ep19(W4A8) | **1.9811(+0.7407)** | 2.5743(+0.7146) | **+0.6376** / +0.6552 | **3.5795(−0.1347 改善)** | 0.8133 |
| SNU repro(FP32) | 1.9494(+0.7089) | **2.6372(+0.7775)** | +0.5618 / **+0.6980** | 5.7056(+1.9915 惡化) | 0.8383 |
| NTU repro(FP32) | 1.8193(+0.5788) | 2.5396(+0.6799) | +0.4307 / +0.5834 | 5.4689(+1.7547 惡化) | 0.8372 |

TIMIT:

| model | PESQ wb(Δ) | PESQ nb(Δ) | d69 Δwb / Δnb | ILD err(vs noisy 3.2902) | MBSTOI |
|---|---|---|---|---|---|
| w24 ep19(W4A8) | 1.9309(+0.6641) | 2.4669(+0.6671) | **+0.5571** / +0.5918 | **3.2943(+0.0041 持平)** | 0.7722 |
| SNU repro(FP32) | **1.9351(+0.6684)** | **2.5724(+0.7727)** | +0.5205 / **+0.6744** | 5.7940(+2.5038 惡化) | 0.8035 |
| NTU repro(FP32) | 1.7642(+0.4974) | 2.4328(+0.6331) | +0.3711 / +0.5291 | 5.5032(+2.2130 惡化) | 0.8006 |

- **主句型:單耳品質打平/反超(且是 W4A8 vs FP32 的不利比法),空間線索兩個世界** — w24 wb VCTK 全面贏 SNU(abs +0.032、d69 Δ +0.076)、TIMIT 打平(−0.0042);ILD 卻是「改善/持平」vs「+1.75~+2.50 dB 毀滅」。把架構推到 PESQ 天花板、放棄 gate,雙耳線索依然完好 — mono 範式在任何品質水位都買不到。
- nb 誠實列輸(abs −0.0629/−0.1055;d69 Δnb −0.0428/−0.0826):torch_pesq 訓練目標為 wb(§2.4 機制)。
- **MBSTOI 欄不做跨範式 head-to-head**(w24 0.8133 < SNU 0.8383 不構成反證):MBSTOI 的 EC 階段對每個時頻帶網格搜索耳間補償 γ∈[−20,+20] dB(40 點)、τ∈[−1,+1] ms(100 點)後取最優(`code/MBSTOI/mbstoi.py:51-56,126-127`),mono 造成的 ~5–9 dB ILD 誤差完全在可補償範圍內、計分前即被 EC 吸收;它量的是「最優耳間補償下的可懂度」,不是「空間線索是否正確」。w24 自身不守 deploy gate(0.8133/0.7722)→ 本節定位 ablation 行,非操作點。
- 數字來源:`results/pesq_eval/w24_ep19_750/`、`21_mono_se_baselines/results/bilateral_eval/{snu,ntu}/`(noisy 參考逐位相同)。晶片對比表版本見 [RESULTS_CHIP_COMPARISON_FINAL_2026-07-09.md](RESULTS_CHIP_COMPARISON_FINAL_2026-07-09.md) §2.10。

## 3. Cross-cuts

### 3.1 卡點解剖:VCTK 0.835 是唯一的牆

TIMIT gate 在 7 顆候選中 5 過;VCTK 全滅。gate-adjacent 家族的 canonical vMBSTOI 中心 ~0.831–0.833,天花板(選最大值)0.8344。

### 3.2 贏家詛咒(方法論教訓)

Sweep 內 epoch-對-epoch canonical vMBSTOI 抖動 ~±0.003–0.004;「27 顆選最大值」本身抬高 ~1–2σ。續訓被選中的極值點必然回歸平均——rescue 三顆單調下滑即此(0.8344→0.8301→0.8290→0.8251)。**推論:再抽籤(多訓 epoch、多挑 ckpt)不能可靠跨過 0.835;需要能移動「平均值」的訓練訊號,而 STOI_W 10→12 已被證明不是。** 唯一未試的均值移動機制 = MetricGAN-on-deploy v2(程式就緒,HANDOFF §5)→ future work。

### 3.3 量測 Pareto(本輪的正資產)

canonical 座標下:(0.8357, 1.5932) deploy → (0.8344, 1.6754) sisdr_v2e5 → (0.8337, 1.6887) s12_v3e0 → (0.8308, 1.7092) s12_v4e1 → (0.8188*, 1.9392*) w24 ep16(*scaled)。斜率:在 gate 邊緣每 +0.01 wb ≈ −0.001 MBSTOI;past 0.83 後急劇變陡。**「gate ⇒ wb <~1.71」是量測結論。**

### 3.4 單工協定驗證

07-10 18:58 起 ~20 小時、3 條 chain、~34 顆 ckpt 訓練+評測,0 外部 kill(對照並行時期 5 分鐘內必死)。並行觸發說維持;`strace` 落網程序仍備而未用。

## 4. Recommendation

**收官,不再追 gate。** 兩點故事:

- **點 1(晶片/gate/空間主角)deploy ep3** — 不動,所有 gate 與 ILD/IPD/ITD/IC 主張由它扛。
- **點 2(PESQ-boosted 副點)`sw_w6cont_s12_v4e1`** — canonical 0.8308/0.7886、wb 1.7092/1.7531(vs deploy **+0.1160/+0.0973**)、ILD 3.3363(仍優於 noisy 3.7141)。句型:「+0.12 wb PESQ 的代價為 MBSTOI −0.005,雙耳線索保持改善」。
  ckpt: `outputs/pesq_finetune/sw_w6cont_s12/ft/logs/lightning_logs/version_4/checkpoints/all-epoch=1-step=968-val=3.3113.ckpt`
- MetricGAN-on-deploy v2 = future work(唯一可能的均值移動機制;開跑前先 SMOKE=1,見 HANDOFF §5)。
- 待辦轉入 backlog:s12_v4e1 的 −6..9 Δwb/ILD 已算(+0.3679;ILD 表未列),寫 RESULTS_CHIP_COMPARISON §2.10 時一併補 per-SNR 表。

## 5. Reproduction

```bash
# sweep(已完成;冪等可重跑)
cd /raid/R12K41024/20_pesq_finetune
bash run_gate_sweep_train.sh; bash run_gate_sweep_screen.sh
# rescue(已完成)
bash run_gate_rescue.sh
# 判決檔
cat logs/gate_sweep/canonical_summary.tsv logs/gate_sweep/rescue_summary.tsv
ls logs/gate_sweep/NO_GATE_PASS logs/gate_sweep/RESCUE_NO_PASS
# canonical eval 指令族(單顆)
cd /raid/R12K41024/18_w4a8_pow2_mixed_lmul_hw_handoff_20260530
CKPT=<ckpt> TAG=<tag> VCTK_ONLY=0 METRIC_WORKERS=32 GPU_ID=<free> \
ACT_FP8_MODE=calibrated_pow2_e4m3 \
BCCTN_ACTIVATION_CALIB_PATH=$PWD/ckpt/activation_calibration_p999_b64.json \
bash run_apot_descale_eval.sh
# Δwb/Δnb(-6..9)計算:per-SNR CSV 之 PESQ−PESQ_noisy 均值(nb 欄名新舊有異:PESQ_noisy_nb vs PESQ_nb_noisy)
```
