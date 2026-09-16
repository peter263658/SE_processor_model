# 23_official_bcctn_ckpt_eval — 官方 BCCTN checkpoint 在本 repo 測試協定下的分數

**目的**：把 VikasTokala/BCCTN 釋出的 paper checkpoint（11.09 M 參數）用**和 deploy 模型完全相同的 eval**
跑一次，讓投影片 / 論文的 FP32 參考點可以是「官方模型」而不是自製重現版（見
`18_.../docs/RESULTS_FP32_REFERENCE_IDENTITY_2026-09-16.md`：repo 裡的 0.8519 / 0.8535 都是自製架構）。

## 內容

| 路徑 | 說明 |
|---|---|
| `official_bcctn/` | **git submodule** → github.com/VikasTokala/BCCTN @ `53da450`（含 `DCNN/Checkpoints/Trained_model.ckpt`，md5 `991c7096…`，與 `/raid/R12K41024/BCCTN/DCNN/Checkpoints/Trained_model.ckpt` 相同） |
| `code/eval_official.py` | 用**官方程式碼本身**建 `BinauralAttentionDCNN`（`config/model.yaml` 只有 4 個 key，其餘全是官方 `DCNN.__init__` 預設：fft 512 / win 400 / hop 100，embed 1024，32 heads，kernel (5,2)），`load_state_dict(strict=True)`；之後把官方套件從 `sys.modules` 卸掉，改用 dir-18 的 `eval.py` 跑 pair 挑選、GPU 推論 + CPU metric worker pool、MBSTOI/STOI/PESQ/SegSNR/ILD/IPD/ITD/IC、paper-style 表 |
| `code/bcctn18_eval.py` | symlink → `../../18_w4a8_pow2_mixed_lmul_hw_handoff_20260530/code/eval.py`（spawn 出來的 worker 要能 `import bcctn18_eval`） |
| `code/mac_profile_official.py` | forward-hook 實際執行 MAC 計數：官方 / fft-512 重現版 / HW-sized，MAC/s 與 MAC/frame |
| `run_eval.sh` | `GPU_ID=<free> [VCTK_ONLY=0] [LIMIT=750] [METRIC_WORKERS=32] bash run_eval.sh` → `results/<TAG>/` |
| `docs/` | RESULTS 文件 |

dir-18 `eval.py` 為此多了一個 `evaluate_model_object(model, device, …)`（`evaluate_model_improved` 載入完模型後直接委派給它，行為不變），外部 wrapper 才能餵已建好的模型。

## 與 deploy eval 完全一致的部分

同一個 750-pair × 8 SNR VCTK subset + 160-pair TIMIT、同一套 metric 程式碼、同樣 32 個 metric worker。
Smoke（2 pairs）已驗證：同檔案的 `MBSTOI_noisy` / `SegSNR_noisy` / `ILD_noisy` 與 `18/results/apot_descale_eval/ft_pow2_ep3_full/` 逐 bit 相同。

## 注意

- 官方 ckpt 是在作者自己的 HRIR / 方位設定下訓練的，在本 repo 的測試集上屬 out-of-domain（見
  `20_pesq_finetune/docs/STATUS_CHIP_PAPER_COMPARISON_POSITIONING_2026-06-28.md`）；分數低於自製重現版時要註明這點，不是 metric 問題。
- 參數量以 tensor `numel` 計（complex64 的 `transform` Linear 算 1 個參數；若以實數計要 +1,049,600）。
