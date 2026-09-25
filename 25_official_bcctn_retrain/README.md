# Official BCCTN retraining

Retrains the official BCCTN architecture on this project's dataset and evaluates it with the shared comparison protocol.

| Path | Purpose |
|---|---|
| `code/train_official.py` | Training entry point |
| `code/config_official.yaml` | Recorded training configuration |
| `docs/` | Versioned training metrics, checkpoint provenance, and VCTK/TIMIT result tables |
| `run_all.sh` | Resumable training, evaluation, result collection, and Git publication |

Initialize the [official BCCTN submodule](../23_official_bcctn_ckpt_eval/official_bcctn) and prepare the `bcctn` environment, datasets, and local paths before running. The evaluation entry point is `../23_official_bcctn_ckpt_eval/code/eval_official.py`.

`run_all.sh` uses workstation paths rooted at `/raid/R12K41024` and accepts `GPU_ID`, `TAG`, and `MAX_EPOCHS`. It resumes from the selected output directory's `checkpoints/last.ckpt`. **It also commits results and pushes to GitHub**, including updating `master`; inspect and adapt it before execution. Use the Python entry points for training/evaluation without those publication steps.

Checkpoints and full training outputs are excluded from Git. The files in `docs/` record previous runs and do not substitute for model weights.
