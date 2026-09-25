# Binaural Speech Enhancement — SE Processor Model

Research code for binaural speech enhancement and hardware-oriented BCCTN optimization, from floating-point baselines to a W4A8 neural core and RTL verification.

Based on **Binaural Speech Enhancement Using Deep Complex Convolutional Transformer Networks** (Tokala et al., ICASSP 2024). The model enhances stereo speech while evaluating spatial cues through ILD and IPD alongside intelligibility metrics.

## Start here

- **Hardware / RTL:** [19_algo_final](19_algo_final/README.md) contains the final handoff documentation, testbench patterns, SRAM images, and reference code. Start with its [hardware handoff](19_algo_final/docs/HANDOFF_HW_ZERO_MULTIPLIER_CORE.md).
- **Training and export:** [18_w4a8_pow2_mixed_lmul](18_w4a8_pow2_mixed_lmul/README.md) and the [hardware export package](18_w4a8_pow2_mixed_lmul_hw_handoff_20260530/README.md).
- **Experiment navigation:** [Project map](docs/PROJECT_MAP.md) groups the numbered experiments and reference implementations.
- **Comparisons:** [Monaural baselines](21_mono_se_baselines/README.md), [official BCCTN evaluation](23_official_bcctn_ckpt_eval/README.md), [parameter progression](24_param_mbstoi_progression/README.md), and [official BCCTN retraining](25_official_bcctn_retrain/README.md).
- **Research overview:** [English feature summary](PAPER_FEATURE_SUMMARY_2026-07-07.md) / [中文研究摘要](PAPER_FEATURE_SUMMARY_2026-07-07.zh.md).

## Clone

```bash
git clone --recurse-submodules https://github.com/peter263658/SE_processor_model.git
cd SE_processor_model
```

For an existing checkout, run `git submodule update --init --recursive`. The LBCCN and official BCCTN references are pinned Git submodules.

## Environment

The repository contains multiple experiment snapshots, not one installable Python package. Start with the recorded main environment:

```bash
conda env create -f bcctn.yml
conda activate bcctn
```

`mamba` can also create this environment. The exported environment targets Linux and includes Python 3.10, PyTorch 2.3.1, and CUDA dependencies; GPU drivers must be compatible. [bcctn_quant.yml](bcctn_quant.yml) records a separate quantization environment. Per-experiment `requirements.txt` files are historical snapshots and may pin different framework versions; consult the chosen experiment before installing them into an existing environment.

## Data, checkpoints, and execution

Datasets and trained checkpoints are local inputs and are not included in a fresh clone. Prepare the required clean/noisy binaural data and obtain the checkpoint for the selected experiment before training, evaluation, or regenerating exports.

Run commands from the experiment directory specified in its README. For the earlier sliding-window variant:

```bash
cd BCCTN_sep_conv_causal_slide_win
python train.py  # Hydra configuration in config/
```

Review dataset paths and training settings first. Some research scripts and configurations still contain workstation paths such as `/raid/R12K41024`; adapt those paths to your checkout. Hardware handoff patterns can be inspected without training, but regeneration requires the checkpoint and export tooling described in the handoff README.

**Script side effects:** some historical orchestration scripts (including `25_official_bcctn_retrain/run_all.sh` and `24_param_mbstoi_progression/finish_and_push.sh`) also commit and push results. Inspect them before running; use the individual Python entry points when only training or evaluation is intended.

## Hardware scope and results

The [final handoff](19_algo_final/README.md) reports VCTK / TIMIT MBSTOI of **0.8357 / 0.7898** for its W4A8 deploy point. These are recorded experiment results, not a fresh validation of this checkout. Its zero-general-multiplier claim applies to the neural inference core; STFT, iSTFT, windowing, and waveform mask application are outside that scope.

The repository retains source code, configurations, documentation, selected result tables, and hardware patterns. Datasets, checkpoints, training output, simulator output, and local tooling are excluded through `.gitignore`. Existing tracked research artifacts remain available; ignoring a path does not remove already tracked files.

## Citation

```bibtex
@inproceedings{tokala2024binaural,
  title={Binaural Speech Enhancement using Deep Complex Convolutional Transformer Networks},
  author={Tokala, Vikas and Grinstein, Eric and Brookes, Mike and Doclo, Simon and Jensen, Jesper and Naylor, Patrick A},
  booktitle={ICASSP 2024-2024 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP)},
  year={2024},
  organization={IEEE}
}
```

## Third-party licensing

The baseline includes an [MIT license](BCCTN/LICENSE). Check each vendored component and submodule for its own license and attribution. This repository does not assign a new blanket license to all research and hardware files.
