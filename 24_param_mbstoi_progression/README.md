# Parameter count vs. MBSTOI progression

Evaluation of historical model variants for the parameter-count / intelligibility comparison.

- [Recorded comparison points](docs/RESULTS_PARAM_VS_MBSTOI_POINTS_2026-09-17.md)
- [Generated point summary](docs/RESULTS_POINTS_AUTO.md)
- `docs/`: versioned model information and VCTK/TIMIT evaluation tables.
- `code/eval_legacy_ckpt.py`: evaluation entry point for historical checkpoints.
- `code/legacy_k52/`: source snapshot for the legacy architecture.
- `run_point.sh`: evaluates one point using `TAG`, `CODE_DIR`, `RUN_DIR`, and `CKPT` environment variables.
- `run_missing_points.sh`: batch orchestration for missing points.

Checkpoints, Hydra run directories, and datasets must be supplied locally. Shell scripts contain workstation-specific paths; adapt them before use. `finish_and_push.sh` includes Git publication steps, so inspect it before execution.
