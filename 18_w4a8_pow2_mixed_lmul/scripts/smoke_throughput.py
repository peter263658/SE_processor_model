"""Throughput / OOM sweep for the *real* BCCTN model under L-Mul.

Builds the actual DCNNLightningModule (so conv/MHA/L-Mul patch + mixed-precision
QAT match training), runs a few fwd+bwd+opt steps at several batch sizes, and
reports peak GPU memory, step time, throughput (samples/s) and an estimated
per-epoch *train* wall (validation excluded). Flags OOM per batch.

Why: the L-Mul matmul materialises a [B, S, out, in] outer product before
summation, so memory AND compute scale ~linearly with batch. This smoke
quantifies whether a larger batch buys any throughput or only risks OOM —
i.e. whether the workload is compute-bound (then speed-ups must come from
fewer epochs / less validation / cheaper L-Mul, not from batch size).

Usage:
  cd /raid/R12K41024/18_w4a8_pow2_mixed_lmul
  CUDA_VISIBLE_DEVICES=<free GPU> conda run --no-capture-output -n bcctn \
      python scripts/smoke_throughput.py
  # options: --ste lmul_exact --mantissa 4 --batches 64,128,192,256 --steps 3
"""
from __future__ import annotations

import argparse
import os
import sys
import time

import torch

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

T_SAMPLES = 32000        # 2 s @ 16 kHz — matches the prepared VCTK/TIMIT files
N_TRAIN_SAMPLES = 30976  # scout train set (242 batches × 128); for epoch-wall est


def build_model(ste_mode: str, mantissa_bits: int, device: str):
    """Construct the real LightningModule with the *exact* recovery operating point.

    Overrides mirror run_combined_recovery.sh common_overrides verbatim (d1_qkv_g4
    + c3 packet+L-Mul + grouped decoder + linear_chunk_size=2048) so memory/compute
    match training. Only difference: KD is disabled (student forward + task loss are
    identical; the binaural KD term adds a cheap non-L-Mul loss eval and needs the
    teacher dir, so we skip it for a throughput probe).
    """
    from hydra import compose, initialize
    from hydra.core.global_hydra import GlobalHydra

    P2_INDEX = os.path.join(_REPO, "Checkpoints/p2_skip_indices_g42_topk_l1_r050.pt")
    m = str(mantissa_bits)
    overrides = [
        # training.* that change the numeric/compute path (must match wrapper):
        "training.qat_scope=full",            # conv too gets W4A8 QAT (not mha_only)
        "training.fp16_accumulator=true",     # MHA custom-forward fp16 accum path
        "training.int4_scale_format=fp32",
        "training.top_k_attention=4",
        "model.attention_softmax.mode=shift_topk_pow2_clamp8_nomantissa",
        "model.attention_datapath.enabled=true",
        "model.attention_datapath.score_quant=none",
        "model.attention_datapath.prob_quant=none",
        "model.attention_datapath.lmul_enabled=true",
        "model.attention_datapath.lmul_targets=[qk]",
        "model.attention_datapath.lmul_bit_exact=true",
        "model.attention_datapath.lmul_all_projection_ops=true",
        "model.attention_datapath.lmul_linear_chunk_size=2048",
        "model.attention_datapath.lmul_checkpoint_chunks=false",
        "model.attention_datapath.prob_packet.enabled=true",
        "model.attention_datapath.prob_packet.top_k=4",
        "model.attention_datapath.prob_packet.local_idx_bits=5",
        "model.attention_datapath.prob_packet.score_v_mode=lmul",
        "model.attention_datapath.prob_packet.validate_local_idx=true",
        "model.attention_datapath.prob_packet.prob_bits=8",
        "model.attention_datapath.prob_packet.prob_scale=256",
        "model.attention_datapath.prob_packet.prob_max_code=255",
        "model.attention_datapath.prob_packet.causal_window=null",
        "model.attention_datapath.prob_packet.renormalize=false",
        f"model.attention_datapath.lmul_mantissa_bits={m}",
        "model.decoder_grouped_pointwise.mode=grouped",
        "model.decoder_grouped_pointwise.groups_x=4",
        "model.decoder_grouped_pointwise.groups_skip=2",
        "model.decoder_grouped_pointwise.skip_select_ratio=0.5",
        "model.decoder_grouped_pointwise.skip_select_mode=topk_l1",
        f"model.decoder_grouped_pointwise.skip_indices_path={P2_INDEX}",
        "model.low_power.mha_projection.enabled=true",
        "model.low_power.mha_projection.qkv_groups=4",
        "model.low_power.all_lmul_multiply.enabled=true",
        "model.low_power.all_lmul_multiply.bit_exact=true",
        "model.low_power.all_lmul_multiply.linear_chunk_size=2048",
        "model.low_power.all_lmul_multiply.checkpoint_chunks=false",
        "model.low_power.all_lmul_multiply.learnable_bias_correction.enabled=true",
        "model.low_power.all_lmul_multiply.learnable_bias_correction.init_value=0.0",
        f"model.low_power.all_lmul_multiply.ste_mode={ste_mode}",
        f"model.low_power.all_lmul_multiply.mantissa_bits={m}",
        f"model.low_power.all_lmul_multiply.attention_mantissa_bits={m}",
        "model.kd.enabled=false",
    ]
    GlobalHydra.instance().clear()
    # config_path is resolved relative to *this* file (scripts/), so step up one.
    initialize(config_path="../config", version_base=None)
    cfg = compose(config_name="config", overrides=overrides)

    from DCNN.trainer import DCNNLightningModule
    model = DCNNLightningModule(cfg).to(device)
    model.train()
    return model


def time_batch(model, batch: int, device: str, n_steps: int = 3):
    """Run n_steps of student fwd + composite loss + bwd + opt; return (mean_step_s, peak_GB)."""
    opt = torch.optim.Adam(model.parameters(), lr=1e-6)
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)
    times = []
    for i in range(n_steps):
        x = torch.randn(batch, 2, T_SAMPLES, device=device)
        y = torch.randn(batch, 2, T_SAMPLES, device=device)
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        opt.zero_grad(set_to_none=True)
        out = model.model(x)          # patched student forward (the L-Mul-heavy path)
        loss = model.loss(out, y)     # SNR+STOI+ILD+IPD composite
        loss.backward()
        opt.step()
        torch.cuda.synchronize()
        dt = time.perf_counter() - t0
        if i > 0:                     # drop step 0 (allocation / autotune warmup)
            times.append(dt)
    peak = torch.cuda.max_memory_allocated(device) / 1e9
    return (sum(times) / len(times) if times else float("nan")), peak


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ste", default="lmul_exact")
    ap.add_argument("--mantissa", type=int, default=4)
    ap.add_argument("--batches", default="64,128,192,256")
    ap.add_argument("--steps", type=int, default=3)
    args = ap.parse_args()

    if not torch.cuda.is_available():
        print("FAIL: CUDA not available", file=sys.stderr)
        sys.exit(2)
    device = "cuda"
    print(f"[throughput] device={torch.cuda.get_device_name(0)}")
    print(f"[throughput] ste={args.ste}  mantissa={args.mantissa}  "
          f"T={T_SAMPLES} (2s@16k)  steps={args.steps} ({args.steps - 1} timed)")
    print(f"[throughput] epoch-wall est uses N_train≈{N_TRAIN_SAMPLES} "
          f"samples (train only, no validation)\n")

    rows = []
    for B in (int(b) for b in args.batches.split(",")):
        model = None
        try:
            model = build_model(args.ste, args.mantissa, device)
            step_t, peak = time_batch(model, B, device, args.steps)
            sps = B / step_t
            epoch_min = (N_TRAIN_SAMPLES / sps) / 60.0
            rows.append((B, step_t, peak, sps, epoch_min))
            print(f"  B={B:4d}  step={step_t:6.2f}s  peak={peak:5.1f}GB  "
                  f"samples/s={sps:5.2f}  est_train_epoch={epoch_min:6.1f}min  OK")
        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                print(f"  B={B:4d}  OOM")
                torch.cuda.empty_cache()
            else:
                raise
        finally:
            del model
            torch.cuda.empty_cache()

    print()
    if not rows:
        print("[throughput] every batch OOM'd — lower batch or enable lmul chunking.")
        sys.exit(1)

    best = min(rows, key=lambda r: r[4])
    print(f"[throughput] fastest fitting: B={best[0]}  "
          f"~{best[4]:.0f} min/train-epoch  (peak {best[2]:.1f}GB)")
    sps_vals = [r[3] for r in rows]
    spread = max(sps_vals) / min(sps_vals)
    if spread < 1.15:
        print(f"[throughput] samples/s ~flat across batch (spread {spread:.2f}×) "
              f"→ COMPUTE-BOUND: bigger batch gives ~no speed-up, only OOM risk.")
        print("[throughput] real speed-ups: fewer epochs (skip/shorten curriculum), "
              "smaller/less-frequent validation, num_sanity_val_steps=0, cheaper L-Mul.")
    else:
        print(f"[throughput] samples/s rises with batch (spread {spread:.2f}×) "
              f"→ pick the largest batch that fits with memory margin.")


if __name__ == "__main__":
    main()
