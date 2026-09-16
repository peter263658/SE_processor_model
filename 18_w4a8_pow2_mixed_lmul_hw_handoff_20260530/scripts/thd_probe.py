#!/usr/bin/env python
"""
THD / harmonic-distortion probe for the BCCTN binaural enhancement pipeline
(STFT -> network -> complex ratio mask -> iSTFT).

Hearing-aid electroacoustic standards (ANSI S3.22, IEC 60118-7) specify total
harmonic distortion with steady pure tones.  This script performs that
measurement through the whole chain and splits the result across three
conditions so distortion can be *attributed*:

  roundtrip : STFT -> identity mask -> iSTFT.  No network.  Front-end floor.
  fp32      : deploy checkpoint, FP32 arithmetic (all HW numerics OFF).
              Distortion here is algorithmic (time-varying complex mask).
  hw        : identical weights on the deployed zero-multiplier datapath
              (W4A8 + pow2 block-float activations + L-Mul + APoT BN/PReLU).
              (hw - fp32) isolates the quantization contribution.

Metrics per (frequency, level, channel):
  gain_db  : output/input level of the fundamental
  thd_pct  : sqrt(sum_{k>=2} P_k / P_1) * 100      -- harmonic bins only
  thdn_pct : sqrt((P_total - P_1) / P_1) * 100     -- harmonics + all other bins

Coherent sampling (integer cycles inside the analysis window) is used, so an
undistorted tone occupies exactly one FFT bin and the rectangular-window
leakage floor is numerically zero.  Anything off-bin in the output is genuine
added energy, not analysis artifact.
"""

import argparse
import json
import os
import re
import sys

import numpy as np
import torch

FS = 16000
CODE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "code")
CODE_DIR = os.path.abspath(CODE_DIR)
PKG = os.path.abspath(os.path.join(CODE_DIR, ".."))

# --------------------------------------------------------------------------
# Hydra override sets.  Architecture-shaping overrides (grouped decoder
# pointwise, MHA projection groups, skip indices) are IDENTICAL in both model
# modes -- only the numerics differ, so fp32 and hw run the same weights.
# --------------------------------------------------------------------------
P2_INDEX = os.path.join(PKG, "ckpt", "p2_skip_indices_g42_topk_l1_r050.pt")

ARCH_OVERRIDES = [
    "model.low_power.mha_projection.enabled=true",
    "model.low_power.mha_projection.qkv_groups=4",
    "model.decoder_grouped_pointwise.mode=grouped",
    "model.decoder_grouped_pointwise.groups_x=4",
    "model.decoder_grouped_pointwise.groups_skip=2",
    "model.decoder_grouped_pointwise.skip_select_ratio=0.5",
    "model.decoder_grouped_pointwise.skip_select_mode=topk_l1",
    f"model.decoder_grouped_pointwise.skip_indices_path={P2_INDEX}",
]

# Mirrors run_apot_descale_eval.sh exactly (the deployed hardware path).
HW_OVERRIDES = ARCH_OVERRIDES + [
    "training.qat_scope=full",
    "training.mixed_precision_qat=true",
    "training.top_k_attention=4",
    "training.fp16_accumulator=true",
    "training.int4_scale_format=apot_clip",
    "model.attention_softmax.mode=shift_topk_pow2_clamp8_nomantissa",
    "model.attention_datapath.enabled=true",
    "model.attention_datapath.score_quant=none",
    "model.attention_datapath.prob_quant=none",
    "model.attention_datapath.lmul_enabled=true",
    "model.attention_datapath.lmul_targets=[qk]",
    "model.attention_datapath.lmul_bit_exact=true",
    "model.attention_datapath.lmul_mantissa_bits=8",
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
    "model.low_power.all_lmul_multiply.enabled=false",
    "model.low_power.outlier_fp8_lmul.enabled=true",
    "model.low_power.outlier_fp8_lmul.mantissa_bits=4",
    "model.low_power.bn_affine.scale_mode=fold_apot",
    "model.low_power.bn_affine.pow2_min_exp=-9",
    "model.low_power.bn_affine.pow2_max_exp=4",
    "model.low_power.bn_affine.tie_real_imag_scale=false",
    "model.low_power.prelu_pow2.enabled=true",
    "model.low_power.prelu_pow2.min_exp=-8",
    "model.low_power.prelu_pow2.max_exp=0",
]

# Same weights, same graph, but every hardware approximation disabled.
FP32_OVERRIDES = ARCH_OVERRIDES + [
    "training.mixed_precision_qat=false",
    "training.fp8_qat=false",
    "training.top_k_attention=null",
    "training.fp16_accumulator=false",
    "model.attention_softmax.mode=softmax",
    "model.attention_datapath.enabled=false",
    "model.attention_datapath.prob_packet.enabled=false",
    "model.low_power.all_lmul_multiply.enabled=false",
    "model.low_power.outlier_fp8_lmul.enabled=false",
    "model.low_power.bn_affine.scale_mode=none",
    "model.low_power.prelu_pow2.enabled=false",
]

_IGNORE = ("loss.", "disc.", "base_loss.")


def build_model(ckpt_path, overrides, device):
    sys.path.insert(0, CODE_DIR)
    from hydra import compose, initialize_config_dir
    from hydra.core.global_hydra import GlobalHydra

    GlobalHydra.instance().clear()
    initialize_config_dir(config_dir=os.path.join(CODE_DIR, "config"),
                          version_base=None)
    config = compose(config_name="config", overrides=list(overrides))

    from DCNN.trainer import DCNNLightningModule
    model = DCNNLightningModule(config)
    model.eval()

    ck = torch.load(ckpt_path, map_location="cpu")
    sd = {k: v for k, v in ck["state_dict"].items() if not k.startswith(_IGNORE)}

    # The W4A8 QAT path registers fake-quant via torch.nn.utils.parametrize, so
    # the deploy ckpt stores weights as "<mod>.parametrizations.<w>.original".
    # The FP32 graph has no parametrization: strip the wrapper so BOTH modes run
    # bit-identical underlying weights and only the arithmetic differs.
    has_param = any(".parametrizations." in k for k in model.state_dict())
    if not has_param:
        pat = re.compile(r"\.parametrizations\.([^.]+)\.original$")
        sd = {pat.sub(r".\1", k): v for k, v in sd.items()}

    missing, unexpected = model.load_state_dict(sd, strict=False)
    missing = [k for k in missing if not k.startswith(_IGNORE)]
    unexpected = [k for k in unexpected if not k.startswith(_IGNORE)]
    if missing or unexpected:
        raise RuntimeError(f"weight mismatch: missing={missing} unexpected={unexpected}")
    return model.to(device)


# --------------------------------------------------------------------------
# stimulus + analysis
# --------------------------------------------------------------------------
def make_tone(f0, level_dbfs, n_total):
    t = np.arange(n_total, dtype=np.float64) / FS
    amp = 10.0 ** (level_dbfs / 20.0) * np.sqrt(2.0)   # level_dbfs = RMS level
    return (amp * np.sin(2.0 * np.pi * f0 * t)).astype(np.float32)


def power_spectrum(seg):
    """One-sided power spectrum whose sum equals the mean square of `seg`."""
    n = len(seg)
    X = np.fft.rfft(seg)
    P = (np.abs(X) / n) ** 2
    P[1:-1 if n % 2 == 0 else None] *= 2.0
    return P


def tone_metrics(x_in, x_out, f0, n_ana, offset, hw_bins=2):
    """Return dict of fundamental / harmonic / residual powers for one channel."""
    seg_in = x_in[offset:offset + n_ana].astype(np.float64)
    seg_out = x_out[offset:offset + n_ana].astype(np.float64)
    Pi = power_spectrum(seg_in)
    Po = power_spectrum(seg_out)

    df = FS / n_ana
    k1 = int(round(f0 / df))
    assert abs(k1 * df - f0) < 1e-9, f"f0={f0} not bin-exact at df={df}"

    def band(P, k):
        lo, hi = max(0, k - hw_bins), min(len(P) - 1, k + hw_bins)
        return P[lo:hi + 1].sum()

    p1_in = band(Pi, k1)
    p1_out = band(Po, k1)

    harm = {}
    p_harm = 0.0
    k = 2
    while k * f0 < FS / 2 - df:
        kk = int(round(k * f0 / df))
        pk = band(Po, kk)
        harm[k] = pk
        p_harm += pk
        k += 1

    p_total = Po[1:].sum()          # exclude DC, counted separately
    p_dc = Po[0]
    p_resid = max(p_total - p1_out, 0.0)          # everything except fundamental

    eps = 1e-30
    return {
        "gain_db": 10 * np.log10((p1_out + eps) / (p1_in + eps)),
        "out_level_dbfs": 10 * np.log10(p1_out + eps),
        "thd_pct": 100.0 * np.sqrt(p_harm / (p1_out + eps)),
        "thdn_pct": 100.0 * np.sqrt(p_resid / (p1_out + eps)),
        "harm_pct": {str(k): 100.0 * np.sqrt(v / (p1_out + eps))
                     for k, v in harm.items()},
        "dc_dbfs": 10 * np.log10(p_dc + eps),
    }


def run_model(model, x_stereo, device, chunk=None):
    """x_stereo: [2, N] float32 -> [2, N] float32"""
    with torch.no_grad():
        t = torch.from_numpy(x_stereo).unsqueeze(0).to(device)
        y = model(t).cpu().numpy()[0].astype(np.float32)
    n = x_stereo.shape[1]
    if y.shape[1] > n:
        y = y[:, :n]
    elif y.shape[1] < n:
        y = np.pad(y, ((0, 0), (0, n - y.shape[1])))
    return y


def run_roundtrip(model, x_stereo, device):
    """STFT -> iSTFT with an identity mask (network bypassed)."""
    import torch.nn.functional as Fn
    core = model.model
    with torch.no_grad():
        t = torch.from_numpy(x_stereo).unsqueeze(0).to(device)
        chans = []
        for ch in range(t.shape[1]):
            spec = core.stft(t[:, ch])            # [B, F-1, T], DC bin dropped
            # identity mask: re-pad the DC bin the same way apply_mask() does
            re = Fn.pad(spec.real, [0, 0, 1, 0])
            im = Fn.pad(spec.imag, [0, 0, 1, 0])
            chans.append(core.istft(torch.complex(re, im)))
        wav = torch.stack(chans, dim=1).cpu().numpy()[0].astype(np.float32)
    n = x_stereo.shape[1]
    if wav.shape[1] > n:
        wav = wav[:, :n]
    elif wav.shape[1] < n:
        wav = np.pad(wav, ((0, 0), (0, n - wav.shape[1])))
    return wav


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", required=True, choices=["roundtrip", "fp32", "hw"])
    ap.add_argument("--ckpt", default=os.path.join(
        PKG, "ckpt", "strict_core_pow2_blockfloat_deploy_ep3_vctk0.8357_timit0.7898.ckpt"))
    ap.add_argument("--freqs", default="500,800,1000,1600,2000")
    ap.add_argument("--levels", default="-40,-30,-20,-10")
    ap.add_argument("--dur", type=float, default=3.0)
    ap.add_argument("--ana_sec", type=float, default=2.0)
    ap.add_argument("--skip_sec", type=float, default=0.5)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    freqs = [float(f) for f in args.freqs.split(",")]
    levels = [float(l) for l in args.levels.split(",")]
    n_total = int(args.dur * FS)
    n_ana = int(args.ana_sec * FS)
    offset = int(args.skip_sec * FS)

    # roundtrip still needs a model instance for its stft/istft modules; use the
    # fp32 graph (identical front end in both numeric modes).
    overrides = FP32_OVERRIDES if args.mode in ("fp32", "roundtrip") else HW_OVERRIDES
    model = build_model(args.ckpt, overrides, device)

    results = []
    for f0 in freqs:
        for lv in levels:
            mono = make_tone(f0, lv, n_total)
            x = np.stack([mono, mono], axis=0)
            y = (run_roundtrip(model, x, device) if args.mode == "roundtrip"
                 else run_model(model, x, device))
            per_ch = [tone_metrics(x[c], y[c], f0, n_ana, offset) for c in range(2)]
            rec = {
                "mode": args.mode, "f0": f0, "level_dbfs": lv,
                "L": per_ch[0], "R": per_ch[1],
                "thd_pct": float(np.mean([p["thd_pct"] for p in per_ch])),
                "thdn_pct": float(np.mean([p["thdn_pct"] for p in per_ch])),
                "gain_db": float(np.mean([p["gain_db"] for p in per_ch])),
            }
            results.append(rec)
            print(f"[{args.mode}] f0={f0:6.0f} Hz  in={lv:+5.0f} dBFS  "
                  f"gain={rec['gain_db']:+7.2f} dB  "
                  f"THD={rec['thd_pct']:8.4f} %  THD+N={rec['thdn_pct']:9.3f} %",
                  flush=True)

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w") as fh:
        json.dump({"mode": args.mode, "ckpt": args.ckpt, "fs": FS,
                   "dur_s": args.dur, "ana_s": args.ana_sec,
                   "results": results}, fh, indent=2)
    print(f"[{args.mode}] wrote {args.out}")


if __name__ == "__main__":
    main()
