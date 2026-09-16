#!/usr/bin/env python3
"""Validation gate for the fp16 conv-PPU numerical model.

Reproduces the RTL side's chained e4m3 code-match table (REQUEST_FP16_PPU_EVAL_
2026-06-10): run the deploy model on a real t0 dual-ear sample, once with the
PPU in fp32 (deploy) and once with the PPU in fp16 (BCCTN_FP16_PPU=all), and
compare the e4m3-cast activation that crosses every encoder conv boundary,
per ear. If our reconstructed fp16 model reproduces the RTL's ~98% -> ~62%
degradation curve, the reconstruction is faithful (no RTL HDL needed).

The two ears are batch elements 0/1 (Stft folds channels into batch). Each
complex depthwise-separable block's real Conv2d is called with both .real and
.imag, so hooking depthwise_r + pointwise_r captures the full complex e4m3
boundary (block-input and dw->pw) for that stage.

Run from the package code dir under the bcctn env, e.g.:
  CUDA_VISIBLE_DEVICES=1 conda run --no-capture-output -n bcctn \
    python ../scripts/verify_fp16_ppu_chain.py
"""
from __future__ import annotations
import glob
import os
import sys

PKG = "/raid/R12K41024/18_w4a8_pow2_mixed_lmul_hw_handoff_20260530"
CODE = os.path.join(PKG, "code")
CKPT = os.path.join(PKG, "ckpt",
                    "strict_core_pow2_blockfloat_deploy_ep3_vctk0.8357_timit0.7898.ckpt")
P2_INDEX = os.path.join(PKG, "ckpt", "p2_skip_indices_g42_topk_l1_r050.pt")
CALIB = os.path.join(PKG, "ckpt", "activation_calibration_p999_b64.json")
VCTK_NOISY = "/raid/R12K41024/BCCTN/Dataset/noisy_testset"

# --- deploy env (mirror run_apot_descale_eval.sh + the pow2 deploy invocation) ---
os.environ.setdefault("NUMBA_CACHE_DIR", "/tmp/numba_cache")
os.environ.setdefault("MPLCONFIGDIR", "/tmp/mpl_cache")
os.environ["BCCTN_APOT_A_MIN"] = "-16"; os.environ["BCCTN_APOT_A_MAX"] = "-1"
os.environ["BCCTN_APOT_B_MIN"] = "-16"; os.environ["BCCTN_APOT_B_MAX"] = "-1"
os.environ["BCCTN_ACTIVATION_FP8_MODE"] = "calibrated_pow2_e4m3"
os.environ["BCCTN_ACTIVATION_CALIB_PATH"] = CALIB
os.environ["BCCTN_OUTLIER_DELAYED_DESCALE"] = "1"
os.environ["BCCTN_OUTLIER_DESCALE_APOT_MIN"] = "-16"
os.environ["BCCTN_OUTLIER_DESCALE_APOT_MAX"] = "-6"
os.environ.pop("BCCTN_FP16_PPU", None)  # toggled in-process via set_fp16_ppu

sys.path.insert(0, CODE)
os.chdir(CODE)

import numpy as np  # noqa: E402
import soundfile as sf  # noqa: E402
import torch  # noqa: E402

OVERRIDES = [
    "training.qat_scope=full",
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
    "model.low_power.mha_projection.enabled=true",
    "model.low_power.mha_projection.qkv_groups=4",
    "model.decoder_grouped_pointwise.mode=grouped",
    "model.decoder_grouped_pointwise.groups_x=4",
    "model.decoder_grouped_pointwise.groups_skip=2",
    "model.decoder_grouped_pointwise.skip_select_ratio=0.5",
    "model.decoder_grouped_pointwise.skip_select_mode=topk_l1",
    f"model.decoder_grouped_pointwise.skip_indices_path={P2_INDEX}",
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


def build_model(device):
    from hydra import compose, initialize_config_dir
    from hydra.core.global_hydra import GlobalHydra
    if GlobalHydra().is_initialized():
        GlobalHydra.instance().clear()
    initialize_config_dir(config_dir=os.path.join(CODE, "config"), version_base=None)
    config = compose(config_name="config", overrides=list(OVERRIDES))
    from DCNN.trainer import DCNNLightningModule
    model = DCNNLightningModule(config)
    model.eval()
    ckpt = torch.load(CKPT, map_location="cpu")
    model.load_state_dict(ckpt["state_dict"])
    model.to(device).eval()
    return model


def load_t0(device):
    files = sorted(glob.glob(os.path.join(VCTK_NOISY, "**", "*.wav"), recursive=True))
    for f in files:
        data, sr = sf.read(f)
        if data.ndim == 2 and data.shape[1] == 2 and len(data) / sr >= 0.3:
            x = torch.tensor(data.T, dtype=torch.float32).unsqueeze(0).to(device)
            return os.path.basename(f), x
    raise RuntimeError("no stereo t0 found")


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    import DCNN.utils.complexPyTorch.complexLayers as L
    model = build_model(device)
    name, x = load_t0(device)
    print(f"[gate] t0 = {name}  input {tuple(x.shape)}  device={device}")

    # Hook the real Conv2d of each encoder dw-sep block: depthwise_r captures the
    # block-input complex e4m3 boundary; pointwise_r captures the dw->pw boundary.
    captures = {}
    order = []         # encoder boundaries (depthwise_r + pointwise_r)
    dec_order = []     # decoder dw-input boundaries (depthwise_r per stage)
    enc_targets, dec_targets = [], []
    for mod_name, m in model.named_modules():
        if not isinstance(m, (torch.nn.Conv2d, torch.nn.ConvTranspose2d)):
            continue
        if "encoder" in mod_name and (mod_name.endswith("depthwise_r")
                                      or mod_name.endswith("pointwise_r")):
            enc_targets.append((mod_name, m))
        elif "decoder" in mod_name and mod_name.endswith("depthwise_r"):
            # input to decoder depthwise = post-pointwise merged activation
            # (= RTL's "dw-input dump", the FP8-outlier corruption probe)
            dec_targets.append((mod_name, m))
    enc_targets.sort(key=lambda t: [int(p) if p.isdigit() else p
                                    for p in t[0].replace("depthwise_r", "0_dw")
                                    .replace("pointwise_r", "1_pw").split(".")])
    dec_targets.sort(key=lambda t: [int(p) if p.isdigit() else p for p in t[0].split(".")])
    targets = enc_targets + dec_targets

    def make_cap(nm):
        def h(module, args):
            captures.setdefault(nm, []).append(args[0].detach().to("cpu"))
        return h

    for nm, m in targets:
        m.register_forward_pre_hook(make_cap(nm))
    order = [nm for nm, _ in enc_targets]
    dec_order = [nm for nm, _ in dec_targets]

    def run(segs):
        L.set_fp16_ppu(segs)
        captures.clear()
        with torch.no_grad():
            model(x)
        return {k: [t.clone() for t in v] for k, v in captures.items()}

    A = run(frozenset())                              # deploy fp32 PPU
    # sanity: deploy run must be e4m3-quantized (few unique values)
    s0 = A[order[0]][0]
    print(f"[gate] sanity boundary0 unique re-values (deploy) = "
          f"{torch.unique(s0.real).numel()} (expect <=256 => e4m3)\n")

    combos = [
        ("all  (tree+bn+prelu)", frozenset({"tree", "bn", "prelu"})),
        ("tree only",            frozenset({"tree"})),
        ("bn+prelu (tree fp32)", frozenset({"bn", "prelu"})),
        ("bn only",              frozenset({"bn"})),
        ("prelu only",           frozenset({"prelu"})),
    ]

    def match(A, B, names):
        rows = []
        for nm in names:
            la, lb = A[nm], B[nm]
            assert len(la) == 4 and len(lb) == 4, f"{nm}: {len(la)} calls"
            per_ear = []
            for cs in ((0, 1), (2, 3)):  # ear0 = L.re,L.im ; ear1 = R.re,R.im
                tot = eq = 0
                for c in cs:
                    eq += int((la[c] == lb[c]).sum().item()); tot += la[c].numel()
                per_ear.append(eq / max(tot, 1))
            rows.append((nm.split("model.")[-1], per_ear[0], per_ear[1]))
        return rows

    # RTL reference checkpoints (request): {L_index: (ear0, ear1)}
    rtl = {0: (98.4, 100.0), 3: (96.5, 94.1), 5: (91.6, 70.9),
           7: (69.7, 54.5), 9: (65.4, 58.0), 10: (62.5, 60.9)}
    cps = [0, 3, 5, 7, 9, 10]

    print("Per-segment chain code-match vs deploy (ear0 / ear1 %). "
          "Checkpoints L0,L3,L5,L7,L9,L10(=enc5 dw):")
    hdr = "  combo                    " + "".join(f"  L{c:<10}" for c in cps)
    print(hdr); print("  " + "-" * (len(hdr) - 2))
    print("  RTL reference (request)  " +
          "".join(f"  {rtl[c][0]:4.1f}/{rtl[c][1]:4.1f}" for c in cps))
    B_all = run(frozenset({"tree", "bn", "prelu"}))
    for label, segs in combos:
        rows = match(A, B_all if label.startswith("all") else run(segs), order)
        cells = "".join(f"  {rows[c][1]*100:4.1f}/{rows[c][2]*100:4.1f}" for c in cps)
        print(f"  {label:<24}{cells}")

    print("\n[gate] full per-L ENCODER curve ('all' fp16-PPU):")
    rows = match(A, B_all, order)
    for i, (short, e0, e1) in enumerate(rows):
        print(f"  L{i:<3} {short:<42} {e0*100:6.1f}%  {e1*100:6.1f}%")

    # ---- DECODER dw-input boundaries: does our fp16-PPU reproduce the RTL's
    #      FP8-outlier (S0/S3/S4) overflow corruption, or leave them benign? ----
    print("\nDECODER dw-input code-match vs deploy ('all' fp16-PPU). "
          "RTL: S0/S3/S4 (FP8-outlier) = 42.2/61.7/70.3%; S1/S2/S5 (INT4) = 98.4/98.8/100%.")
    drows = match(A, B_all, dec_order)
    for nm, (short, e0, e1) in zip(dec_order, drows):
        stage = short.split(".")[0]
        kind = "FP8-outlier" if stage in ("0", "3", "4") else "INT4 split-k"
        print(f"  S{stage:<2} {kind:<12} {short:<30} {e0*100:6.1f}%  {e1*100:6.1f}%")
    L.set_fp16_ppu(frozenset())


if __name__ == "__main__":
    main()
