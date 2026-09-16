#!/usr/bin/env python
"""Experiment A + C: weight-shared bilateral ablation, and ITD-based localization.

A -- BILATERAL ABLATION (no retraining)
   The binaural coupling in BinauralAttentionDCNN is confined to one line:
       enc_combined = cat([enc_L, enc_R]) -> self.mattn(...)
   The encoder/decoder are weight-shared and run per-ear. So feeding the model a
   DUPLICATED single ear removes every cross-ear information path while keeping
   all weights identical:
       L_enh = model(stack([L, L]))[:, 0]
       R_enh = model(stack([R, R]))[:, 0]
   This is a stricter control than training a separate monaural model (as in the
   Westhausen TASLP'24 comparison): only the information path changes, not the
   weights or the training.

C -- LOCALIZATION (replicates the Goli & van de Par TASLP'23 chart *type*)
   Ground-truth azimuth is in every filename (`_az<deg>`). We build an empirical
   ITD -> azimuth calibration from the CLEAN signals (so the dataset's own HRIR
   rendering defines the mapping -- no analytic head model), then estimate the
   azimuth of each processed signal by GCC-PHAT ITD and report mean |error| and
   the fraction within a tolerance.

Usage (see run_bilateral_ablation.sh):
   python eval_bilateral_ablation.py --model_checkpoint <ckpt> \
       --test_dir ... --clean_dir ... --output_dir ... --limit_pairs 600 \
       --metric_workers 48 --hydra_overrides <deploy overrides>
"""
import argparse
import json
import os
import re
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
import soundfile as sf
import torch
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import eval as ev  # noqa: E402  (reuse the audited metric implementations)

SR = 16000
CONFIGS = ("noisy", "binaural", "bilateral")

# The TRUE §3.4 deploy path, mirroring run_apot_descale_eval.sh. Kept here (not
# only in the shell wrapper) so other scripts -- e.g. paper_figures/ -- can build
# the exact deploy model without duplicating the list.
P2_INDEX = "/raid/R12K41024/18_w4a8_pow2_mixed_lmul_hw_handoff_20260530/ckpt/p2_skip_indices_g42_topk_l1_r050.pt"
HW_OVERRIDES = [
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


# ---------------------------------------------------------------- ITD (raw)
def gcc_phat_itd_us(x, y, sr=SR, max_lag_ms=1.0):
    """Raw interaural time delay in microseconds (GCC-PHAT, parabolic peak).

    Same algorithm as eval.compute_itd_error's inner helper, but returns the
    signed delay itself rather than |clean - processed|, because the azimuth
    estimate needs the absolute value.
    """
    x = np.asarray(x, np.float64)
    y = np.asarray(y, np.float64)
    max_lag = max(1, int(max_lag_ms * 1e-3 * sr))
    n = 1
    while n < len(x) + len(y):
        n <<= 1
    X = np.fft.rfft(x, n=n)
    Y = np.fft.rfft(y, n=n)
    R = X * np.conj(Y)
    R /= (np.abs(R) + 1e-12)
    cc = np.fft.irfft(R, n=n)
    cc = np.concatenate((cc[-max_lag:], cc[:max_lag + 1]))
    k = int(np.argmax(cc))
    if 0 < k < len(cc) - 1:
        y0, y1, y2 = cc[k - 1], cc[k], cc[k + 1]
        denom = y0 - 2 * y1 + y2
        delta = 0.5 * (y0 - y2) / denom if abs(denom) > 1e-12 else 0.0
    else:
        delta = 0.0
    return ((k - max_lag) + delta) / sr * 1e6


# ------------------------------------------------------- per-pair metrics
def _metrics_for_payload(payload):
    """Process-pool safe: numpy in, dict out."""
    clean = payload["clean"]          # (N, 2)
    az = payload["az"]
    out = {"filename": payload["filename"], "az_true": az, "snr": payload["snr"]}

    cl, cr = clean[:, 0], clean[:, 1]
    out["ITD_raw_clean"] = gcc_phat_itd_us(cl, cr)

    for cfg in CONFIGS:
        sig = payload[cfg]            # (2, N)
        pl, pr = sig[0], sig[1]
        try:
            m = ev.safe_mbstoi(cl, cr, pl, pr, sr=SR)
            out[f"MBSTOI_{cfg}"] = m if m is not None else np.nan
        except Exception:
            out[f"MBSTOI_{cfg}"] = np.nan
        try:
            out[f"ILD_{cfg}"] = ev.compute_ild_error_with_mask(cl, cr, pl, pr, sr=SR)
        except Exception:
            out[f"ILD_{cfg}"] = np.nan
        try:
            ipd = ev.compute_ipd_error_with_mask(cl, cr, pl, pr, sr=SR)
            out[f"IPD_{cfg}"] = np.degrees(ipd) if ipd is not None else np.nan
        except Exception:
            out[f"IPD_{cfg}"] = np.nan
        try:
            out[f"ITDerr_{cfg}"] = ev.compute_itd_error(cl, cr, pl, pr, sr=SR)
        except Exception:
            out[f"ITDerr_{cfg}"] = np.nan
        try:
            s = ev.safe_stoi(cl, cr, pl, pr, sr=SR)
            out[f"STOI_{cfg}"] = s if s is not None else np.nan
        except Exception:
            out[f"STOI_{cfg}"] = np.nan
        out[f"ITD_raw_{cfg}"] = gcc_phat_itd_us(pl, pr)
    return out


# ------------------------------------------------------------- model I/O
def load_model(ckpt, overrides, device):
    from hydra import compose, initialize_config_dir
    from hydra.core.global_hydra import GlobalHydra

    here = os.path.dirname(os.path.abspath(__file__))
    GlobalHydra.instance().clear()
    initialize_config_dir(config_dir=os.path.join(here, "config"), version_base=None)
    config = compose(config_name="config", overrides=list(overrides or []))

    from DCNN.trainer import DCNNLightningModule
    model = DCNNLightningModule(config)
    model.eval()

    _IGNORE = ("loss.", "disc.", "base_loss.")
    sd = torch.load(ckpt, map_location="cpu")["state_dict"]
    sd = {k: v for k, v in sd.items() if not k.startswith(_IGNORE)}
    missing, unexpected = model.load_state_dict(sd, strict=False)
    bad_m = [k for k in missing if not k.startswith(_IGNORE)]
    bad_u = [k for k in unexpected if not k.startswith(_IGNORE)]
    if bad_m or bad_u:
        raise RuntimeError(f"weight mismatch: missing={bad_m} unexpected={bad_u}")
    return model.to(device)


def _fit_len(sig, n):
    """Trim/zero-pad a (2, M) signal to (2, n) so metric STFT shapes line up."""
    if sig.shape[1] == n:
        return sig
    if sig.shape[1] > n:
        return sig[:, :n]
    return np.pad(sig, ((0, 0), (0, n - sig.shape[1])))


@torch.no_grad()
def run_configs(model, noisy, device, n_out, cross_ear_probe=False):
    """noisy: (N, 2) float32 -> (2, n_out) arrays for binaural + bilateral.

    Bilateral = the SAME network applied to each ear on its own. Feeding a
    duplicated ear means the output is a function of that ear alone -- there is
    no path by which the other ear can influence it. We always read the left
    output pathway so that both ears go through an identical function, matching
    the BilateralWrapper convention in 21_mono_se_baselines.
    """
    x = torch.from_numpy(noisy.T.astype(np.float32)).unsqueeze(0).to(device)

    binaural = model(x)[0].cpu().numpy()

    dup_l = torch.stack([x[:, 0], x[:, 0]], dim=1)
    dup_r = torch.stack([x[:, 1], x[:, 1]], dim=1)
    out_l = model(dup_l)[0]
    out_r = model(dup_r)[0]

    probe = None
    if cross_ear_probe:
        # Does the binaural model actually USE the other ear? Perturb only the
        # right input and measure the change in the LEFT output. If this were
        # ~0 the model would already be effectively bilateral and the whole
        # ablation would be vacuous, so this is the load-bearing diagnostic.
        g = torch.Generator(device="cpu").manual_seed(0)
        noise = torch.randn(x[:, 1].shape, generator=g).to(x.device)
        noise = noise * x[:, 1].std()
        alt = torch.stack([x[:, 0], noise], dim=1)
        base_l = model(x)[0][0]
        alt_l = model(alt)[0][0]
        probe = {
            "max_abs_delta": float((base_l - alt_l).abs().max().item()),
            "rel_rms_delta": float(((base_l - alt_l).pow(2).mean().sqrt()
                                    / (base_l.pow(2).mean().sqrt() + 1e-12)).item()),
        }

    bilateral = torch.stack([out_l[0], out_r[0]], dim=0).cpu().numpy()
    return _fit_len(binaural, n_out), _fit_len(bilateral, n_out), probe


# --------------------------------------------------------------- azimuth
def build_itd_azimuth_lut(df):
    """Median clean ITD per ground-truth azimuth -> monotone lookup table."""
    g = df.groupby("az_true")["ITD_raw_clean"].median().sort_index()
    az = g.index.to_numpy(dtype=np.float64)
    itd = g.to_numpy(dtype=np.float64)
    order = np.argsort(itd)          # interp needs increasing x
    return itd[order], az[order]


def estimate_azimuth(itd_us, itd_grid, az_grid):
    return np.interp(itd_us, itd_grid, az_grid)


def localization_stats(df, cfg, itd_grid, az_grid, tol_deg):
    est = estimate_azimuth(df[f"ITD_raw_{cfg}"].to_numpy(), itd_grid, az_grid)
    err = np.abs(est - df["az_true"].to_numpy())
    return {
        "mean_abs_err_deg": float(np.nanmean(err)),
        "median_abs_err_deg": float(np.nanmedian(err)),
        "within_tol_pct": float(np.nanmean(err <= tol_deg) * 100.0),
        "tol_deg": float(tol_deg),
    }


# ------------------------------------------------------------------ main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_checkpoint", required=True)
    ap.add_argument("--test_dir", required=True)
    ap.add_argument("--clean_dir", required=True)
    ap.add_argument("--output_dir", required=True)
    ap.add_argument("--limit_pairs", type=int, default=600)
    ap.add_argument("--metric_workers", type=int, default=32)
    ap.add_argument("--tol_deg", type=float, default=10.0)
    ap.add_argument("--hydra_overrides", nargs="*", default=[])
    args = ap.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[ablation] device={device}")

    model = load_model(args.model_checkpoint, args.hydra_overrides, device)

    def parse_tag(stem):
        m = re.search(r'_az(-?\d+)(?:_snr([-+]?\d+(?:\.\d+)?))?$', stem)
        az = int(m.group(1)) if m else None
        snr = float(m.group(2)) if m and m.group(2) else None
        return stem.split('_az')[0], az, snr

    clean_dict = {}
    for f in Path(args.clean_dir).glob('**/*.wav'):
        base, az, _ = parse_tag(f.stem)
        if az is not None:
            clean_dict[(base, az)] = f
    noisy_dict = {}
    for f in Path(args.test_dir).glob('**/*.wav'):
        base, az, snr = parse_tag(f.stem)
        if az is not None:
            noisy_dict[(base, az, snr)] = f

    pairs = [(clean_dict[(b, a)], noisy_dict[(b, a, s)], a, s)
             for (b, a, s) in noisy_dict if (b, a) in clean_dict]
    pairs.sort(key=lambda t: str(t[0]))
    if 0 < args.limit_pairs < len(pairs):
        # stride-sample so every azimuth/SNR stays represented
        idx = np.linspace(0, len(pairs) - 1, args.limit_pairs).astype(int)
        pairs = [pairs[i] for i in idx]
    print(f"[ablation] {len(pairs)} pairs, azimuths={sorted({p[2] for p in pairs})}")

    payloads = []
    probe_result = None
    for i, (cpath, npath, az, snr) in enumerate(tqdm(pairs, desc="forward")):
        clean, _ = sf.read(str(cpath), dtype="float32", always_2d=True)
        noisy, _ = sf.read(str(npath), dtype="float32", always_2d=True)
        n = min(len(clean), len(noisy))
        clean, noisy = clean[:n], noisy[:n]

        binaural, bilateral, probe = run_configs(
            model, noisy, device, n, cross_ear_probe=(i == 0))
        if probe is not None:
            probe_result = probe
            print(f"[ablation] cross-ear sensitivity of the binaural model: "
                  f"max|delta L| = {probe['max_abs_delta']:.4e}, "
                  f"rel-RMS = {probe['rel_rms_delta']:.4%} "
                  f"({'binaural coupling is REAL' if probe['rel_rms_delta'] > 1e-3 else 'WARNING: coupling ~0, ablation vacuous'})")

        payloads.append({
            "filename": npath.name, "az": az, "snr": snr,
            "clean": clean,
            "noisy": noisy.T.copy(),
            "binaural": binaural,
            "bilateral": bilateral,
        })

    workers = max(1, args.metric_workers)
    rows = []
    if workers == 1:
        for p in tqdm(payloads, desc="metrics"):
            rows.append(_metrics_for_payload(p))
    else:
        with ProcessPoolExecutor(max_workers=workers) as ex:
            for r in tqdm(ex.map(_metrics_for_payload, payloads),
                          total=len(payloads), desc="metrics"):
                rows.append(r)

    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(args.output_dir, "per_file.csv"), index=False)

    itd_grid, az_grid = build_itd_azimuth_lut(df)

    summary = {"n_pairs": len(df), "cross_ear_probe": probe_result,
               "itd_azimuth_lut": {"itd_us": itd_grid.tolist(),
                                   "az_deg": az_grid.tolist()}}
    print("\n=== A: cue metrics (mean over all SNR) ===")
    hdr = f"{'config':>10} {'MBSTOI':>8} {'STOI':>8} {'ILD dB':>8} {'IPD deg':>8} {'ITDerr us':>10}"
    print(hdr)
    for cfg in CONFIGS:
        s = {k: float(np.nanmean(df[f"{k}_{cfg}"])) for k in
             ("MBSTOI", "STOI", "ILD", "IPD", "ITDerr")}
        summary[cfg] = s
        print(f"{cfg:>10} {s['MBSTOI']:>8.4f} {s['STOI']:>8.4f} {s['ILD']:>8.4f} "
              f"{s['IPD']:>8.2f} {s['ITDerr']:>10.2f}")

    # Anechoic HRIR renderings localize easily, so a single loose tolerance
    # saturates at 100%; sweep tolerances to see where (if anywhere) the
    # configurations actually separate.
    tols = sorted({args.tol_deg, 1.0, 2.0, 5.0, 10.0})
    print("\n=== C: ITD-based localization ===")
    hdr_t = "".join(f"{'<=' + format(t, 'g') + 'deg %':>12}" for t in tols)
    print(f"{'config':>10} {'mean|err| deg':>14} {'median deg':>11}{hdr_t}")
    for cfg in CONFIGS:
        st = localization_stats(df, cfg, itd_grid, az_grid, args.tol_deg)
        est = estimate_azimuth(df[f"ITD_raw_{cfg}"].to_numpy(), itd_grid, az_grid)
        err = np.abs(est - df["az_true"].to_numpy())
        st["within_pct_by_tol"] = {format(t, 'g'): float(np.nanmean(err <= t) * 100.0)
                                   for t in tols}
        summary[f"loc_{cfg}"] = st
        cells = "".join(f"{st['within_pct_by_tol'][format(t, 'g')]:>12.1f}" for t in tols)
        print(f"{cfg:>10} {st['mean_abs_err_deg']:>14.2f} "
              f"{st['median_abs_err_deg']:>11.2f}{cells}")
    print("\nNote: the LUT is fitted on the clean signals, so a 'clean' row would "
          "score ~0 by construction and is omitted as uninformative.")

    with open(os.path.join(args.output_dir, "summary.json"), "w") as fh:
        json.dump(summary, fh, indent=2)
    print(f"\n[ablation] wrote {args.output_dir}/per_file.csv + summary.json")


if __name__ == "__main__":
    main()
