#!/usr/bin/env python3
"""Qualitative examples: noisy input -> zero-multiplier W4A8 enhanced output.

Two binaural utterances from the VCTK paper-style test set, run through the
DEPLOY checkpoint on the TRUE §3.4 hardware path (same env + Hydra overrides as
run_apot_descale_eval.sh), plotted as log-magnitude spectrograms.

Layout: 3 rows (Noisy / Enhanced / Clean) x 4 columns (Ex-A L, Ex-A R,
Ex-B L, Ex-B R). Both ears are shown because the selling point is that noise is
removed *without* flattening the L/R level difference -- with only one ear on the
page a reader cannot see that.

Display STFT is 512-pt (hop 128), NOT the model's internal 128-pt transform: the
model's own resolution is too coarse to read harmonics visually. This is a
display choice only and is stated in the caption.

All 12 panels share one dB colour scale so panel-to-panel comparison is honest.
Colormap is 'magma' -- perceptually uniform with monotonic lightness; a rainbow
map (jet) would manufacture banding that is not in the data.

Run:  ~/miniforge3/envs/bcctn/bin/python make_spectrogram_examples_fig.py
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

PKG = Path("/raid/R12K41024/18_w4a8_pow2_mixed_lmul_hw_handoff_20260530")

# Hardware-path env MUST be set before the DCNN import chain is touched.
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")        # 2 files: CPU is plenty
os.environ.update({
    "BCCTN_APOT_A_MIN": "-16", "BCCTN_APOT_A_MAX": "-1",
    "BCCTN_APOT_B_MIN": "-16", "BCCTN_APOT_B_MAX": "-1",
    "BCCTN_ACTIVATION_FP8_MODE": "calibrated_pow2_e4m3",
    "BCCTN_ACTIVATION_CALIB_PATH": str(PKG / "ckpt" / "activation_calibration_p999_b64.json"),
    "BCCTN_OUTLIER_DELAYED_DESCALE": "1",
    "BCCTN_OUTLIER_DESCALE_APOT_MIN": "-16",
    "BCCTN_OUTLIER_DESCALE_APOT_MAX": "-6",
    "NUMBA_CACHE_DIR": os.environ.get("NUMBA_CACHE_DIR", "/tmp/numba_cache"),
    "MPLCONFIGDIR": os.environ.get("MPLCONFIGDIR", "/tmp/mpl_cache"),
})

import numpy as np
import soundfile as sf
import torch

from make_paper_figures import (BASE, COL2, INK, INK2, MUTED, OUT_SVG,
                                convert_emf, plt, save)

sys.path.insert(0, str(PKG / "code"))
import eval_bilateral_ablation as abl  # noqa: E402  (model loader + HW overrides)
import eval as ev                      # noqa: E402  (audited metric fns)

DATA = Path("/raid/R12K41024/BCCTN/Dataset")
CKPT = PKG / "ckpt" / "strict_core_pow2_blockfloat_deploy_ep3_vctk0.8357_timit0.7898.ckpt"
SR = 16000
NFFT, HOP = 512, 128
DB_FLOOR = 60.0          # dynamic range below the shared peak

# (stem, azimuth tag, SNR dir, SNR tag, human label)
# Chosen for near-100% speech activity (no long silent lead-in, which would
# waste half the panel) and OPPOSITE azimuths, so the L/R level asymmetry flips
# between the two examples -- that flip is the visual evidence that ILD survives.
# Screened from the dense-speech candidates for (a) a valid MBSTOI on BOTH the
# noisy and enhanced signal -- a few files hit the known negative-score edge case
# -- and (b) improvement on BOTH MBSTOI and ILD. Azimuths are opposite so the L/R
# level asymmetry flips between examples; that flip is the visual evidence ILD
# survives. Example A's +0.121 MBSTOI gain sits essentially on the -6 dB set
# average (+0.128); Example B's +0.114 is above the 0 dB average (+0.070).
EXAMPLES = [
    ("p276_p276_212", "az-50", "snr_-6dB", "snr-6.0", "-6 dB SNR"),
    ("p244_p244_023", "az50", "snr_0dB", "snr+0.0", "0 dB SNR"),
]


def load_pair(stem, aztag, snrdir, snrtag):
    clean, _ = sf.read(str(DATA / "clean_testset" / f"{stem}_{aztag}.wav"),
                       dtype="float32", always_2d=True)
    noisy, _ = sf.read(str(DATA / "noisy_testset" / snrdir / f"{stem}_{aztag}_{snrtag}.wav"),
                       dtype="float32", always_2d=True)
    n = min(len(clean), len(noisy))
    return clean[:n], noisy[:n]


def spec_db(x):
    """Log-magnitude spectrogram, (freq, time), in dB."""
    import librosa
    S = np.abs(librosa.stft(np.asarray(x, np.float32), n_fft=NFFT, hop_length=HOP))
    return 20.0 * np.log10(S + 1e-8)


def main():
    device = torch.device("cpu")
    model = abl.load_model(str(CKPT), abl.HW_OVERRIDES, device)
    print(f"  model loaded (hardware path, CPU)")

    panels, meta = [], []
    for stem, aztag, snrdir, snrtag, label in EXAMPLES:
        clean, noisy = load_pair(stem, aztag, snrdir, snrtag)
        with torch.no_grad():
            x = torch.from_numpy(noisy.T).unsqueeze(0)
            enh = model(x)[0].numpy()
        enh = abl._fit_len(enh, len(clean))

        cl, cr = clean[:, 0], clean[:, 1]
        m_noisy = ev.safe_mbstoi(cl, cr, noisy[:, 0], noisy[:, 1], sr=SR)
        m_enh = ev.safe_mbstoi(cl, cr, enh[0], enh[1], sr=SR)
        i_noisy = ev.compute_ild_error_with_mask(cl, cr, noisy[:, 0], noisy[:, 1], sr=SR)
        i_enh = ev.compute_ild_error_with_mask(cl, cr, enh[0], enh[1], sr=SR)

        panels.append({
            "Noisy": (spec_db(noisy[:, 0]), spec_db(noisy[:, 1])),
            "Enhanced": (spec_db(enh[0]), spec_db(enh[1])),
            "Clean": (spec_db(cl), spec_db(cr)),
        })
        az = aztag.replace("az", "") + "°"
        meta.append({"label": label, "az": az,
                     "mbstoi": (m_noisy, m_enh), "ild": (i_noisy, i_enh)})
        print(f"  {stem} {label}: MBSTOI {m_noisy:.3f}->{m_enh:.3f}, "
              f"ILD err {i_noisy:.2f}->{i_enh:.2f} dB")

    vmax = max(p[r][c].max() for p in panels for r in p for c in (0, 1))
    vmin = vmax - DB_FLOOR

    rows = ["Noisy", "Enhanced", "Clean"]
    fig, axes = plt.subplots(3, 4, figsize=(COL2, 4.35), sharex="col", sharey=True)
    for ei, (p, m) in enumerate(zip(panels, meta)):
        for ci, ear in enumerate(("L", "R")):
            col = ei * 2 + ci
            for ri, row in enumerate(rows):
                ax = axes[ri][col]
                S = p[row][ci]
                im = ax.imshow(S, origin="lower", aspect="auto", cmap="magma",
                               vmin=vmin, vmax=vmax,
                               extent=[0, S.shape[1] * HOP / SR, 0, SR / 2000.0])
                ax.grid(False)   # shared rcParams grid would draw over the image
                ax.tick_params(labelsize=6, color=BASE, length=2)
                for s in ax.spines.values():
                    s.set_color(BASE)
                    s.set_linewidth(0.6)
                if ri == 0:
                    ax.set_title(f"{ear}", fontsize=7.5, color=INK, pad=3)
                if col == 0:
                    ax.set_ylabel(row, fontsize=8, color=INK)
                if ri == 2:
                    ax.set_xlabel("Time (s)", fontsize=7, color=MUTED)

    # per-example headers + metric deltas, centred over each ear-pair
    for ei, m in enumerate(meta):
        x0 = axes[0][ei * 2].get_position().x0
        x1 = axes[0][ei * 2 + 1].get_position().x1
        fig.text((x0 + x1) / 2, 0.965,
                 f"Example {'AB'[ei]}   {m['label']},  azimuth {m['az']}",
                 ha="center", fontsize=8, color=INK, fontweight="bold")
        fig.text((x0 + x1) / 2, 0.932,
                 f"MBSTOI {m['mbstoi'][0]:.3f} → {m['mbstoi'][1]:.3f}    "
                 f"ILD err {m['ild'][0]:.2f} → {m['ild'][1]:.2f} dB",
                 ha="center", fontsize=6.8, color=INK2)

    axes[1][0].set_ylabel("Enhanced\n(W4A8 deploy)", fontsize=8, color=INK)
    fig.text(0.008, 0.5, "Frequency (kHz)", rotation=90, va="center",
             fontsize=8, color=INK)

    fig.subplots_adjust(left=0.105, right=0.90, top=0.895, bottom=0.085,
                        wspace=0.13, hspace=0.17)
    cax = fig.add_axes([0.915, 0.085, 0.016, 0.81])
    cb = fig.colorbar(im, cax=cax)
    cb.set_label("Magnitude (dB)", fontsize=7, color=INK2)
    cb.ax.tick_params(labelsize=6, color=BASE, length=2)
    cb.outline.set_edgecolor(BASE)
    cb.outline.set_linewidth(0.6)

    save("fig_spectrogram_examples")
    convert_emf()


if __name__ == "__main__":
    main()
