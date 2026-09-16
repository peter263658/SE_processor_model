"""Is MBSTOI able to see an ILD distortion at all?

MBSTOI's EC stage searches interaural level compensation gamma over +-20 dB and
interaural delay tau over +-1 ms, and keeps the *maximum* over that 100x40 grid
(MBSTOI/ec.py: `p.max(axis=0)` -> `tmp.max()`).  So a constant interaural level
offset should be compensated away and leave the score untouched.

This probe applies a known constant gain to the right ear of an already-processed
signal and reports MBSTOI and the eval's own ILD error side by side.
"""
import sys, glob, os, re
import numpy as np, soundfile as sf
from concurrent.futures import ProcessPoolExecutor

sys.path.insert(0, "/raid/R12K41024/21_mono_se_baselines/code")
os.chdir("/raid/R12K41024/21_mono_se_baselines/code")
from MBSTOI import mbstoi
from eval_bilateral import compute_ild_error_with_mask

D = "/raid/R12K41024/BCCTN/Dataset"
GAINS_DB = [0, 1, 3, 6, 10, 15, 20, 25, 30]


def one(args):
    npath, cpath, g_db = args
    n, _ = sf.read(npath, dtype="float32", always_2d=True)
    c, _ = sf.read(cpath, dtype="float32", always_2d=True)
    y = n.copy()
    y[:, 1] *= 10 ** (g_db / 20.0)          # constant ILD offset of g_db dB
    try:
        m = mbstoi(c[:, 0], c[:, 1], y[:, 0], y[:, 1])
    except Exception:
        return None
    ild = compute_ild_error_with_mask(c[:, 0], c[:, 1], y[:, 0], y[:, 1])
    return g_db, m, ild


def main():
    files = sorted(glob.glob(f"{D}/noisy_testset/snr_0dB/*.wav"))[:12]
    pairs = [(f, f"{D}/clean_testset/" + re.sub(r'_snr[-+0-9.]+\.wav$', '.wav', os.path.basename(f)))
             for f in files]
    pairs = [(n, c) for n, c in pairs if os.path.exists(c)]
    jobs = [(n, c, g) for g in GAINS_DB for n, c in pairs]
    with ProcessPoolExecutor(max_workers=8) as ex:
        res = [r for r in ex.map(one, jobs) if r]

    print(f"{len(pairs)} noisy pairs @ 0 dB, right ear scaled by a constant gain\n")
    print("| added ILD offset | MBSTOI | ILD error (dB) |")
    print("|---|---|---|")
    for g in GAINS_DB:
        ms = [m for gg, m, _ in res if gg == g]
        il = [i for gg, _, i in res if gg == g and i is not None and np.isfinite(i)]
        print(f"| {g:+d} dB | {np.mean(ms):.4f} | {np.mean(il):.3f} |")


if __name__ == "__main__":
    main()
