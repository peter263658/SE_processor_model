"""How much of MBSTOI is carried by magnitude vs phase?

Swaps magnitude and phase between the clean and noisy signals and scores the
result against the clean reference.  If MBSTOI is an envelope measure, then
clean-magnitude + noisy-phase should score near 1.0 and noisy-magnitude +
clean-phase should score near the noisy baseline.
"""
import sys, glob, os, re
import numpy as np, soundfile as sf, torch
from concurrent.futures import ProcessPoolExecutor

sys.path.insert(0, "/raid/R12K41024/21_mono_se_baselines/code")
os.chdir("/raid/R12K41024/21_mono_se_baselines/code")
from MBSTOI import mbstoi

D = "/raid/R12K41024/BCCTN/Dataset"
KW = dict(n_fft=512, hop_length=256, window=torch.hann_window(512), center=True)


def swap(mag_src, phase_src, n):
    """Rebuild a signal from one source's magnitude and another's phase."""
    a = torch.stft(torch.from_numpy(mag_src.T.copy()), return_complex=True, **KW)
    b = torch.stft(torch.from_numpy(phase_src.T.copy()), return_complex=True, **KW)
    y = torch.polar(a.abs(), torch.angle(b))
    return torch.istft(y, length=n, **KW).numpy()


def one(args):
    npath, cpath = args
    n, _ = sf.read(npath, dtype="float32", always_2d=True)
    c, _ = sf.read(cpath, dtype="float32", always_2d=True)
    L = len(c)
    cm_np = swap(c, n, L)       # clean magnitude + noisy phase
    nm_cp = swap(n, c, L)       # noisy magnitude + clean phase
    rt = swap(c, c, L)          # clean magnitude + clean phase (round-trip control)
    try:
        return (mbstoi(c[:, 0], c[:, 1], n[:, 0], n[:, 1]),
                mbstoi(c[:, 0], c[:, 1], rt[0], rt[1]),
                mbstoi(c[:, 0], c[:, 1], cm_np[0], cm_np[1]),
                mbstoi(c[:, 0], c[:, 1], nm_cp[0], nm_cp[1]))
    except Exception:
        return None


def main():
    files = sorted(glob.glob(f"{D}/noisy_testset/snr_0dB/*.wav"))[:12]
    pairs = [(f, f"{D}/clean_testset/" + re.sub(r'_snr[-+0-9.]+\.wav$', '.wav', os.path.basename(f)))
             for f in files]
    pairs = [(n, c) for n, c in pairs if os.path.exists(c)]
    with ProcessPoolExecutor(max_workers=8) as ex:
        res = np.array([r for r in ex.map(one, pairs) if r])
    lbl = ["noisy (reference point)", "clean mag + clean phase (round-trip)",
           "clean mag + NOISY phase", "NOISY mag + clean phase"]
    print(f"{len(res)} pairs @ 0 dB SNR, scored against the clean binaural reference\n")
    print("| signal | MBSTOI |")
    print("|---|---|")
    for i, l in enumerate(lbl):
        print(f"| {l} | {res[:, i].mean():.4f} |")


if __name__ == "__main__":
    main()
