#!/usr/bin/env python
"""Build a directional-interferer test set on the deploy考卷.

The existing `noisy_testset` mixes the target with **isotropic** noise: 37
uncorrelated sources every 5 degrees across the whole azimuthal plane
(BCCTN/prepare_data.py:create_isotropic_noise).  A diffuse field is the case
where binaural processing has the least to exploit -- there is no interferer
direction to null -- so it cannot show whether a binaural model does anything a
per-ear mono model cannot.

This script rebuilds the same test pairs with the interferer concentrated at a
**single azimuth**, separated from the target.  Everything else is held fixed:
same clean binaural targets (reused verbatim from `clean_testset`), same SNR
grid, same SNR definition, same file naming -- so `eval.py` / `eval_bilateral.py`
run against it unchanged and the numbers are directly comparable to the
isotropic condition.

    python make_directional_testset.py --mode noise  --separation 90
    python make_directional_testset.py --mode speech --separation 90
"""

import argparse
import hashlib
import os
import random
import sys
from pathlib import Path

import numpy as np
import soundfile as sf
import tqdm

sys.path.insert(0, "/raid/R12K41024/BCCTN")
from prepare_data import (apply_hrir_fixed, load_and_prepare_speech, load_hrir,
                          mix_speech_and_noise)

HRIR_PATH = "/raid/R12K41024/Dataset/HRIR_database_wav/hrir/anechoic"
NOISE_DIR = "/raid/R12K41024/Dataset/Noises-master/NoiseX-92"
VCTK_DIR = "/raid/R12K41024/Dataset/VCTK-Corpus/VCTK-Corpus/wav48"
CLEAN_TEST = "/raid/R12K41024/BCCTN/Dataset/clean_testset"
SNRS = [-6, -3, 0, 3, 6, 9, 12, 15]
SR = 16000
TARGET_LEN = 2 * SR


def pick_interferer_azimuth(az_target, separation, available):
    """Azimuth `separation` degrees away from the target, kept inside the array.

    Both signs are tried; when both land on a real HRIR the choice is
    deterministic in the target azimuth so the set is reproducible.
    """
    cands = []
    for sign in (+1, -1):
        az = az_target + sign * separation
        if -90 <= az <= 90 and az in available:
            cands.append(az)
    if not cands:  # fall back to the furthest available azimuth
        return max(available, key=lambda a: abs(a - az_target))
    return cands[az_target % len(cands)]


_NOISE_CACHE = {}
_SPEECH_POOL = {}       # speaker -> list of pre-loaded 2 s excerpts


def load_noise(path):
    """Full NOISEX file, resampled once and kept in memory across mixtures."""
    if path not in _NOISE_CACHE:
        import librosa
        y, sr = librosa.load(path, sr=None, mono=True)
        if sr != SR:
            y = librosa.resample(y, orig_sr=sr, target_sr=SR)
        _NOISE_CACHE[path] = y
    return _NOISE_CACHE[path]


def interferer_azimuths(az_target, separation, n, available):
    """Azimuths of the interfering sources around the target."""
    if n == 1:
        return [pick_interferer_azimuth(az_target, separation, available)]
    both = [az_target + separation, az_target - separation]
    if not all(a in available for a in both):
        raise ValueError(f"symmetric pair {both} outside the HRIR grid for "
                         f"target {az_target}; restrict with --max_target_az")
    return both


def build_interferer(mode, rng, hrirs, az_int, noise_files, vctk_by_speaker, target_speaker):
    """One interferer source, HRIR-convolved at az_int.  (2, TARGET_LEN)."""
    hrir = hrirs[az_int]
    if mode == "noise":
        pool = load_noise(rng.choice(noise_files))
        start = rng.randint(0, max(0, len(pool) - TARGET_LEN))
        src = pool[start:start + TARGET_LEN]
        if len(src) < TARGET_LEN:
            src = np.pad(src, (0, TARGET_LEN - len(src)))
    elif _SPEECH_POOL:
        speakers = [s for s in _SPEECH_POOL if s != target_speaker]
        if not speakers:
            return None
        src = rng.choice(_SPEECH_POOL[rng.choice(speakers)])
    else:
        speakers = [s for s in vctk_by_speaker if s != target_speaker]
        for _ in range(10):
            spk = rng.choice(speakers)
            src = load_and_prepare_speech(rng.choice(vctk_by_speaker[spk]), SR, TARGET_LEN)
            if src is not None:
                break
        else:
            return None
    if src is None:
        return None
    # normalize=False keeps the native HRIR gain, matching how the isotropic
    # noise builder treats each direction.
    return apply_hrir_fixed(src, hrir[0], hrir[1], target_sr=SR,
                            target_length=TARGET_LEN, normalize=False)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["noise", "speech"], required=True,
                    help="noise = NOISEX-92 at one azimuth; speech = competing talker")
    ap.add_argument("--separation", type=int, default=90,
                    help="target-to-interferer angular separation in degrees")
    ap.add_argument("--n_interferers", type=int, default=1, choices=[1, 2],
                    help="2 places them symmetrically at AT+-separation, the "
                         "configuration Westhausen et al. (TASLP 2024) use as "
                         "S0N+-60: with interference on both sides no ear is "
                         "clean, so better-ear listening cannot carry the score "
                         "and only real binaural processing helps")
    ap.add_argument("--interferer_pool", type=int, default=0,
                    help="cache this many competing utterances and sample from "
                         "them instead of re-reading VCTK per mixture (0 = off). "
                         "Westhausen et al. use a single fixed IFFM masker, so a "
                         "bounded pool is faithful, not a shortcut -- and it is "
                         "what makes a 31k-mix train set tractable")
    ap.add_argument("--tag_suffix", default="",
                    help="appended to the output directory name; required when "
                         "two runs share mode/n_interferers/separation and would "
                         "otherwise overwrite each other (e.g. '_s0')")
    ap.add_argument("--max_target_az", type=int, default=None,
                    help="keep only targets within +-this azimuth, so the "
                         "symmetric pair always fits inside the array without "
                         "clamping (use 30 with --n_interferers 2 --separation 60)")
    ap.add_argument("--limit", type=int, default=800,
                    help="clean files to process (test: eval uses the first 750; 0 = all)")
    ap.add_argument("--split", choices=["test", "train", "val"], default="test",
                    help="test writes snr_XdB/ subdirs on the 8-point grid; "
                         "train/val mirror the existing flat sets one mix per file, "
                         "reusing each file's own SNR so the distribution is unchanged")
    ap.add_argument("--data_root", default="/raid/R12K41024/BCCTN/Dataset",
                    help="where the clean targets and the isotropic sets live")
    ap.add_argument("--out_root", default="/raid/R12K41024/BCCTN/Dataset",
                    help="where the new set is written")
    args = ap.parse_args()

    cache = Path(__file__).resolve().parent.parent / "hrir_cache_16k.npz"
    if cache.exists():
        hrirs = {int(k): v for k, v in np.load(cache).items()}
        print(f"HRIRs from cache: {len(hrirs)} azimuths")
    else:
        hrirs = load_hrir(HRIR_PATH, format="wav", target_sr=SR)
        np.savez(cache, **{str(k): v for k, v in hrirs.items()})
    available = sorted(a for a in hrirs if -90 <= a <= 90)
    noise_files = sorted(str(p) for p in Path(NOISE_DIR).glob("*.wav"))
    vctk_by_speaker = {}
    if args.mode == "speech":
        for spk_dir in sorted(Path(VCTK_DIR).iterdir()):
            if spk_dir.is_dir():
                wavs = sorted(str(p) for p in spk_dir.glob("*.wav"))
                if wavs:
                    vctk_by_speaker[spk_dir.name] = wavs

    n_tag = "" if args.n_interferers == 1 else f"{args.n_interferers}x"
    if args.mode == "speech" and args.interferer_pool:
        rng0 = random.Random(1234)
        spks = sorted(vctk_by_speaker)
        per = max(1, args.interferer_pool // len(spks))
        for spk in tqdm.tqdm(spks, desc="caching interferer pool"):
            got = []
            for f in rng0.sample(vctk_by_speaker[spk], min(per * 3, len(vctk_by_speaker[spk]))):
                y = load_and_prepare_speech(f, SR, TARGET_LEN)
                if y is not None:
                    got.append(y)
                if len(got) >= per:
                    break
            if got:
                _SPEECH_POOL[spk] = got
        print(f"interferer pool: {sum(len(v) for v in _SPEECH_POOL.values())} "
              f"utterances over {len(_SPEECH_POOL)} speakers", flush=True)

    tag = f"dir{args.mode}{n_tag}{args.separation}{args.tag_suffix}"
    split_dirs = {"test": ("clean_testset", "noisy_testset"),
                  "train": ("clean_trainset", "noisy_trainset"),
                  "val": ("clean_valset", "noisy_valset")}
    clean_name, noisy_name = split_dirs[args.split]
    out_base = Path(args.out_root) / f"{noisy_name}_{tag}"
    out_base.mkdir(parents=True, exist_ok=True)
    if args.split == "test":
        for snr in SNRS:
            (out_base / f"snr_{snr}dB").mkdir(parents=True, exist_ok=True)

    clean_files = sorted((Path(args.data_root) / clean_name).glob("*.wav"))
    if args.max_target_az is not None:
        clean_files = [f for f in clean_files
                       if abs(int(f.stem.rsplit("_az", 1)[1])) <= args.max_target_az]
    if args.limit:
        clean_files = clean_files[:args.limit]

    # train/val: one mix per file at that file's own SNR, read back off the
    # existing isotropic filenames so the SNR distribution is untouched.
    snr_by_stem = {}
    if args.split != "test":
        import re as _re
        for f in (Path(args.data_root) / noisy_name).glob("*.wav"):
            m = _re.search(r"^(.*)_snr([-+][\d.]+)$", f.stem)
            if m:
                snr_by_stem.setdefault(m.group(1), []).append(float(m.group(2)))

    print(f"mode={args.mode} separation={args.separation} deg split={args.split} -> {out_base}")
    n_mix = len(clean_files) * (len(SNRS) if args.split == "test" else 1)
    print(f"{len(clean_files)} clean targets -> {n_mix} mixtures")

    manifest = [("file", "az_target", "az_interferer", "snr", "interferer_source")]
    for cf in tqdm.tqdm(clean_files):
        stem = cf.stem                                   # <base>_az<AT>
        az_target = int(stem.rsplit("_az", 1)[1])
        base = stem.rsplit("_az", 1)[0]
        target_speaker = base.split("_")[0]
        az_ints = interferer_azimuths(az_target, args.separation,
                                      args.n_interferers, available)

        clean, _ = sf.read(cf, dtype="float32", always_2d=True)
        clean = clean.T                                  # (2, N)

        snr_list = SNRS if args.split == "test" else snr_by_stem.get(stem, [])
        for snr in snr_list:
            # Deterministic per (file, SNR) so the set is reproducible.
            seed = int(hashlib.md5(f"{stem}|{snr}|{tag}".encode()).hexdigest()[:8], 16)
            rng = random.Random(seed)
            # Sources are summed before the SNR scaling, so the nominal SNR is
            # always defined against the total interferer power.
            parts = [build_interferer(args.mode, rng, hrirs, a, noise_files,
                                      vctk_by_speaker, target_speaker)
                     for a in az_ints]
            if any(x is None for x in parts):
                continue
            interf = sum(parts)
            mix = mix_speech_and_noise(clean, interf, snr)
            name = f"{base}_az{az_target}_snr{snr:+.1f}.wav"
            dest = out_base / (f"snr_{snr}dB" if args.split == "test" else "")
            sf.write(dest / name, mix.T, SR)
            manifest.append((name, az_target, "|".join(str(a) for a in az_ints), snr, "-"))

    with open(out_base / "manifest.csv", "w") as f:
        for row in manifest:
            f.write(",".join(str(x) for x in row) + "\n")
    print(f"done -> {out_base}")


if __name__ == "__main__":
    main()
