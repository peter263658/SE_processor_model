"""
Upsample the 8 kHz sanity-check dataset to 16 kHz so the 16 kHz-trained
checkpoint can process it. The audio remains bandlimited to 4 kHz — that
is exactly the experimental question we want to isolate.

Reads:  Dataset_8khz/{clean_testset,noisy_testset/snr_*dB}/*.wav  (8 kHz)
Writes: Dataset_8khz_upsampled_16k/...                            (16 kHz)
"""

import os
import sys
from pathlib import Path

import librosa
import numpy as np
import soundfile as sf
from tqdm import tqdm


SRC = Path("/raid/R12K41024/01_utilization_analysis_8khz/Dataset_8khz")
DST = Path("/raid/R12K41024/01_utilization_analysis_8khz/Dataset_8khz_upsampled_16k")
ORIG_SR = 8000
TARGET_SR = 16000


def resample_wav(src_path: Path, dst_path: Path) -> None:
    data, sr = sf.read(str(src_path))
    assert sr == ORIG_SR, f"expected {ORIG_SR} Hz, got {sr} for {src_path}"
    if data.ndim == 1:
        data = data[:, None]
    out = np.stack(
        [librosa.resample(data[:, ch], orig_sr=sr, target_sr=TARGET_SR)
         for ch in range(data.shape[1])],
        axis=1,
    )
    dst_path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(dst_path), out, TARGET_SR)


def main() -> int:
    if not SRC.exists():
        print(f"missing source: {SRC}", file=sys.stderr)
        return 1

    wavs = list(SRC.rglob("*.wav"))
    print(f"resampling {len(wavs)} files {ORIG_SR}->{TARGET_SR}...")
    for src in tqdm(wavs):
        rel = src.relative_to(SRC)
        resample_wav(src, DST / rel)
    print(f"done -> {DST}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
