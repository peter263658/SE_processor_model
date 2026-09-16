"""Mono-ear dataset built from the binaural VCTK train/val sets.

Each stereo file contributes two independent mono samples: (noisy left ->
clean left) and (noisy right -> clean right).  This is the matched-data,
favorable-to-baseline setup for the mono-SE paradigm reference row of the
chip-paper comparison table.

File naming: noisy `pXXX_pYYY_NNN_azAA_snrSS.S.wav` pairs with clean
`pXXX_pYYY_NNN_azAA.wav`.  All files are 2 s stereo at 16 kHz.
"""

import re
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
from torch.utils.data import Dataset

_SNR_SUFFIX = re.compile(r"_snr[+-]?[\d.]+\.wav$")
_SNR_VALUE = re.compile(r"_snr([+-]?[\d.]+)\.wav$")


class MonoEarDataset(Dataset):
    def __init__(self, noisy_dir, clean_dir, seg_len=32000, snr_band=None):
        """snr_band: keep only mixes with |SNR| <= snr_band dB (None = all).

        Used to reproduce the NTU JSSC'20 training regime, which mixes every
        utterance at a single SNR (0 dB) instead of the full -7..16 dB spread.
        """
        self.noisy_dir = Path(noisy_dir)
        self.clean_dir = Path(clean_dir)
        self.seg_len = seg_len
        self.snr_band = snr_band
        self.pairs = []
        for noisy in sorted(self.noisy_dir.glob("*.wav")):
            if snr_band is not None:
                m = _SNR_VALUE.search(noisy.name)
                if m is None or abs(float(m.group(1))) > snr_band:
                    continue
            clean = self.clean_dir / _SNR_SUFFIX.sub(".wav", noisy.name)
            if clean.exists():
                self.pairs.append((noisy, clean))
        if not self.pairs:
            raise RuntimeError(f"no noisy/clean pairs found under {noisy_dir}")

    def __len__(self):
        return 2 * len(self.pairs)

    def __getitem__(self, idx):
        noisy_path, clean_path = self.pairs[idx // 2]
        ear = idx % 2
        noisy, _ = sf.read(noisy_path, dtype="float32", always_2d=True)
        clean, _ = sf.read(clean_path, dtype="float32", always_2d=True)
        noisy, clean = noisy[:, ear], clean[:, ear]
        n = self.seg_len
        if len(noisy) < n:
            noisy = np.pad(noisy, (0, n - len(noisy)))
            clean = np.pad(clean, (0, n - len(clean)))
        return torch.from_numpy(noisy[:n].copy()), torch.from_numpy(clean[:n].copy())
