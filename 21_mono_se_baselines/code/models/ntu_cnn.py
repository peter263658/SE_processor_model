"""NTU JSSC'20 speech-enhancement CNN (FP32 reimplementation).

Lee et al., "A 2.17-mW Acoustic DSP Processor With CNN-FFT Accelerators for
Intelligent Hearing Assistive Devices," IEEE JSSC 2020.  Network reproduced
from Table I; the chip's model is a simplified version of the CNN in
Lai et al., "Plastic multi-resolution auditory model based neural network for
speech enhancement" (APSIPA 2017 -- ref [13] of the JSSC paper).

Pipeline (Sec. II / VI): 16 kHz waveform -> 256-pt DFT (Hamming window,
50% overlap, 129 bins) -> CNN predicts a per-frame 129-D mask from a 7-frame
spectrogram context -> mask applied to the noisy spectrum -> IDFT.

Table I:
  Conv1     1x129x7  -> 32x129x3   filter 1x15x5   (32 filters)
  Max-pool  32x129x3 -> 32x43x3    1x3x1
  Conv2     32x43x3  -> 64x39x1    filter 32x5x3   (64 filters)
  Max-pool  64x39x1  -> 64x13x1    1x3x1
  FC1       832 -> 480
  FC2       480 -> 129

~496k parameters.  The paper does not specify the input scaling or the mask
nonlinearity; we use log-magnitude input and a sigmoid mask (the standard
choice for this model family), which favors the baseline.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class NTUCnnNet(nn.Module):
    """Waveform-in / waveform-out mono speech enhancer (NTU JSSC'20 CNN)."""

    N_FFT = 256
    HOP = 128
    CONTEXT = 7  # frames of spectrogram context per output mask frame

    def __init__(self):
        super().__init__()
        self.register_buffer("window", torch.hamming_window(self.N_FFT))
        # freq padded (15//2), time valid over the 7-frame context
        self.conv1 = nn.Conv2d(1, 32, (15, 5), padding=(7, 0))
        self.pool = nn.MaxPool2d((3, 1))
        self.conv2 = nn.Conv2d(32, 64, (5, 3))
        self.fc1 = nn.Linear(64 * 13, 480)
        self.fc2 = nn.Linear(480, 129)

    def mask_from_mag(self, mag):  # (B, 129, T) -> (B, 129, T) in (0,1)
        x = torch.log(mag + 1e-5).unsqueeze(1)          # (B, 1, 129, T)
        pad = (self.CONTEXT - 1) // 2
        x = F.pad(x, (pad, pad, 0, 0), mode="replicate")  # center-aligned context
        x = F.relu(self.conv1(x))                       # (B, 32, 129, T+2)
        x = self.pool(x)                                # (B, 32, 43, T+2)
        x = F.relu(self.conv2(x))                       # (B, 64, 39, T)
        x = self.pool(x)                                # (B, 64, 13, T)
        x = x.flatten(1, 2).permute(0, 2, 1)            # (B, T, 832)
        x = F.relu(self.fc1(x))
        mask = torch.sigmoid(self.fc2(x))               # (B, T, 129)
        return mask.permute(0, 2, 1)

    def forward(self, wav):  # (B, N) -> (B, N)
        n = wav.shape[-1]
        spec = torch.stft(wav, self.N_FFT, self.HOP, window=self.window,
                          center=True, return_complex=True)  # (B, 129, T)
        mag, phase = spec.abs(), torch.angle(spec)
        mask = self.mask_from_mag(mag)
        enh_spec = torch.polar(mask * mag, phase)
        return torch.istft(enh_spec, self.N_FFT, self.HOP, window=self.window,
                           center=True, length=n)

    def forward_pair_shared(self, wav):  # (B, 2, N) -> (B, 2, N)
        """Bilateral forward with a single mask shared by both ears.

        The shared mask is the mean of the two ears' own masks and is applied
        identically to each ear, so the inter-ear level ratio of every T-F bin
        is left exactly as it was in the noisy input -- the cheapest way a mono
        mask-based network can stop destroying ILD.  Averaging the two masks
        (rather than masking a downmix) keeps both ears' information in the
        estimate, which favors the baseline.
        """
        B, C, N = wav.shape
        spec = torch.stft(wav.reshape(B * C, N), self.N_FFT, self.HOP,
                          window=self.window, center=True, return_complex=True)
        mag, phase = spec.abs(), torch.angle(spec)
        F_, T = mag.shape[-2:]
        mask = self.mask_from_mag(mag).reshape(B, C, F_, T)
        mask = mask.mean(dim=1, keepdim=True).expand(-1, C, -1, -1).reshape(B * C, F_, T)
        enh_spec = torch.polar(mask * mag, phase)
        return torch.istft(enh_spec, self.N_FFT, self.HOP, window=self.window,
                           center=True, length=N).reshape(B, C, N)
