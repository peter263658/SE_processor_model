"""SNU JSSC'24 speech-enhancement network (FP32 reimplementation).

Park et al., "A Real-Time Speech Enhancement Processor for Hearing Aids in
28-nm CMOS," IEEE JSSC 2024.  Network reproduced from the paper text
(Sec. II) and the per-layer table in Fig. 9.  The model is the band-separated
variant of TRU-Net (Choi et al., "Real-time denoising and dereverberation
with tiny recurrent U-Net," ICASSP 2021 -- ref [6] of the JSSC paper).

Pipeline (Sec. II-A):
  16 kHz waveform -> STFT (512-pt, 32 ms window, 16 ms hop)
  -> 128-D Mel magnitude -> NN -> sigmoid mask on Mel magnitude
  -> pseudo-inverse Mel -> noisy phase -> iSTFT.

Network (Fig. 9), all activations ReLU6, ~56k parameters:
  - Encoder split into low/high frequency halves (64 Mel bins each),
    3x (depthwise 5x1 stride-2) with pointwise convs in between.
  - Bi-directional FGRU (input 64, hidden 32) over 16 freq positions + FC 64->32.
  - BS-TGRU: 4 unidirectional time-axis GRUs (hidden 8/16/32/32).  The h=8
    GRU covers all 4 sub-bands, h=16 covers the 3 lowest, h=32 the 2 lowest,
    and the second h=32 only the lowest band.  Per-band FC combine:
    88->32, 56->32, 24->16, 8->16.
  - Decoder mirrors the encoder with transposed depthwise convs
    (low band 32 ch, high band 16 ch), final pointwise -> 1 ch mask.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


def _mel_filterbank(n_freqs=257, n_mels=128, sample_rate=16000, f_min=0.0, f_max=8000.0):
    """HTK-style triangular Mel filterbank, shape (n_mels, n_freqs).

    Any filter row that quantizes to all zeros on the linear-frequency grid
    (possible at low frequencies with 128 mels over a 31.25 Hz grid) falls
    back to a one-hot at the nearest linear bin so the 128-D Mel space stays
    full rank for the pseudo-inverse reconstruction.
    """
    def hz_to_mel(f):
        return 2595.0 * torch.log10(1.0 + f / 700.0)

    def mel_to_hz(m):
        return 700.0 * (10.0 ** (m / 2595.0) - 1.0)

    all_freqs = torch.linspace(0, sample_rate // 2, n_freqs)
    m_pts = torch.linspace(hz_to_mel(torch.tensor(f_min)), hz_to_mel(torch.tensor(f_max)), n_mels + 2)
    f_pts = mel_to_hz(m_pts)

    fb = torch.zeros(n_mels, n_freqs)
    for i in range(n_mels):
        lo, ctr, hi = f_pts[i], f_pts[i + 1], f_pts[i + 2]
        up = (all_freqs - lo) / max(ctr - lo, torch.tensor(1e-6))
        down = (hi - all_freqs) / max(hi - ctr, torch.tensor(1e-6))
        fb[i] = torch.clamp(torch.minimum(up, down), min=0.0)
        if fb[i].sum() == 0:
            fb[i, torch.argmin((all_freqs - ctr).abs())] = 1.0
    return fb


class _BandEncoder(nn.Module):
    """One frequency-band encoder column of Fig. 9 (freq-axis 1-D convs)."""

    def __init__(self, c1, c2):
        super().__init__()
        # Fig. 9 rows: DW(1->c1,5x1) PW(c1->c2) DW(c2,5x1) PW(c2->64) DW(64,5x1)
        self.dw1 = nn.Conv2d(1, c1, (5, 1), stride=(2, 1), padding=(2, 0))
        self.pw1 = nn.Conv2d(c1, c2, 1)
        self.dw2 = nn.Conv2d(c2, c2, (5, 1), stride=(2, 1), padding=(2, 0), groups=c2)
        self.pw2 = nn.Conv2d(c2, 64, 1)
        self.dw3 = nn.Conv2d(64, 64, (5, 1), stride=(2, 1), padding=(2, 0), groups=64)

    def forward(self, x):  # (B, 1, 64, T) -> (B, 64, 8, T)
        for layer in (self.dw1, self.pw1, self.dw2, self.pw2, self.dw3):
            x = F.relu6(layer(x))
        return x


class _BandDecoder(nn.Module):
    """One frequency-band decoder column of Fig. 9 (transposed DW convs)."""

    def __init__(self, c):
        super().__init__()
        def dwt():
            return nn.ConvTranspose2d(c, c, (5, 1), stride=(2, 1), padding=(2, 0),
                                      output_padding=(1, 0), groups=c)
        self.dw1, self.pw1 = dwt(), nn.Conv2d(c, c, 1)
        self.dw2, self.pw2 = dwt(), nn.Conv2d(c, c, 1)
        self.dw3, self.pw3 = dwt(), nn.Conv2d(c, 1, 1)

    def forward(self, x):  # (B, c, 8, T) -> (B, 1, 64, T), pre-sigmoid
        for layer in (self.dw1, self.pw1, self.dw2, self.pw2, self.dw3):
            x = F.relu6(layer(x))
        return self.pw3(x)


class _BSTGRU(nn.Module):
    """Band-separated time-axis GRU (Fig. 5-7).

    Frequency position order: 0 = lowest.  16 positions = 4 sub-bands x 4.
    GRU coverage (positions counted from the lowest band):
      h=8 -> all 16, h=16 -> lowest 12, h=32 -> lowest 8, h=32 -> lowest 4.
    """

    COVER = (16, 12, 8, 4)
    HIDDEN = (8, 16, 32, 32)

    def __init__(self, in_ch=32):
        super().__init__()
        self.grus = nn.ModuleList(nn.GRU(in_ch, h, batch_first=True) for h in self.HIDDEN)
        # FC combine per band, lowest first: 88->32, 56->32, 24->16, 8->16
        self.fcs = nn.ModuleList([nn.Linear(88, 32), nn.Linear(56, 32),
                                  nn.Linear(24, 16), nn.Linear(8, 16)])

    def forward(self, x):  # (B, C, 16, T)
        B, C, Fq, T = x.shape
        seq = x.permute(0, 2, 3, 1)  # (B, 16, T, C)
        outs = []
        for gru, cover in zip(self.grus, self.COVER):
            inp = seq[:, :cover].reshape(B * cover, T, C)
            out, _ = gru(inp)
            outs.append(out.reshape(B, cover, T, -1))
        bands = []
        for b, fc in enumerate(self.fcs):
            pos = slice(4 * b, 4 * b + 4)
            feats = [o[:, pos] for o, cover in zip(outs, self.COVER) if cover >= 4 * (b + 1)]
            bands.append(F.relu6(fc(torch.cat(feats, dim=-1))))  # (B, 4, T, 32|16)
        low = torch.cat(bands[0:2], dim=1).permute(0, 3, 1, 2)   # (B, 32, 8, T)
        high = torch.cat(bands[2:4], dim=1).permute(0, 3, 1, 2)  # (B, 16, 8, T)
        return low, high


class SNUBstgruNet(nn.Module):
    """Waveform-in / waveform-out mono speech enhancer (SNU JSSC'24 network)."""

    N_FFT = 512
    HOP = 256
    N_MELS = 128

    def __init__(self):
        super().__init__()
        fb = _mel_filterbank(self.N_FFT // 2 + 1, self.N_MELS)
        self.register_buffer("mel_fb", fb)                       # (128, 257)
        self.register_buffer("mel_inv", torch.linalg.pinv(fb))   # (257, 128)
        self.register_buffer("window", torch.hann_window(self.N_FFT))

        self.enc_low = _BandEncoder(c1=32, c2=64)
        self.enc_high = _BandEncoder(c1=16, c2=32)
        self.fgru = nn.GRU(64, 32, batch_first=True, bidirectional=True)
        self.fgru_fc = nn.Linear(64, 32)
        self.bstgru = _BSTGRU(in_ch=32)
        self.dec_low = _BandDecoder(c=32)
        self.dec_high = _BandDecoder(c=16)

    def mask_from_mel(self, mel_mag):  # (B, 128, T) -> (B, 128, T) in (0,1)
        x = mel_mag.unsqueeze(1)                       # (B, 1, 128, T)
        e = torch.cat([self.enc_low(x[:, :, :64]),
                       self.enc_high(x[:, :, 64:])], dim=2)  # (B, 64, 16, T)

        B, C, Fq, T = e.shape
        seq = e.permute(0, 3, 2, 1).reshape(B * T, Fq, C)      # (B*T, 16, 64)
        f, _ = self.fgru(seq)
        f = F.relu6(self.fgru_fc(f))                            # (B*T, 16, 32)
        f = f.reshape(B, T, Fq, 32).permute(0, 3, 2, 1)         # (B, 32, 16, T)

        low, high = self.bstgru(f)
        mask = torch.cat([self.dec_low(low), self.dec_high(high)], dim=2)  # (B,1,128,T)
        return torch.sigmoid(mask).squeeze(1)

    def forward(self, wav):  # (B, N) -> (B, N)
        n = wav.shape[-1]
        spec = torch.stft(wav, self.N_FFT, self.HOP, window=self.window,
                          center=True, return_complex=True)     # (B, 257, T)
        mag, phase = spec.abs(), torch.angle(spec)
        mel_mag = torch.matmul(self.mel_fb, mag)                # (B, 128, T)

        mask = self.mask_from_mel(mel_mag)
        enh_mel = mask * mel_mag
        enh_mag = torch.clamp(torch.matmul(self.mel_inv, enh_mel), min=0.0)
        enh_spec = torch.polar(enh_mag, phase)
        return torch.istft(enh_spec, self.N_FFT, self.HOP, window=self.window,
                           center=True, length=n)

    def forward_pair_shared(self, wav):  # (B, 2, N) -> (B, 2, N)
        """Bilateral forward with a single Mel mask shared by both ears.

        Counterpart of `NTUCnnNet.forward_pair_shared`; the mask is averaged in
        this network's own mask domain (Mel), so ILD is preserved only up to
        the pseudo-inverse Mel reconstruction rather than exactly.
        """
        B, C, N = wav.shape
        spec = torch.stft(wav.reshape(B * C, N), self.N_FFT, self.HOP,
                          window=self.window, center=True, return_complex=True)
        mag, phase = spec.abs(), torch.angle(spec)
        mel_mag = torch.matmul(self.mel_fb, mag)               # (B*C, 128, T)
        M, T = mel_mag.shape[-2:]
        mask = self.mask_from_mel(mel_mag).reshape(B, C, M, T)
        mask = mask.mean(dim=1, keepdim=True).expand(-1, C, -1, -1).reshape(B * C, M, T)
        enh_mag = torch.clamp(torch.matmul(self.mel_inv, mask * mel_mag), min=0.0)
        enh_spec = torch.polar(enh_mag, phase)
        return torch.istft(enh_spec, self.N_FFT, self.HOP, window=self.window,
                           center=True, length=N).reshape(B, C, N)
