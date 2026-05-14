import torch
import torch.nn as nn
import torch.nn.functional as F


def _as_bool(value):
    if isinstance(value, str):
        return value.lower() in {"1", "true", "yes", "on"}
    return bool(value)


def _fixed_point_quantize(x: torch.Tensor, total_bits: int, frac_bits: int):
    """Fake fixed-point quantize/dequantize while keeping torch ops in float."""
    if total_bits <= 0:
        raise ValueError(f"total_bits must be positive, got {total_bits}")

    scale = 2.0 ** frac_bits
    qmin = -(2 ** (total_bits - 1))
    qmax = 2 ** (total_bits - 1) - 1

    def quantize_real(t):
        q = torch.round(t * scale).clamp(qmin, qmax)
        return q / scale

    if torch.is_complex(x):
        real = quantize_real(x.real)
        imag = quantize_real(x.imag)
        return torch.complex(real, imag)

    return quantize_real(x)


def _fp8_quantize_real(x: torch.Tensor, fp8_dtype):
    max_val = 448.0 if fp8_dtype == torch.float8_e4m3fn else 57344.0
    amax = x.detach().abs().max().clamp(min=1e-8)
    scale = max_val / amax
    q = torch.clamp(x * scale, -max_val, max_val).to(fp8_dtype)
    return q.to(torch.float32) / scale


def _float_format_quantize(x: torch.Tensor, fmt: str):
    fmt = (fmt or "fp32").lower()
    if fmt in {"fp32", "float32"}:
        return x

    def quantize_real(t):
        if fmt in {"fp16", "float16", "half"}:
            return t.to(torch.float16).to(torch.float32)
        if fmt in {"bf16", "bfloat16"}:
            return t.to(torch.bfloat16).to(torch.float32)
        if fmt in {"fp8", "fp8_e4m3", "e4m3"}:
            return _fp8_quantize_real(t, torch.float8_e4m3fn)
        if fmt in {"fp8_e5m2", "e5m2"}:
            return _fp8_quantize_real(t, torch.float8_e5m2)
        raise ValueError(f"Unsupported STFT float precision format: {fmt}")

    if torch.is_complex(x):
        return torch.complex(quantize_real(x.real), quantize_real(x.imag))
    return quantize_real(x)


class Stft(nn.Module):
    def __init__(self, n_dft=512, hop_size=256, win_length=None,
                 onesided=True, is_complex=True, fixed_point=None,
                 float_precision=None):

        super().__init__()

        self.n_dft = n_dft
        self.hop_size = hop_size
        self.win_length = n_dft if win_length is None else win_length
        self.onesided = onesided
        self.is_complex = is_complex
        self.fixed_point = fixed_point or {}
        self.fixed_point_enabled = _as_bool(self.fixed_point.get("enabled", False))
        self.float_precision = float_precision or {}
        self.float_precision_enabled = _as_bool(self.float_precision.get("enabled", False))
        if self.fixed_point_enabled and self.float_precision_enabled:
            raise ValueError("Enable either stft_fixed_point or stft_float_precision, not both.")

    def _stage_enabled(self, stage_name: str, default=True):
        if not self.fixed_point_enabled:
            return False
        return _as_bool(self.fixed_point.get(f"quantize_{stage_name}", default))

    def _stage_bits(self, stage_name: str):
        total_bits = int(self.fixed_point.get(
            f"{stage_name}_total_bits",
            self.fixed_point.get("total_bits", 16),
        ))
        frac_bits = int(self.fixed_point.get(
            f"{stage_name}_frac_bits",
            self.fixed_point.get("frac_bits", 10),
        ))
        return total_bits, frac_bits

    def _quantize_stage(self, x: torch.Tensor, stage_name: str, default=True):
        if not self._stage_enabled(stage_name, default=default):
            if self._float_stage_enabled(stage_name, default=default):
                return _float_format_quantize(x, self._float_stage_format(stage_name))
            return x

        total_bits, frac_bits = self._stage_bits(stage_name)
        return _fixed_point_quantize(x, total_bits, frac_bits)

    def _float_stage_enabled(self, stage_name: str, default=True):
        if not self.float_precision_enabled:
            return False
        return _as_bool(self.float_precision.get(f"quantize_{stage_name}", default))

    def _float_stage_format(self, stage_name: str):
        return self.float_precision.get(
            f"{stage_name}_format",
            self.float_precision.get("format", "fp32"),
        )

    def forward(self, x: torch.Tensor):
        "Expected input has shape (batch_size, n_channels, time_steps)"

        x = self._quantize_stage(x, "input")

        # window = torch.hann_window(self.win_length, device=x.device)
        window = torch.hamming_window(self.win_length, periodic=False, device=x.device)
        window = self._quantize_stage(window, "window")

        y = torch.stft(x, self.n_dft, hop_length=self.hop_size,
                       win_length=self.win_length, onesided=self.onesided,
                       return_complex=True, window=window, center=False, normalized=True)
        y = self._quantize_stage(y, "spectrum")
        
        # y = y[:, 1:] # Remove DC component (f=0hz)
        y = y.narrow(dim=-2, start=1, length=y.size(-2) - 1)
        # y.shape == (batch_size*channels, time, freqs)

        if not self.is_complex:
            y = torch.view_as_real(y)
            y = y.movedim(-1, 1) # move complex dim to front

        return y


class IStft(Stft):

    def forward(self, x: torch.Tensor):
        "Expected input has shape (batch_size, n_channels=freq_bins, time_steps)"
        x = self._quantize_stage(x, "istft_input")

        # window = torch.hann_window(self.win_length, device=x.device)
        window = torch.hamming_window(self.win_length, periodic=False, device=x.device)
        window = self._quantize_stage(window, "window")

        y = torch.istft(x, self.n_dft, hop_length=self.hop_size,
                        win_length=self.win_length, onesided=self.onesided,
                        window=window,center=False, normalized=True)
        y = self._quantize_stage(y, "output")

        return y
