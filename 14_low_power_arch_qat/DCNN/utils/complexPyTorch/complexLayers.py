#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 19 10:30:02 2019

@author: Sebastien M. Popoff


Based on https://openreview.net/forum?id=H1T2hmZAb
"""

import math

import torch
import torch.nn.functional as F
from torch.nn import (
    Module, Parameter, init,
    Conv2d, ConvTranspose2d, Linear, LSTM, GRU,
    BatchNorm1d, BatchNorm2d,
    PReLU, MultiheadAttention
)

from .complexFunctions import (
    complex_amp_tanh,
    complex_relu,
    complex_tanh,
    complex_sigmoid,
    complex_max_pool2d,
    complex_avg_pool2d,
    complex_dropout,
    complex_dropout2d,
    complex_opposite,
)


def apply_complex(fr, fi, input, dtype=torch.complex64):
    return (fr(input.real)-fi(input.imag)).type(dtype) \
        + 1j*(fr(input.imag)+fi(input.real)).type(dtype)


class ComplexDropout(Module):
    def __init__(self, p=0.5):
        super().__init__()
        self.p = p

    def forward(self, input):
        if self.training:
            return complex_dropout(input, self.p)
        else:
            return input


class ComplexDropout2d(Module):
    def __init__(self, p=0.5):
        super().__init__()
        self.p = p

    def forward(self, input):
        if self.training:
            return complex_dropout2d(input, self.p)
        else:
            return input


class ComplexMaxPool2d(Module):

    def __init__(self, kernel_size, stride=None, padding=0,
                 dilation=1, return_indices=False, ceil_mode=False):
        super().__init__()
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.ceil_mode = ceil_mode
        self.return_indices = return_indices

    def forward(self, input):
        return complex_max_pool2d(input, kernel_size=self.kernel_size,
                                  stride=self.stride, padding=self.padding,
                                  dilation=self.dilation, ceil_mode=self.ceil_mode,
                                  return_indices=self.return_indices)


class ComplexAvgPool2d(Module):

    def __init__(self, kernel_size, stride=None, padding=0,
                 ceil_mode=False):
        super().__init__()
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.ceil_mode = ceil_mode

    def forward(self, input):
        return complex_avg_pool2d(input, kernel_size=self.kernel_size,
                                  stride=self.stride, padding=self.padding,
                                  ceil_mode=self.ceil_mode)


class ComplexReLU(Module):

    def forward(self, input):
        return complex_relu(input)


class ComplexSigmoid(Module):

    def forward(self, input):
        return complex_sigmoid(input)


class ComplexTanh(Module):
    def forward(self, input):
        return complex_tanh(input)


class ComplexAmpTanh(Module):
    def forward(self, input):
        return complex_amp_tanh(input)


def _round_ste_or_plain(x):
    fn = globals().get("_ste_round")
    return fn(x) if fn is not None else torch.round(x)


def _pow2_quantize_scale(scale, min_exp=-8, max_exp=8):
    signed = torch.sign(scale)
    magnitude = scale.abs().clamp_min(2.0 ** int(min_exp))
    exponent = _round_ste_or_plain(torch.log2(magnitude)).clamp(
        int(min_exp), int(max_exp)
    )
    quantized = signed * torch.pow(torch.full_like(magnitude, 2.0), exponent)
    return torch.where(scale == 0, torch.zeros_like(quantized), quantized)


def _quantize_bn_scale(weight, mode="none", pow2_min_exp=-8, pow2_max_exp=8):
    if weight is None:
        return None
    mode = str(mode or "none").lower()
    if mode in ("none", "off", "false"):
        return weight
    if mode == "pow2":
        return _pow2_quantize_scale(weight, pow2_min_exp, pow2_max_exp)
    if mode in ("fp8_e4m3", "e4m3", "fp8_e5m2", "e5m2"):
        return _fake_fp8_quantize_tensor(weight, _fp8_dtype_from_mode(mode))
    raise ValueError(f"Unsupported BN affine scale_mode: {mode}")


class ComplexPReLU(Module):
    def __init__(self, pow2_slope=False, min_exp=-8, max_exp=0):
        super().__init__()
        self.r_prelu = PReLU(dtype=torch.float32)
        self.i_prelu = PReLU(dtype=torch.float32)
        self.pow2_slope = bool(pow2_slope)
        self.pow2_min_exp = int(min_exp)
        self.pow2_max_exp = int(max_exp)

    def _apply_prelu(self, x, module):
        if not self.pow2_slope:
            return module(x)
        slope = _pow2_quantize_scale(
            module.weight, self.pow2_min_exp, self.pow2_max_exp
        )
        return F.prelu(x, slope)

    def forward(self, input):
        result = self._apply_prelu(input.real, self.r_prelu) \
            + 1j*self._apply_prelu(input.imag, self.i_prelu)
        return result


class ComplexConvTranspose2d(Module):

    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0,
                 output_padding=0, groups=1, bias=True, dilation=1, padding_mode='zeros'):

        super().__init__()

        self.conv_tran_r = ConvTranspose2d(in_channels, out_channels, kernel_size, stride, padding,
                                           output_padding, groups, bias, dilation, padding_mode)
        self.conv_tran_i = ConvTranspose2d(in_channels, out_channels, kernel_size, stride, padding,
                                           output_padding, groups, bias, dilation, padding_mode)

    def forward(self, input):
        return apply_complex(self.conv_tran_r, self.conv_tran_i, input)


class ComplexConv2d(Module):

    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=0,
                 dilation=1, groups=1, bias=True):
        super().__init__()
        self.conv_r = Conv2d(in_channels, out_channels,
                             kernel_size, stride, padding, dilation, groups, bias)
        self.conv_i = Conv2d(in_channels, out_channels,
                             kernel_size, stride, padding, dilation, groups, bias)

    def forward(self, input):
        return apply_complex(self.conv_r, self.conv_i, input)


class ComplexLinear(Module):

    def __init__(self, in_features, out_features):
        super().__init__()
        self.fc_r = Linear(in_features, out_features)
        self.fc_i = Linear(in_features, out_features)

    def forward(self, input):
        return apply_complex(self.fc_r, self.fc_i, input)


def _valid_group_count(requested_groups, in_features, out_features):
    requested_groups = max(1, int(requested_groups))
    max_groups = math.gcd(int(in_features), int(out_features))
    for groups in range(min(requested_groups, max_groups), 0, -1):
        if max_groups % groups == 0:
            return groups
    return 1


def _linear_group_mask(weight, groups):
    groups = _valid_group_count(groups, weight.shape[1], weight.shape[0])
    if groups <= 1:
        return None
    out_features, in_features = weight.shape
    out_per_group = out_features // groups
    in_per_group = in_features // groups
    mask = torch.zeros(
        (out_features, in_features), dtype=torch.float32, device=weight.device
    )
    for group_idx in range(groups):
        out_lo = group_idx * out_per_group
        in_lo = group_idx * in_per_group
        mask[out_lo:out_lo + out_per_group, in_lo:in_lo + in_per_group] = 1.0
    return mask.to(weight.dtype)


def _mha_in_proj_group_mask(weight, embed_dim, groups):
    groups = _valid_group_count(groups, embed_dim, embed_dim)
    if groups <= 1:
        return None
    pieces = []
    for _ in range(3):
        pieces.append(_linear_group_mask(weight[:embed_dim], groups))
    return torch.cat(pieces, dim=0).to(weight.dtype)


class GroupedLinear(Linear):
    """Linear with a compile-time block-diagonal connectivity mask.

    The weight and bias parameter names stay identical to nn.Linear, so dense
    checkpoints can warm-start grouped projection experiments directly.
    """
    def __init__(self, in_features, out_features, bias=True, dtype=None,
                 device=None, groups=1):
        super().__init__(
            in_features, out_features, bias=bias, dtype=dtype, device=device
        )
        self.groups = _valid_group_count(groups, in_features, out_features)

    @property
    def grouped_active(self):
        return self.groups > 1

    def forward(self, input):
        mask = _linear_group_mask(self.weight, self.groups)
        weight = self.weight if mask is None else self.weight * mask
        return F.linear(input, weight, self.bias)


class NaiveComplexBatchNorm1d(Module):
    '''
    Naive approach to complex batch norm, perform batch norm independently on real and imaginary part.
    '''

    def __init__(self, num_features, eps=1e-5, momentum=0.1, affine=True,
                 track_running_stats=True, scale_mode="none",
                 tie_real_imag_scale=False, pow2_min_exp=-8, pow2_max_exp=8):
        super().__init__()
        self.bn_r = BatchNorm1d(
            num_features, eps, momentum, affine, track_running_stats)
        self.bn_i = BatchNorm1d(
            num_features, eps, momentum, affine, track_running_stats)
        self.scale_mode = scale_mode
        self.tie_real_imag_scale = bool(tie_real_imag_scale)
        self.pow2_min_exp = int(pow2_min_exp)
        self.pow2_max_exp = int(pow2_max_exp)

    def forward(self, input):
        if str(self.scale_mode).lower() in ("none", "off", "false") and not self.tie_real_imag_scale:
            return self.bn_r(input.real).type(torch.complex64) + 1j*self.bn_i(input.imag).type(torch.complex64)
        r_weight = self.bn_r.weight
        i_weight = self.bn_i.weight
        if self.tie_real_imag_scale and r_weight is not None and i_weight is not None:
            tied = 0.5 * (r_weight + i_weight)
            r_weight = tied
            i_weight = tied
        r_weight = _quantize_bn_scale(
            r_weight, self.scale_mode, self.pow2_min_exp, self.pow2_max_exp
        )
        i_weight = _quantize_bn_scale(
            i_weight, self.scale_mode, self.pow2_min_exp, self.pow2_max_exp
        )
        real = F.batch_norm(
            input.real, self.bn_r.running_mean, self.bn_r.running_var,
            r_weight, self.bn_r.bias, self.bn_r.training or not self.bn_r.track_running_stats,
            self.bn_r.momentum, self.bn_r.eps
        )
        imag = F.batch_norm(
            input.imag, self.bn_i.running_mean, self.bn_i.running_var,
            i_weight, self.bn_i.bias, self.bn_i.training or not self.bn_i.track_running_stats,
            self.bn_i.momentum, self.bn_i.eps
        )
        return real.type(torch.complex64) + 1j*imag.type(torch.complex64)


class NaiveComplexBatchNorm2d(Module):
    '''
    Naive approach to complex batch norm, perform batch norm independently on real and imaginary part.
    '''

    def __init__(self, num_features, eps=1e-5, momentum=0.1, affine=True,
                 track_running_stats=True, scale_mode="none",
                 tie_real_imag_scale=False, pow2_min_exp=-8, pow2_max_exp=8):
        super().__init__()
        self.bn_r = BatchNorm2d(
            num_features, eps, momentum, affine, track_running_stats)
        self.bn_i = BatchNorm2d(
            num_features, eps, momentum, affine, track_running_stats)
        self.scale_mode = scale_mode
        self.tie_real_imag_scale = bool(tie_real_imag_scale)
        self.pow2_min_exp = int(pow2_min_exp)
        self.pow2_max_exp = int(pow2_max_exp)

    def forward(self, input):
        if str(self.scale_mode).lower() in ("none", "off", "false") and not self.tie_real_imag_scale:
            return self.bn_r(input.real).type(torch.complex64) + 1j*self.bn_i(input.imag).type(torch.complex64)
        r_weight = self.bn_r.weight
        i_weight = self.bn_i.weight
        if self.tie_real_imag_scale and r_weight is not None and i_weight is not None:
            tied = 0.5 * (r_weight + i_weight)
            r_weight = tied
            i_weight = tied
        r_weight = _quantize_bn_scale(
            r_weight, self.scale_mode, self.pow2_min_exp, self.pow2_max_exp
        )
        i_weight = _quantize_bn_scale(
            i_weight, self.scale_mode, self.pow2_min_exp, self.pow2_max_exp
        )
        real = F.batch_norm(
            input.real, self.bn_r.running_mean, self.bn_r.running_var,
            r_weight, self.bn_r.bias, self.bn_r.training or not self.bn_r.track_running_stats,
            self.bn_r.momentum, self.bn_r.eps
        )
        imag = F.batch_norm(
            input.imag, self.bn_i.running_mean, self.bn_i.running_var,
            i_weight, self.bn_i.bias, self.bn_i.training or not self.bn_i.track_running_stats,
            self.bn_i.momentum, self.bn_i.eps
        )
        return real.type(torch.complex64) + 1j*imag.type(torch.complex64)


class _ComplexBatchNorm(Module):

    def __init__(self, num_features, eps=1e-5, momentum=0.1, affine=True,
                 track_running_stats=True):
        super().__init__()
        self.num_features = num_features
        self.eps = eps
        self.momentum = momentum
        self.affine = affine
        self.track_running_stats = track_running_stats
        if self.affine:
            self.weight = Parameter(torch.Tensor(num_features, 3))
            self.bias = Parameter(torch.Tensor(num_features, 2))
        else:
            self.register_parameter('weight', None)
            self.register_parameter('bias', None)
        if self.track_running_stats:
            self.register_buffer('running_mean', torch.zeros(
                num_features, dtype=torch.complex64))
            self.register_buffer('running_covar', torch.zeros(num_features, 3))
            self.running_covar[:, 0] = 1.4142135623730951
            self.running_covar[:, 1] = 1.4142135623730951
            self.register_buffer('num_batches_tracked',
                                 torch.tensor(0, dtype=torch.long))
        else:
            self.register_parameter('running_mean', None)
            self.register_parameter('running_covar', None)
            self.register_parameter('num_batches_tracked', None)
        self.reset_parameters()

    def reset_running_stats(self):
        if self.track_running_stats:
            self.running_mean.zero_()
            self.running_covar.zero_()
            self.running_covar[:, 0] = 1.4142135623730951
            self.running_covar[:, 1] = 1.4142135623730951
            self.num_batches_tracked.zero_()

    def reset_parameters(self):
        self.reset_running_stats()
        if self.affine:
            init.constant_(self.weight[:, :2], 1.4142135623730951)
            init.zeros_(self.weight[:, 2])
            init.zeros_(self.bias)


class ComplexBatchNorm2d(_ComplexBatchNorm):
    def forward(self, input):
        exponential_average_factor = 0.0

        if self.training and self.track_running_stats:
            if self.num_batches_tracked is not None:
                self.num_batches_tracked += 1
                if self.momentum is None:  # use cumulative moving average
                    exponential_average_factor = 1.0 / \
                        float(self.num_batches_tracked)
                else:  # use exponential moving average
                    exponential_average_factor = self.momentum

        if self.training or (not self.training and not self.track_running_stats):
            # calculate mean of real and imaginary part
            # mean does not support automatic differentiation for outputs with complex dtype.
            mean_r = input.real.mean([0, 2, 3]).type(torch.complex64)
            mean_i = input.imag.mean([0, 2, 3]).type(torch.complex64)
            mean = mean_r + 1j*mean_i
        else:
            mean = self.running_mean

        if self.training and self.track_running_stats:
            # update running mean
            with torch.no_grad():
                self.running_mean = exponential_average_factor * mean\
                    + (1 - exponential_average_factor) * self.running_mean

        input = input - mean[None, :, None, None]

        if self.training or (not self.training and not self.track_running_stats):
            # Elements of the covariance matrix (biased for train)
            n = input.numel() / input.size(1)
            Crr = 1./n*input.real.pow(2).sum(dim=[0, 2, 3])+self.eps
            Cii = 1./n*input.imag.pow(2).sum(dim=[0, 2, 3])+self.eps
            Cri = (input.real.mul(input.imag)).mean(dim=[0, 2, 3])
        else:
            Crr = self.running_covar[:, 0]+self.eps
            Cii = self.running_covar[:, 1]+self.eps
            Cri = self.running_covar[:, 2]  # +self.eps

        if self.training and self.track_running_stats:
            with torch.no_grad():
                self.running_covar[:, 0] = exponential_average_factor * Crr * n / (n - 1)\
                    + (1 - exponential_average_factor) * \
                    self.running_covar[:, 0]

                self.running_covar[:, 1] = exponential_average_factor * Cii * n / (n - 1)\
                    + (1 - exponential_average_factor) * \
                    self.running_covar[:, 1]

                self.running_covar[:, 2] = exponential_average_factor * Cri * n / (n - 1)\
                    + (1 - exponential_average_factor) * \
                    self.running_covar[:, 2]

        # calculate the inverse square root the covariance matrix
        det = Crr*Cii-Cri.pow(2)
        s = torch.sqrt(det)
        t = torch.sqrt(Cii+Crr + 2 * s)
        inverse_st = 1.0 / (s * t)
        Rrr = (Cii + s) * inverse_st
        Rii = (Crr + s) * inverse_st
        Rri = -Cri * inverse_st

        input = (Rrr[None, :, None, None]*input.real+Rri[None, :, None, None]*input.imag).type(torch.complex64) \
            + 1j*(Rii[None, :, None, None]*input.imag+Rri[None,
                  :, None, None]*input.real).type(torch.complex64)

        if self.affine:
            input = (self.weight[None, :, 0, None, None]*input.real+self.weight[None, :, 2, None, None]*input.imag +
                     self.bias[None, :, 0, None, None]).type(torch.complex64) \
                + 1j*(self.weight[None, :, 2, None, None]*input.real+self.weight[None, :, 1, None, None]*input.imag +
                      self.bias[None, :, 1, None, None]).type(torch.complex64)

        return input


class ComplexBatchNorm1d(_ComplexBatchNorm):
    def forward(self, input):
        exponential_average_factor = 0.0

        if self.training and self.track_running_stats:
            if self.num_batches_tracked is not None:
                self.num_batches_tracked += 1
                if self.momentum is None:  # use cumulative moving average
                    exponential_average_factor = 1.0 / \
                        float(self.num_batches_tracked)
                else:  # use exponential moving average
                    exponential_average_factor = self.momentum

        if self.training or (not self.training and not self.track_running_stats):
            # calculate mean of real and imaginary part
            mean_r = input.real.mean(dim=0).type(torch.complex64)
            mean_i = input.imag.mean(dim=0).type(torch.complex64)
            mean = mean_r + 1j*mean_i
        else:
            mean = self.running_mean

        if self.training and self.track_running_stats:
            # update running mean
            with torch.no_grad():
                self.running_mean = exponential_average_factor * mean\
                    + (1 - exponential_average_factor) * self.running_mean

        input = input - mean[None, ...]

        if self.training or (not self.training and not self.track_running_stats):
            # Elements of the covariance matrix (biased for train)
            n = input.numel() / input.size(1)
            Crr = input.real.var(dim=0, unbiased=False)+self.eps
            Cii = input.imag.var(dim=0, unbiased=False)+self.eps
            Cri = (input.real.mul(input.imag)).mean(dim=0)
        else:
            Crr = self.running_covar[:, 0]+self.eps
            Cii = self.running_covar[:, 1]+self.eps
            Cri = self.running_covar[:, 2]

        if self.training and self.track_running_stats:
            self.running_covar[:, 0] = exponential_average_factor * Crr * n / (n - 1)\
                + (1 - exponential_average_factor) * self.running_covar[:, 0]

            self.running_covar[:, 1] = exponential_average_factor * Cii * n / (n - 1)\
                + (1 - exponential_average_factor) * self.running_covar[:, 1]

            self.running_covar[:, 2] = exponential_average_factor * Cri * n / (n - 1)\
                + (1 - exponential_average_factor) * self.running_covar[:, 2]

        # calculate the inverse square root the covariance matrix
        det = Crr*Cii-Cri.pow(2)
        s = torch.sqrt(det)
        t = torch.sqrt(Cii+Crr + 2 * s)
        inverse_st = 1.0 / (s * t)
        Rrr = (Cii + s) * inverse_st
        Rii = (Crr + s) * inverse_st
        Rri = -Cri * inverse_st

        input = (Rrr[None, :]*input.real+Rri[None, :]*input.imag).type(torch.complex64) \
            + 1j*(Rii[None, :]*input.imag+Rri[None, :]
                  * input.real).type(torch.complex64)

        if self.affine:
            input = (self.weight[None, :, 0]*input.real+self.weight[None, :, 2]*input.imag +
                     self.bias[None, :, 0]).type(torch.complex64) \
                + 1j*(self.weight[None, :, 2]*input.real+self.weight[None, :, 1]*input.imag +
                      self.bias[None, :, 1]).type(torch.complex64)

        del Crr, Cri, Cii, Rrr, Rii, Rri, det, s, t
        return input


class ComplexGRUCell(Module):
    """
    A GRU cell for complex-valued inputs
    """

    def __init__(self, input_length, hidden_length):
        super().__init__()
        self.input_length = input_length
        self.hidden_length = hidden_length

        # reset gate components
        self.linear_reset_w1 = ComplexLinear(
            self.input_length, self.hidden_length)
        self.linear_reset_r1 = ComplexLinear(
            self.hidden_length, self.hidden_length)

        self.linear_reset_w2 = ComplexLinear(
            self.input_length, self.hidden_length)
        self.linear_reset_r2 = ComplexLinear(
            self.hidden_length, self.hidden_length)

        # update gate components
        self.linear_gate_w3 = ComplexLinear(
            self.input_length, self.hidden_length)
        self.linear_gate_r3 = ComplexLinear(
            self.hidden_length, self.hidden_length)

        self.activation_gate = ComplexSigmoid()
        self.activation_candidate = ComplexTanh()

    def reset_gate(self, x, h):
        x_1 = self.linear_reset_w1(x)
        h_1 = self.linear_reset_r1(h)
        # gate update
        reset = self.activation_gate(x_1 + h_1)
        return reset

    def update_gate(self, x, h):
        x_2 = self.linear_reset_w2(x)
        h_2 = self.linear_reset_r2(h)
        z = self.activation_gate(h_2 + x_2)
        return z

    def update_component(self, x, h, r):
        x_3 = self.linear_gate_w3(x)
        h_3 = r * self.linear_gate_r3(h)  # element-wise multiplication
        gate_update = self.activation_candidate(x_3 + h_3)
        return gate_update

    def forward(self, x, h):
        # Equation 1. reset gate vector
        r = self.reset_gate(x, h)

        # Equation 2: the update gate - the shared update gate vector z
        z = self.update_gate(x, h)

        # Equation 3: The almost output component
        n = self.update_component(x, h, r)

        # Equation 4: the new hidden state
        h_new = (1 + complex_opposite(z)) * n + \
            z * h  # element-wise multiplication

        return h_new


class ComplexBNGRUCell(Module):
    """
    A BN-GRU cell for complex-valued inputs
    """

    def __init__(self, input_length=10, hidden_length=20):
        super().__init__()
        self.input_length = input_length
        self.hidden_length = hidden_length

        # reset gate components
        self.linear_reset_w1 = ComplexLinear(
            self.input_length, self.hidden_length)
        self.linear_reset_r1 = ComplexLinear(
            self.hidden_length, self.hidden_length)

        self.linear_reset_w2 = ComplexLinear(
            self.input_length, self.hidden_length)
        self.linear_reset_r2 = ComplexLinear(
            self.hidden_length, self.hidden_length)

        # update gate components
        self.linear_gate_w3 = ComplexLinear(
            self.input_length, self.hidden_length)
        self.linear_gate_r3 = ComplexLinear(
            self.hidden_length, self.hidden_length)

        self.activation_gate = ComplexSigmoid()
        self.activation_candidate = ComplexTanh()

        self.bn = ComplexBatchNorm2d(1)

    def reset_gate(self, x, h):
        x_1 = self.linear_reset_w1(x)
        h_1 = self.linear_reset_r1(h)
        # gate update
        reset = self.activation_gate(self.bn(x_1) + self.bn(h_1))
        return reset

    def update_gate(self, x, h):
        x_2 = self.linear_reset_w2(x)
        h_2 = self.linear_reset_r2(h)
        z = self.activation_gate(self.bn(h_2) + self.bn(x_2))
        return z

    def update_component(self, x, h, r):
        x_3 = self.linear_gate_w3(x)
        # element-wise multiplication
        h_3 = r * self.bn(self.linear_gate_r3(h))
        gate_update = self.activation_candidate(self.bn(self.bn(x_3) + h_3))
        return gate_update

    def forward(self, x, h):
        # Equation 1. reset gate vector
        r = self.reset_gate(x, h)

        # Equation 2: the update gate - the shared update gate vector z
        z = self.update_gate(x, h)

        # Equation 3: The almost output component
        n = self.update_component(x, h, r)

        # Equation 4: the new hidden state
        h_new = (1 + complex_opposite(z)) * n + \
            z * h  # element-wise multiplication

        return h_new


class ComplexGRU(Module):
    def __init__(self, input_size, hidden_size, num_layers=1, bias=True,
                 batch_first=False, dropout=0, bidirectional=False):
        super().__init__()

        self.gru_re = GRU(input_size=input_size, hidden_size=hidden_size,
                            num_layers=num_layers, bias=bias,
                            batch_first=batch_first, dropout=dropout,
                            bidirectional=bidirectional)
        self.gru_im = GRU(input_size=input_size, hidden_size=hidden_size,
                            num_layers=num_layers, bias=bias,
                            batch_first=batch_first, dropout=dropout,
                            bidirectional=bidirectional)

    def forward(self, x):
        real, state_real = self._forward_real(x)
        imaginary, state_imag = self._forward_imaginary(x)

        output = torch.complex(real, imaginary)
        state = torch.complex(state_real, state_imag)

        return output, state

    def forward(self, x):
        r2r_out = self.gru_re(x.real)[0]
        r2i_out = self.gru_im(x.real)[0]
        i2r_out = self.gru_re(x.imag)[0]
        i2i_out = self.gru_im(x.imag)[0]
        real_out = r2r_out - i2i_out
        imag_out = i2r_out + r2i_out 

        return torch.complex(real_out, imag_out), None

    def _forward_real(self, x):
        real_real, h_real = self.gru_re(x.real)
        imag_imag, h_imag = self.gru_im(x.imag)
        real = real_real - imag_imag

        return real, torch.complex(h_real, h_imag)

    def _forward_imaginary(self, x):
        imag_real, h_real = self.gru_re(x.imag)
        real_imag, h_imag = self.gru_im(x.real)
        imaginary = imag_real + real_imag

        return imaginary, torch.complex(h_real, h_imag)


class ComplexLSTM(Module):
    def __init__(self, input_size, hidden_size, num_layers=1, bias=True,
                 batch_first=False, dropout=0, bidirectional=False):
        super().__init__()
        self.num_layer = num_layers
        self.hidden_size = hidden_size
        self.batch_dim = 0 if batch_first else 1
        self.bidirectional = bidirectional

        self.lstm_re = LSTM(input_size=input_size, hidden_size=hidden_size,
                            num_layers=num_layers, bias=bias,
                            batch_first=batch_first, dropout=dropout,
                            bidirectional=bidirectional,
                            dtype=torch.float32)
        self.lstm_im = LSTM(input_size=input_size, hidden_size=hidden_size,
                            num_layers=num_layers, bias=bias,
                            batch_first=batch_first, dropout=dropout,
                            bidirectional=bidirectional,
                            dtype=torch.float32)
    def forward(self, x):
        real, state_real = self._forward_real(x)
        imaginary, state_imag = self._forward_imaginary(x)

        output = torch.complex(real, imaginary)

        return output, (state_real, state_imag)

    def _forward_real(self, x):
        h_real, h_imag, c_real, c_imag = self._init_state(self._get_batch_size(x), x.is_cuda)
        real_real, (h_real, c_real) = self.lstm_re(x.real, (h_real, c_real))
        imag_imag, (h_imag, c_imag) = self.lstm_im(x.imag, (h_imag, c_imag))
        real = real_real - imag_imag
        return real, ((h_real, c_real), (h_imag, c_imag))

    def _forward_imaginary(self, x):
        h_real, h_imag, c_real, c_imag = self._init_state(self._get_batch_size(x), x.is_cuda)
        imag_real, (h_real, c_real) = self.lstm_re(x.imag, (h_real, c_real))
        real_imag, (h_imag, c_imag) = self.lstm_im(x.real, (h_imag, c_imag))
        imaginary = imag_real + real_imag

        return imaginary, ((h_real, c_real), (h_imag, c_imag))

    def _init_state(self, batch_size, to_gpu=False):
        dim_0 = 2 if self.bidirectional else 1
        dims = (dim_0*self.num_layer, batch_size, self.hidden_size)

        h_real, h_imag, c_real, c_imag = [
            torch.zeros(dims, dtype=torch.float32) for i in range(4)]

        if to_gpu:
            h_real, h_imag, c_real, c_imag = [
                t.cuda() for t in [h_real, h_imag, c_real, c_imag]]

        return h_real, h_imag, c_real, c_imag
    
    def _get_batch_size(self, x):
        return x.size(self.batch_dim)


# class ComplexMultiheadAttention(Module):
#     def __init__(self, embed_dim=512, num_heads=8, batch_first=True):
#         super().__init__()
#         self.embed_dim = embed_dim
#         self.num_heads = num_heads
#         print(f"ComplexMultiheadAttention: embed_dim={embed_dim}, num_heads={num_heads}, batch_first={batch_first}")

#         self.MultiheadAttention_re = MultiheadAttention(self.embed_dim, self.num_heads, batch_first=batch_first)
#         self.MultiheadAttention_im = MultiheadAttention(self.embed_dim, self.num_heads, batch_first=batch_first)
   
#     def forward(self, x):
#         real = self._forward_real(x)
#         imaginary = self._forward_imaginary(x)

#         output = torch.complex(real, imaginary)

#         return output

#     def _forward_real(self, x):
#         real_real,_ = self.MultiheadAttention_re(x.real, x.real, x.real)
#         imag_imag,_ = self.MultiheadAttention_im(x.imag, x.imag, x.imag)
#         real = real_real - imag_imag
#         return real

#     def _forward_imaginary(self, x):
#         imag_real,_ = self.MultiheadAttention_re(x.imag, x.imag, x.imag)
#         real_imag,_ = self.MultiheadAttention_im(x.real, x.real, x.real)
#         imaginary = imag_real + real_imag

#         return imaginary

def _canonical_softmax_mode(mode):
    if mode is None:
        return "softmax"
    mode = str(mode).lower().replace("-", "_")
    aliases = {
        "default": "softmax",
        "exact": "softmax",
        "torch": "softmax",
        "shift_based": "shift",
        "shift_pow2": "shift",
        "shift_topk": "shift_topk_pow2",
        "shift_top4": "shift_topk_pow2",
        "topk_shift": "shift_topk_pow2",
        "fused_shift": "shift_topk_pow2",
        "shift_topk_fused": "shift_topk_pow2",
        "shift_topk_floor": "shift_topk_pow2_floor",
        "topk_shift_floor": "shift_topk_pow2_floor",
        "floor": "shift_topk_pow2_floor",
        "shift_topk_ceil": "shift_topk_pow2_ceil",
        "topk_shift_ceil": "shift_topk_pow2_ceil",
        "ceil": "shift_topk_pow2_ceil",
        "shift_topk_clamp8": "shift_topk_pow2_clamp8",
        "topk_shift_clamp8": "shift_topk_pow2_clamp8",
        "clamp8": "shift_topk_pow2_clamp8",
        "shift_topk_mantissa_free": "shift_topk_pow2_nomantissa",
        "shift_topk_no_mantissa": "shift_topk_pow2_nomantissa",
        "shift_topk_mfree": "shift_topk_pow2_nomantissa",
        "mantissa_free": "shift_topk_pow2_nomantissa",
        "no_mantissa": "shift_topk_pow2_nomantissa",
        "nomantissa": "shift_topk_pow2_nomantissa",
        "shift_topk_clamp8_nomantissa": "shift_topk_pow2_clamp8_nomantissa",
        "shift_topk_clamp8_mantissa_free": "shift_topk_pow2_clamp8_nomantissa",
        "shift_topk_nomantissa_clamp8": "shift_topk_pow2_clamp8_nomantissa",
        "clamp8_nomantissa": "shift_topk_pow2_clamp8_nomantissa",
        "clamp8_mantissa_free": "shift_topk_pow2_clamp8_nomantissa",
        "nomantissa_clamp8": "shift_topk_pow2_clamp8_nomantissa",
        "taylor": "taylor2",
        "poly2": "taylor2",
        "relu_sq": "relu2",
        "relu_square": "relu2",
        "square_relu": "relu2",
    }
    return aliases.get(mode, mode)


def _normalize_attention_weights(weights, valid_mask):
    weights = weights.masked_fill(~valid_mask, 0.0)
    denom = weights.sum(dim=-1, keepdim=True)
    valid_count = valid_mask.sum(dim=-1, keepdim=True).clamp_min(1)
    uniform = valid_mask.to(weights.dtype) / valid_count.to(weights.dtype)
    eps = 1e-6 if weights.dtype == torch.float16 else 1e-12
    return torch.where(denom > eps, weights / denom.clamp_min(eps), uniform)


def _attention_datapath_enabled(attention_datapath):
    return bool(attention_datapath and attention_datapath.get("enabled", False))


def _attention_datapath_value(attention_datapath, key, default=None):
    if not attention_datapath:
        return default
    return attention_datapath.get(key, default)


def _fp8_dtype_from_mode(mode):
    mode = str(mode).lower()
    if mode in ("fp8_e4m3", "e4m3", "float8_e4m3"):
        return torch.float8_e4m3fn
    if mode in ("fp8_e5m2", "e5m2", "float8_e5m2"):
        return torch.float8_e5m2
    raise ValueError(f"Unsupported FP8 mode: {mode}")


def _fp8_max_value(fp8_dtype):
    return 448.0 if fp8_dtype == torch.float8_e4m3fn else 57344.0


def _fake_fp8_quantize_tensor(x, fp8_dtype):
    max_val = _fp8_max_value(fp8_dtype)
    current_amax = x.abs().max().clamp(min=1e-12)
    scale = max_val / current_amax
    return (x * scale).to(fp8_dtype).to(x.dtype) / scale


class _RoundSTE(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        return torch.round(x)

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output


class _FloorSTE(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        return torch.floor(x)

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output


def _ste_round(x):
    return _RoundSTE.apply(x)


def _ste_floor(x):
    return _FloorSTE.apply(x)


def _finite_score_value_mask(attn, finite_mask_threshold):
    value_mask = torch.isfinite(attn)
    if finite_mask_threshold is not None:
        value_mask = value_mask & (attn > float(finite_mask_threshold))
    return value_mask


def _quantize_attention_scores(attn, attention_datapath):
    if not _attention_datapath_enabled(attention_datapath):
        return attn

    mode = str(_attention_datapath_value(attention_datapath, "score_quant", "none")).lower()
    if mode in ("none", "off", "false"):
        return attn

    finite_mask_threshold = _attention_datapath_value(
        attention_datapath, "finite_mask_threshold", -10000.0
    )
    value_mask = _finite_score_value_mask(attn, finite_mask_threshold)
    finite_mask_like = torch.isfinite(attn) & ~value_mask

    if mode in ("fp8_e4m3", "e4m3", "fp8_e5m2", "e5m2"):
        fp8_dtype = _fp8_dtype_from_mode(mode)
        safe_scores = torch.where(value_mask, attn, torch.zeros_like(attn))
        quantized = _fake_fp8_quantize_tensor(safe_scores, fp8_dtype)
        quantized = torch.where(value_mask, quantized, attn)
        if finite_mask_like.any():
            clamp_value = torch.full_like(attn, -_fp8_max_value(fp8_dtype))
            quantized = torch.where(finite_mask_like, clamp_value, quantized)
        return quantized

    if mode == "fixed":
        int_bits = _attention_datapath_value(attention_datapath, "score_fixed_int_bits")
        frac_bits = _attention_datapath_value(attention_datapath, "score_fixed_frac_bits")
        if int_bits is None or frac_bits is None:
            raise ValueError(
                "score_quant=fixed requires score_fixed_int_bits and "
                "score_fixed_frac_bits from the B0 score dump."
            )
        int_bits = int(int_bits)
        frac_bits = int(frac_bits)
        step = 2.0 ** (-frac_bits)
        qmin = -(2.0 ** (int_bits - 1))
        qmax = (2.0 ** (int_bits - 1)) - step
        safe_scores = torch.where(value_mask, attn, torch.zeros_like(attn))
        quantized = _ste_round(safe_scores / step).clamp(qmin / step, qmax / step) * step
        quantized = torch.where(value_mask, quantized, attn)
        if finite_mask_like.any():
            clamp_value = torch.full_like(attn, qmin)
            quantized = torch.where(finite_mask_like, clamp_value, quantized)
        return quantized

    raise ValueError(f"Unsupported attention score_quant mode: {mode}")


def _quantize_attention_probs(probs, attention_datapath):
    if not _attention_datapath_enabled(attention_datapath):
        return probs

    mode = str(_attention_datapath_value(attention_datapath, "prob_quant", "none")).lower()
    if mode in ("none", "off", "false"):
        return probs

    valid = torch.isfinite(probs)
    values = probs.clamp(min=0.0, max=1.0)

    if mode in ("q0.8", "q08", "u0.8"):
        scale = 256.0
        quantized = _ste_round(values * scale).clamp(0.0, scale) / scale
        return torch.where(valid, quantized, probs)

    if mode in ("fp8_e4m3", "e4m3", "fp8_e5m2", "e5m2"):
        fp8_dtype = _fp8_dtype_from_mode(mode)
        safe_values = torch.where(valid, values, torch.zeros_like(values))
        quantized = _fake_fp8_quantize_tensor(safe_values, fp8_dtype).clamp(0.0, 1.0)
        return torch.where(valid, quantized, probs)

    if mode == "pow2":
        min_exp = int(_attention_datapath_value(attention_datapath, "prob_pow2_min_exp", -16))
        positive = valid & (values > 0)
        exponent = torch.round(torch.log2(values.clamp_min(2.0 ** min_exp))).clamp_min(min_exp)
        quantized = torch.pow(torch.full_like(values, 2.0), exponent).clamp(max=1.0)
        return torch.where(positive, quantized, torch.zeros_like(values))

    raise ValueError(f"Unsupported attention prob_quant mode: {mode}")


def _prob_packet_config(attention_datapath):
    if not _attention_datapath_enabled(attention_datapath):
        return None
    packet_cfg = _attention_datapath_value(attention_datapath, "prob_packet", None)
    if not packet_cfg or not bool(packet_cfg.get("enabled", False)):
        return None
    return packet_cfg


def _prob_packet_window_width(packet_cfg, idx_bits):
    encoded_width = 2 ** int(idx_bits)
    requested = packet_cfg.get("causal_window", None)
    if requested is None:
        return encoded_width

    window_width = int(requested)
    if window_width <= 0:
        raise ValueError("q0.8 probability packet causal_window must be positive.")
    if window_width > encoded_width:
        raise ValueError(
            "q0.8 probability packet causal_window exceeds local_idx encoding: "
            f"causal_window={window_width}, local_idx_bits={idx_bits}, "
            f"max_encodable={encoded_width}."
        )
    return window_width


def _packetize_q08_attention_probs(probs, packet_cfg):
    top_k = int(packet_cfg.get("top_k", 4))
    idx_bits = int(packet_cfg.get("local_idx_bits", 5))
    encoded_width = 2 ** idx_bits
    window_width = _prob_packet_window_width(packet_cfg, idx_bits)
    prob_scale = float(packet_cfg.get("prob_scale", 256))
    prob_max_code = int(packet_cfg.get("prob_max_code", 255))
    validate_local_idx = bool(packet_cfg.get("validate_local_idx", True))
    renormalize = bool(packet_cfg.get("renormalize", False))

    q_len = probs.size(-2)
    key_len = probs.size(-1)
    k = min(max(1, top_k), key_len)

    row = torch.arange(q_len, device=probs.device).view(
        *([1] * (probs.dim() - 2)), q_len, 1
    )
    key = torch.arange(key_len, device=probs.device).view(
        *([1] * (probs.dim() - 1)), key_len
    )
    local_idx_full = row - key
    in_window = (local_idx_full >= 0) & (local_idx_full < window_width)
    in_window = in_window.expand_as(probs)

    masked_probs = probs.clamp(min=0.0).masked_fill(~in_window, float("-inf"))
    topk_probs, topk_idx = torch.topk(masked_probs, k=k, dim=-1)
    valid_slot = torch.gather(in_window, -1, topk_idx) & torch.isfinite(topk_probs)
    safe_row = row.expand_as(topk_idx)
    safe_idx = torch.where(valid_slot, topk_idx, safe_row)
    safe_probs = torch.where(valid_slot, topk_probs, torch.zeros_like(topk_probs))

    if validate_local_idx:
        local_idx = row - safe_idx
        invalid = valid_slot & ((local_idx < 0) | (local_idx >= encoded_width))
        if torch.any(invalid):
            raise ValueError(
                "Internal bug: q0.8 probability packet local_idx overflow after "
                "window-aware top-k. The packetizer selected an entry outside "
                f"the encodable causal window ({encoded_width} keys)."
            )

    codes = _ste_round(safe_probs * prob_scale).clamp(0.0, float(prob_max_code))
    packet_probs = codes / prob_scale
    if renormalize:
        denom = packet_probs.sum(dim=-1, keepdim=True)
        packet_probs = torch.where(
            denom > 0,
            packet_probs / denom.clamp_min(1.0 / prob_scale),
            packet_probs,
        )

    dense_probs = torch.zeros_like(probs).scatter_add(
        -1, safe_idx, packet_probs.to(probs.dtype)
    )
    dense_codes = torch.zeros_like(probs).scatter_add(
        -1, safe_idx, codes.to(probs.dtype)
    )
    return {
        "probs": dense_probs,
        "codes": dense_codes,
        "indices": safe_idx,
        "probs_topk": packet_probs,
        "codes_topk": codes,
        "valid_slot": valid_slot,
        "window_width": window_width,
    }


def _gather_packet_values(v, packet_indices):
    key_len = v.size(-2)
    value_dim = v.size(-1)
    q_len = packet_indices.size(-2)
    top_k = packet_indices.size(-1)

    flat_v = v.reshape(-1, key_len, value_dim)
    flat_idx = packet_indices.reshape(-1, q_len, top_k)
    batch_idx = torch.arange(flat_v.size(0), device=v.device).view(-1, 1, 1)
    gathered = flat_v[batch_idx, flat_idx]
    return gathered.reshape(*v.shape[:-2], q_len, top_k, value_dim)


def _packet_slots_in_key_order(packet):
    order = torch.argsort(packet["indices"], dim=-1)
    return {
        "indices": torch.gather(packet["indices"], -1, order),
        "probs_topk": torch.gather(packet["probs_topk"], -1, order),
        "codes_topk": torch.gather(packet["codes_topk"], -1, order),
    }


def _q08_packet_fixed_matmul(packet, v, packet_cfg):
    prob_scale = float(packet_cfg.get("prob_scale", 256))
    slots = _packet_slots_in_key_order(packet)
    selected_v = _gather_packet_values(v, slots["indices"]).to(torch.float32)
    products = slots["codes_topk"].unsqueeze(-1).to(torch.float32) * selected_v
    return (products.sum(dim=-2) / prob_scale).to(v.dtype)


def _q08_packet_dequant_matmul(packet, v, out_dtype):
    slots = _packet_slots_in_key_order(packet)
    selected_v = _gather_packet_values(v, slots["indices"])
    packet_probs = slots["probs_topk"].to(out_dtype)
    selected_v = selected_v.to(out_dtype)
    return (packet_probs.unsqueeze(-1) * selected_v).sum(dim=-2).to(v.dtype)


def _q08_packet_lmul_matmul(packet, v, mantissa_bits=4, bit_exact=False):
    slots = _packet_slots_in_key_order(packet)
    selected_v = _gather_packet_values(v, slots["indices"])
    multiply = _lmul_multiply_bit_exact if bit_exact else _lmul_multiply
    products = multiply(
        slots["probs_topk"].unsqueeze(-1),
        selected_v,
        mantissa_bits,
    )
    return products.sum(dim=-2).to(v.dtype)


def _apply_qk_scale(scores, head_dim, attention_datapath):
    if not _attention_datapath_enabled(attention_datapath):
        return scores / math.sqrt(head_dim)

    mode = str(_attention_datapath_value(attention_datapath, "qk_scale_mode", "pow2")).lower()
    if mode == "pow2":
        sqrt_head = math.sqrt(head_dim)
        exponent = math.log2(sqrt_head)
        if abs(exponent - round(exponent)) <= 1e-12:
            shift = torch.tensor(
                -int(round(exponent)), dtype=torch.int32, device=scores.device
            )
            return torch.ldexp(scores, shift)
        return scores / sqrt_head

    if mode == "fp16":
        scale = torch.tensor(1.0 / math.sqrt(head_dim), device=scores.device, dtype=torch.float16)
        return scores * scale.to(scores.dtype)

    if mode in ("fp8_e4m3", "e4m3", "fp8_e5m2", "e5m2"):
        fp8_dtype = _fp8_dtype_from_mode(mode)
        scale = torch.tensor(1.0 / math.sqrt(head_dim), device=scores.device, dtype=torch.float32)
        scale = scale.to(fp8_dtype).to(scores.dtype)
        return scores * scale

    raise ValueError(f"Unsupported qk_scale_mode: {mode}")


def _capture_attention_scores(attention_datapath, scores, mask):
    if not attention_datapath:
        return
    capture = attention_datapath.get("score_capture")
    if callable(capture):
        capture(scores.detach(), mask.detach() if isinstance(mask, torch.Tensor) else None)


def _lmul_offset(mantissa_bits):
    mantissa_bits = int(mantissa_bits)
    if mantissa_bits <= 3:
        return mantissa_bits
    if mantissa_bits == 4:
        return 3
    return 4


def _quantize_fraction_bits(frac, mantissa_bits):
    mantissa_bits = int(mantissa_bits)
    if mantissa_bits <= 0:
        return torch.zeros_like(frac)
    scale = float(2 ** mantissa_bits)
    return _ste_floor(frac.clamp(0.0, 1.0 - 1.0 / scale) * scale) / scale


def _lmul_multiply(x, y, mantissa_bits=4):
    """Approximate floating-point multiply with L-Mul mantissa addition.

    This is a tensor-level simulator for feasibility screening, not a bit-exact
    RTL model. It keeps sign/exponent handling explicit, replaces mantissa
    multiplication by mantissa addition plus the paper's 2^-l(m) offset, and
    accumulates products with normal torch additions.
    """
    out_dtype = torch.promote_types(x.dtype, y.dtype)
    x_f = x.to(torch.float32)
    y_f = y.to(torch.float32)
    zero = (x_f == 0) | (y_f == 0) | ~torch.isfinite(x_f) | ~torch.isfinite(y_f)

    sign = torch.sign(x_f) * torch.sign(y_f)
    x_abs = x_f.abs()
    y_abs = y_f.abs()

    mx, ex = torch.frexp(x_abs)
    my, ey = torch.frexp(y_abs)
    sx = (mx * 2.0).clamp(1.0, 2.0)
    sy = (my * 2.0).clamp(1.0, 2.0)
    ex = ex - 1
    ey = ey - 1

    fx = _quantize_fraction_bits(sx - 1.0, mantissa_bits)
    fy = _quantize_fraction_bits(sy - 1.0, mantissa_bits)
    sig = 1.0 + fx + fy + (2.0 ** (-_lmul_offset(mantissa_bits)))

    carry = sig >= 2.0
    sig = torch.where(carry, sig * 0.5, sig)
    exp = ex + ey + carry.to(ex.dtype)
    approx = torch.ldexp(sig, exp) * sign
    approx = torch.where(zero, torch.zeros_like(approx), approx)
    return approx.to(out_dtype)


def _lmul_multiply_bit_exact(x, y, mantissa_bits=4):
    """Bit-field L-Mul simulator with explicit mantissa integer codes."""
    out_dtype = torch.promote_types(x.dtype, y.dtype)
    x_f = x.to(torch.float32)
    y_f = y.to(torch.float32)
    zero = (x_f == 0) | (y_f == 0) | ~torch.isfinite(x_f) | ~torch.isfinite(y_f)

    sign = torch.sign(x_f) * torch.sign(y_f)
    mx, ex = torch.frexp(x_f.abs())
    my, ey = torch.frexp(y_f.abs())
    sx = (mx * 2.0).clamp(1.0, 2.0)
    sy = (my * 2.0).clamp(1.0, 2.0)
    ex = ex - 1
    ey = ey - 1

    scale = float(2 ** int(mantissa_bits))
    max_code = scale - 1.0
    fx_code = _ste_floor((sx - 1.0).clamp(0.0, 1.0 - 1.0 / scale) * scale).clamp(0.0, max_code)
    fy_code = _ste_floor((sy - 1.0).clamp(0.0, 1.0 - 1.0 / scale) * scale).clamp(0.0, max_code)
    offset_code = float(round(scale * (2.0 ** (-_lmul_offset(mantissa_bits)))))
    sig_code = scale + fx_code + fy_code + offset_code

    carry = sig_code >= (2.0 * scale)
    sig_code = torch.where(carry, _ste_floor(sig_code / 2.0), sig_code)
    sig = sig_code / scale
    exp = ex + ey + carry.to(ex.dtype)
    approx = torch.ldexp(sig, exp) * sign
    approx = torch.where(zero, torch.zeros_like(approx), approx)
    return approx.to(out_dtype)


def _lmul_matmul(a, b, mantissa_bits=4, bit_exact=False):
    multiply = _lmul_multiply_bit_exact if bit_exact else _lmul_multiply
    products = multiply(a.unsqueeze(-1), b.unsqueeze(-3), mantissa_bits)
    return products.sum(dim=-2)


def _infer_causal_window_from_mask(attn_mask, q_len, key_len):
    if attn_mask is None or q_len != key_len or q_len <= 0:
        return None

    mask = attn_mask
    while mask.dim() > 2:
        mask = mask[0]
    if mask.dim() != 2 or mask.shape[-2:] != (q_len, key_len):
        return None

    valid = ~mask if mask.dtype == torch.bool else torch.isfinite(mask)
    valid_per_row = valid.sum(dim=-1)
    if valid_per_row.numel() == 0:
        return None

    window_width = int(valid_per_row.max().detach().cpu().item())
    if window_width <= 0 or window_width >= key_len:
        return None
    return window_width


def _lmul_windowed_qk_matmul(q, k_t, window_width, mantissa_bits=4,
                             bit_exact=False, chunk_size=32):
    q_len = q.size(-2)
    key_len = k_t.size(-1)
    if window_width is None or window_width >= key_len:
        return _lmul_matmul(
            q, k_t, mantissa_bits=mantissa_bits, bit_exact=bit_exact
        )

    chunk_size = max(1, int(chunk_size))
    scores = q.new_zeros(*q.shape[:-1], key_len)
    for q_start in range(0, q_len, chunk_size):
        q_end = min(q_start + chunk_size, q_len)
        key_start = max(0, q_start - int(window_width) + 1)
        key_end = min(key_len, q_end)
        if key_start >= key_end:
            continue

        sub_scores = _lmul_matmul(
            q[..., q_start:q_end, :],
            k_t[..., :, key_start:key_end],
            mantissa_bits=mantissa_bits,
            bit_exact=bit_exact,
        )
        rows = torch.arange(q_start, q_end, device=q.device).view(-1, 1)
        keys = torch.arange(key_start, key_end, device=q.device).view(1, -1)
        local_idx = rows - keys
        valid = (local_idx >= 0) & (local_idx < int(window_width))
        valid = valid.view(*([1] * (sub_scores.dim() - 2)), q_end - q_start, key_end - key_start)
        scores[..., q_start:q_end, key_start:key_end] = sub_scores.masked_fill(~valid, 0.0)

    return scores


def _lmul_target_enabled(attention_datapath, target):
    if not _attention_datapath_enabled(attention_datapath):
        return False
    if not bool(_attention_datapath_value(attention_datapath, "lmul_enabled", False)):
        return False
    targets = _attention_datapath_value(attention_datapath, "lmul_targets", [])
    if isinstance(targets, str):
        targets = [targets]
    return target in set(targets)


def _shift_based_softmax(attn, valid_mask):
    """Adder/shifter-friendly exp approximation for hardware experiments.

    Approximate exp(x) as 2^(1.5x). The fractional 2^f term is linearly
    approximated as 1 + f, and the denominator is rounded to the nearest
    power-of-two to mimic a shift reciprocal.
    """
    min_value = torch.finfo(attn.dtype).min
    safe_attn = attn.masked_fill(~valid_mask, min_value)
    row_max = safe_attn.max(dim=-1, keepdim=True).values
    row_max = row_max.masked_fill(~torch.isfinite(row_max), 0.0)

    x = (safe_attn - row_max).masked_fill(~valid_mask, 0.0)
    scaled = (x * 1.5).clamp(min=-16.0, max=0.0)
    integer = torch.floor(scaled)
    frac = scaled - integer
    mantissa = 1.0 + frac
    weights = torch.ldexp(mantissa, integer.to(torch.int32))
    weights = weights.masked_fill(~valid_mask, 0.0)

    denom = weights.sum(dim=-1, keepdim=True)
    valid_count = valid_mask.sum(dim=-1, keepdim=True).clamp_min(1)
    uniform = valid_mask.to(weights.dtype) / valid_count.to(weights.dtype)

    denom_f32 = denom.to(torch.float32).clamp_min(1e-12)
    denom_pow2 = torch.pow(
        torch.full_like(denom_f32, 2.0),
        torch.round(torch.log2(denom_f32)),
    ).to(weights.dtype)
    approx = weights / denom_pow2.clamp_min(1e-6 if weights.dtype == torch.float16 else 1e-12)
    return torch.where(denom > 0, approx, uniform)


def _round_power_of_two(denom, dtype, rounding="nearest"):
    denom_f32 = denom.to(torch.float32).clamp_min(1e-12)
    exponent = torch.log2(denom_f32)
    if rounding == "floor":
        exponent = torch.floor(exponent)
    elif rounding == "ceil":
        exponent = torch.ceil(exponent)
    elif rounding == "nearest":
        exponent = torch.round(exponent)
    else:
        raise ValueError(f"Unsupported power-of-two rounding mode: {rounding}")
    denom_pow2 = torch.pow(
        torch.full_like(denom_f32, 2.0),
        exponent,
    )
    return denom_pow2.to(dtype)


def _shift_topk_config(softmax_mode):
    softmax_mode = _canonical_softmax_mode(softmax_mode)
    configs = {
        "shift_topk_exact": {
            "pow2_denom": False,
            "denom_rounding": "nearest",
            "clamp_min": -16.0,
            "use_mantissa": True,
        },
        "shift_topk_pow2": {
            "pow2_denom": True,
            "denom_rounding": "nearest",
            "clamp_min": -16.0,
            "use_mantissa": True,
        },
        "shift_topk_pow2_floor": {
            "pow2_denom": True,
            "denom_rounding": "floor",
            "clamp_min": -16.0,
            "use_mantissa": True,
        },
        "shift_topk_pow2_ceil": {
            "pow2_denom": True,
            "denom_rounding": "ceil",
            "clamp_min": -16.0,
            "use_mantissa": True,
        },
        "shift_topk_pow2_clamp8": {
            "pow2_denom": True,
            "denom_rounding": "nearest",
            "clamp_min": -8.0,
            "use_mantissa": True,
        },
        "shift_topk_pow2_nomantissa": {
            "pow2_denom": True,
            "denom_rounding": "nearest",
            "clamp_min": -16.0,
            "use_mantissa": False,
        },
        "shift_topk_pow2_clamp8_nomantissa": {
            "pow2_denom": True,
            "denom_rounding": "nearest",
            "clamp_min": -8.0,
            "use_mantissa": False,
        },
    }
    return configs.get(softmax_mode)


def _shift_topk_softmax(attn, valid_mask, top_k, pow2_denom=True,
                        denom_rounding="nearest", clamp_min=-16.0,
                        use_mantissa=True):
    """Fuse top-k selection into the shift approximation.

    Existing strict eval already keeps only top-k probabilities after softmax.
    Since the shift transform is monotonic in the score, selecting top-k scores
    first keeps the same support while computing weights for only k entries.
    """
    if top_k is None or top_k <= 0:
        return _shift_based_softmax(attn, valid_mask)

    k = min(int(top_k), attn.size(-1))
    masked_attn = attn.masked_fill(~valid_mask, float("-inf"))
    topk_scores, topk_idx = torch.topk(masked_attn, k=k, dim=-1)
    topk_valid = torch.isfinite(topk_scores)

    row_max = topk_scores[..., :1].masked_fill(~topk_valid[..., :1], 0.0)
    x = (topk_scores - row_max).masked_fill(~topk_valid, 0.0)
    scaled = (x * 1.5).clamp(min=clamp_min, max=0.0)
    integer = torch.floor(scaled)
    frac = scaled - integer
    mantissa = (1.0 + frac) if use_mantissa else torch.ones_like(frac)
    weights = torch.ldexp(mantissa, integer.to(torch.int32))
    weights = weights.masked_fill(~topk_valid, 0.0)

    denom = weights.sum(dim=-1, keepdim=True)
    valid_count = valid_mask.sum(dim=-1, keepdim=True).clamp_min(1)
    uniform = valid_mask.to(weights.dtype) / valid_count.to(weights.dtype)

    if pow2_denom:
        denom_norm = _round_power_of_two(denom, weights.dtype, denom_rounding)
    else:
        denom_norm = denom

    eps = 1e-6 if weights.dtype == torch.float16 else 1e-12
    topk_probs = (weights / denom_norm.clamp_min(eps)).to(attn.dtype)
    probs = torch.zeros_like(attn).scatter(-1, topk_idx, topk_probs)
    return torch.where(denom > 0, probs, uniform.to(attn.dtype)).to(attn.dtype)


def _approx_attention_probs(attn, softmax_mode):
    softmax_mode = _canonical_softmax_mode(softmax_mode)
    if softmax_mode == "softmax":
        return F.softmax(attn, dim=-1)

    valid_mask = torch.isfinite(attn)
    safe_attn = attn.masked_fill(~valid_mask, 0.0)

    if softmax_mode == "shift":
        return _shift_based_softmax(attn, valid_mask)
    if softmax_mode == "shift_exact":
        min_value = torch.finfo(attn.dtype).min
        centered = attn.masked_fill(~valid_mask, min_value)
        centered = centered - centered.max(dim=-1, keepdim=True).values
        scaled = (centered * 1.5).clamp(min=-16.0, max=0.0)
        integer = torch.floor(scaled)
        frac = scaled - integer
        weights = torch.ldexp(1.0 + frac, integer.to(torch.int32))
        return _normalize_attention_weights(weights, valid_mask)
    if softmax_mode == "taylor2":
        weights = (1.0 + safe_attn + 0.5 * safe_attn * safe_attn).clamp_min(0.0)
        return _normalize_attention_weights(weights, valid_mask)
    if softmax_mode == "relu2":
        weights = F.relu(safe_attn).square()
        return _normalize_attention_weights(weights, valid_mask)

    raise ValueError(
        f"Unsupported attention softmax mode: {softmax_mode}. "
        "Expected one of: softmax, shift, shift_exact, shift_topk_exact, "
        "shift_topk_pow2, shift_topk_pow2_floor, shift_topk_pow2_ceil, "
        "shift_topk_pow2_clamp8, shift_topk_pow2_nomantissa, "
        "shift_topk_pow2_clamp8_nomantissa, taylor2, relu2."
    )


def _mha_attention_forward(mha, x, attn_mask, top_k, activation_quantizer=None,
                           fp16_accumulator=False, softmax_mode="softmax",
                           attention_datapath=None, mha_projection=None):
    """Manually recreate nn.MultiheadAttention forward with optional top-k masking
    after softmax. Uses `mha.in_proj_weight`, `mha.in_proj_bias`, `mha.out_proj`
    directly so any parametrization (e.g. INT4 fake quant) applies transparently.

    Args:
        mha: nn.MultiheadAttention (batch_first=True expected).
        x:   (B, T, E) real tensor (one half of the complex split).
        attn_mask: broadcastable additive mask (T, T) or (B*num_heads, T, T).
        top_k: int or None — retain only top-k entries per query row of softmax.
        activation_quantizer: optional callable tensor→tensor for FP8 activation
            quant before matmuls (simulates W4A8 data path).
        fp16_accumulator: if True, run QK^T, softmax, and Score×V in fp16 to
            match the hardware datapath (per 07 handoff §2.1). Output is cast
            back to the original activation dtype before out_proj.
        softmax_mode: "softmax", "shift", "taylor2", or "relu2".
        attention_datapath: optional P3/P4/P5 datapath dict for QK scale,
            finite score quantization, probability quantization, and B0 score
            capture. Kept out of state_dict by ComplexMultiheadAttention.
        mha_projection: optional low-power grouped QKV/O projection settings.
    """
    B, T, E = x.shape
    H = mha.num_heads
    D = E // H
    orig_dtype = x.dtype

    if activation_quantizer is not None:
        x_q = activation_quantizer(x)
    else:
        x_q = x

    mha_projection = mha_projection or {}
    qkv_groups = int(mha_projection.get("qkv_groups", 1) or 1)
    out_proj_groups = int(mha_projection.get("out_proj_groups", 1) or 1)
    in_proj_weight = mha.in_proj_weight
    in_proj_mask = _mha_in_proj_group_mask(in_proj_weight, E, qkv_groups)
    if in_proj_mask is not None:
        in_proj_weight = in_proj_weight * in_proj_mask
    qkv = F.linear(x_q, in_proj_weight, mha.in_proj_bias)
    q, k, v = qkv.chunk(3, dim=-1)

    # (B, T, E) → (B, H, T, D)
    q = q.reshape(B, T, H, D).transpose(1, 2)
    k = k.reshape(B, T, H, D).transpose(1, 2)
    v = v.reshape(B, T, H, D).transpose(1, 2)

    if activation_quantizer is not None:
        q = activation_quantizer(q)
        k = activation_quantizer(k)
        v = activation_quantizer(v)

    if fp16_accumulator:
        q = q.to(torch.float16)
        k = k.to(torch.float16)
        v = v.to(torch.float16)

    lmul_bits = int(_attention_datapath_value(attention_datapath, "lmul_mantissa_bits", 4))
    lmul_bit_exact = bool(_attention_datapath_value(attention_datapath, "lmul_bit_exact", False))

    # QK^T / sqrt(D). Hardware datapath: fp16 accumulator when enabled.
    if _lmul_target_enabled(attention_datapath, "qk"):
        k_t = k.transpose(-2, -1)
        qk_force_dense = bool(_attention_datapath_value(
            attention_datapath, "lmul_qk_force_dense", False
        ))
        qk_window_width = (
            None if qk_force_dense
            else _infer_causal_window_from_mask(attn_mask, T, T)
        )
        if qk_window_width is not None:
            qk_chunk_size = int(_attention_datapath_value(
                attention_datapath, "lmul_qk_chunk_size", 32
            ))
            qk_scores = _lmul_windowed_qk_matmul(
                q, k_t, qk_window_width,
                mantissa_bits=lmul_bits,
                bit_exact=lmul_bit_exact,
                chunk_size=qk_chunk_size,
            )
        else:
            qk_scores = _lmul_matmul(
                q, k_t,
                mantissa_bits=lmul_bits,
                bit_exact=lmul_bit_exact,
            )
    else:
        qk_scores = torch.matmul(q, k.transpose(-2, -1))
    attn = _apply_qk_scale(qk_scores, D, attention_datapath)

    mask_for_capture = None
    if attn_mask is not None:
        mask = attn_mask.to(attn.dtype) if fp16_accumulator else attn_mask
        mask_for_capture = mask
        if mask.dim() == 2:
            attn = attn + mask  # broadcasts over (B, H)
        else:
            attn = attn + mask

    _capture_attention_scores(attention_datapath, attn, mask_for_capture)
    attn = _quantize_attention_scores(attn, attention_datapath)

    softmax_mode = _canonical_softmax_mode(softmax_mode)
    top_k_applied = False
    shift_topk_cfg = _shift_topk_config(softmax_mode)
    if shift_topk_cfg is not None:
        valid_mask = torch.isfinite(attn)
        attn = _shift_topk_softmax(
            attn,
            valid_mask,
            top_k=top_k,
            **shift_topk_cfg,
        )
        top_k_applied = top_k is not None and top_k > 0
    else:
        attn = _approx_attention_probs(attn, softmax_mode)

    if top_k is not None and top_k < T and not top_k_applied:
        # Top-k hard masking. scatter_ has a well-defined gradient, STE-like
        # behavior emerges naturally: gradients flow through the kept entries.
        topk_val, topk_idx = torch.topk(attn, k=top_k, dim=-1)
        mask = torch.zeros_like(attn)
        mask.scatter_(-1, topk_idx, 1.0)
        attn = attn * mask
        attn = attn / (attn.sum(dim=-1, keepdim=True) + 1e-8)

    attn = _quantize_attention_probs(attn, attention_datapath)
    packet_cfg = _prob_packet_config(attention_datapath)
    packet = None
    if packet_cfg is not None:
        packet = _packetize_q08_attention_probs(attn, packet_cfg)
        attn = packet["probs"]

    # Score × V
    score_v_mode = (
        str(packet_cfg.get("score_v_mode", "dequant_fp16")).lower()
        if packet_cfg is not None else "dequant_fp16"
    )
    if packet is not None and score_v_mode == "fixed_shift_add":
        out = _q08_packet_fixed_matmul(packet, v, packet_cfg)
    elif packet is not None and score_v_mode == "lmul":
        out = _q08_packet_lmul_matmul(
            packet, v,
            mantissa_bits=lmul_bits,
            bit_exact=lmul_bit_exact,
        )
    elif packet is not None:
        packet_dtype = torch.float16 if fp16_accumulator else v.dtype
        out = _q08_packet_dequant_matmul(packet, v, packet_dtype)
    elif _lmul_target_enabled(attention_datapath, "score_v"):
        out = _lmul_matmul(
            attn, v,
            mantissa_bits=lmul_bits,
            bit_exact=lmul_bit_exact,
        )
    else:
        out = torch.matmul(attn, v)  # (B, H, T, D)
    out = out.transpose(1, 2).contiguous().reshape(B, T, E)

    if fp16_accumulator:
        out = out.to(orig_dtype)

    out_proj_weight = mha.out_proj.weight
    out_proj_mask = _linear_group_mask(out_proj_weight, out_proj_groups)
    if out_proj_mask is not None:
        out_proj_weight = out_proj_weight * out_proj_mask
    out = F.linear(out, out_proj_weight, mha.out_proj.bias)
    return out


class ComplexMultiheadAttention(Module):
    def __init__(self, embed_dim=512, num_heads=8, batch_first=True, tied=False,
                 softmax_mode="softmax", mha_projection=None):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.tied = tied
        self.softmax_mode = _canonical_softmax_mode(softmax_mode)
        self.mha_projection = dict(mha_projection or {})
        self.projection_grouped_active = (
            int(self.mha_projection.get("qkv_groups", 1) or 1) > 1
            or int(self.mha_projection.get("out_proj_groups", 1) or 1) > 1
        )
        print(
            f"ComplexMultiheadAttention: embed_dim={embed_dim}, num_heads={num_heads}, "
            f"batch_first={batch_first}, tied={tied}, softmax_mode={self.softmax_mode}, "
            f"mha_projection={self.mha_projection}"
        )

        self.MultiheadAttention_re = MultiheadAttention(self.embed_dim, self.num_heads, batch_first=batch_first)
        if not tied:
            self.MultiheadAttention_im = MultiheadAttention(self.embed_dim, self.num_heads, batch_first=batch_first)

        # Mixed-precision QAT hooks (set externally by enable_mixed_precision_qat)
        self.top_k = None
        self.use_custom_forward = self.softmax_mode != "softmax" or self.projection_grouped_active
        self.activation_quantizer = None
        self.fp16_accumulator = False
        self.attention_datapath = {}


    def forward(self, x, attn_mask=None):
        if self.tied:
            return self._forward_tied(x, attn_mask)

        if self.use_custom_forward:
            real      = _mha_attention_forward(self.MultiheadAttention_re, x.real, attn_mask,
                                               self.top_k, self.activation_quantizer,
                                               self.fp16_accumulator,
                                               self.softmax_mode,
                                               self.attention_datapath,
                                               self.mha_projection) \
                      - _mha_attention_forward(self.MultiheadAttention_im, x.imag, attn_mask,
                                               self.top_k, self.activation_quantizer,
                                               self.fp16_accumulator,
                                               self.softmax_mode,
                                               self.attention_datapath,
                                               self.mha_projection)
            imaginary = _mha_attention_forward(self.MultiheadAttention_re, x.imag, attn_mask,
                                               self.top_k, self.activation_quantizer,
                                               self.fp16_accumulator,
                                               self.softmax_mode,
                                               self.attention_datapath,
                                               self.mha_projection) \
                      + _mha_attention_forward(self.MultiheadAttention_im, x.real, attn_mask,
                                               self.top_k, self.activation_quantizer,
                                               self.fp16_accumulator,
                                               self.softmax_mode,
                                               self.attention_datapath,
                                               self.mha_projection)
            return torch.complex(real, imaginary)

        real      = self._forward_real(x, attn_mask)
        imaginary = self._forward_imaginary(x, attn_mask)
        output = torch.complex(real, imaginary)

        return output

    def _forward_tied(self, x, attn_mask=None):
        # Weight-tied: MHA_re ≡ MHA_im = M.
        # ComplexMHA(x) = (1+i) · [M(x.real) + i·M(x.imag)]  =  (a − b) + i(a + b)
        # 2 real-MHA passes instead of 4. See HANDOFF_2026-05-09 §3.3.
        if self.use_custom_forward:
            a = _mha_attention_forward(self.MultiheadAttention_re, x.real, attn_mask,
                                       self.top_k, self.activation_quantizer,
                                       self.fp16_accumulator,
                                       self.softmax_mode,
                                       self.attention_datapath,
                                       self.mha_projection)
            b = _mha_attention_forward(self.MultiheadAttention_re, x.imag, attn_mask,
                                       self.top_k, self.activation_quantizer,
                                       self.fp16_accumulator,
                                       self.softmax_mode,
                                       self.attention_datapath,
                                       self.mha_projection)
        else:
            a, _ = self.MultiheadAttention_re(x.real, x.real, x.real, attn_mask=attn_mask)
            b, _ = self.MultiheadAttention_re(x.imag, x.imag, x.imag, attn_mask=attn_mask)
        return torch.complex(a - b, a + b)

    def _forward_real(self, x, attn_mask=None):
        real_real,_ = self.MultiheadAttention_re(
            x.real, x.real, x.real, attn_mask=attn_mask
        )
        imag_imag,_ = self.MultiheadAttention_im(
            x.imag, x.imag, x.imag, attn_mask=attn_mask
        )
        real = real_real - imag_imag
        return real

    def _forward_imaginary(self, x, attn_mask=None):
        imag_real,_ = self.MultiheadAttention_re(
            x.imag, x.imag, x.imag, attn_mask=attn_mask
        )
        real_imag,_ = self.MultiheadAttention_im(
            x.real, x.real, x.real, attn_mask=attn_mask
        )
        imaginary = imag_real + real_imag

        return imaginary
    
# Add these classes to complexLayers.py

class ComplexDepthwiseSeparableConv2d(Module):
    """
    Complex Depthwise Separable Convolution = Depthwise Conv + Pointwise Conv
    """
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=0,
                 dilation=1, bias=True):
        super().__init__()
        
        # Depthwise convolution (groups = in_channels)
        self.depthwise_r = Conv2d(in_channels, in_channels, kernel_size, stride, padding, 
                                  dilation, groups=in_channels, bias=False)
        self.depthwise_i = Conv2d(in_channels, in_channels, kernel_size, stride, padding, 
                                  dilation, groups=in_channels, bias=False)
        
        # Pointwise convolution (1x1 conv)
        self.pointwise_r = Conv2d(in_channels, out_channels, kernel_size=1, bias=bias)
        self.pointwise_i = Conv2d(in_channels, out_channels, kernel_size=1, bias=bias)
    
    def forward(self, input):
        # Depthwise convolution
        depth_out = apply_complex(self.depthwise_r, self.depthwise_i, input)
        
        # Pointwise convolution
        out = apply_complex(self.pointwise_r, self.pointwise_i, depth_out)
        
        return out


class ComplexDepthwiseSeparableConvTranspose2d(Module):
    """
    Complex Depthwise Separable Transposed Convolution
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0,
                 output_padding=0, dilation=1, bias=True):
        super().__init__()
        
        # Pointwise convolution first (1x1 conv) to change channels
        self.pointwise_r = Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        self.pointwise_i = Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        
        # Then depthwise transposed convolution
        self.depthwise_r = ConvTranspose2d(out_channels, out_channels, kernel_size, stride, 
                                           padding, output_padding, groups=out_channels, 
                                           bias=bias, dilation=dilation)
        self.depthwise_i = ConvTranspose2d(out_channels, out_channels, kernel_size, stride, 
                                           padding, output_padding, groups=out_channels, 
                                           bias=bias, dilation=dilation)
    
    def forward(self, input):
        # Pointwise convolution to change channels
        point_out = apply_complex(self.pointwise_r, self.pointwise_i, input)
        
        # Depthwise transposed convolution
        out = apply_complex(self.depthwise_r, self.depthwise_i, point_out)
        
        return out


class ComplexDepthwiseSeparableConvTranspose2dSplit(Module):
    """
    Split decoder pointwise into x and skip branches.

    With groups_x = groups_skip = 1 and weights initialized by slicing the
    legacy pointwise weight along its input-channel dimension, this is
    equivalent to applying ComplexDepthwiseSeparableConvTranspose2d to
    cat([x, skip], dim=1). Grouped branches are the P1 experiment.
    """
    def __init__(self, in_x_channels, in_skip_channels, out_channels,
                 kernel_size, stride=1, padding=0, output_padding=0,
                 dilation=1, bias=True, groups_x=1, groups_skip=1):
        super().__init__()

        if in_x_channels % groups_x != 0 or out_channels % groups_x != 0:
            raise ValueError(
                f"invalid groups_x={groups_x} for in_x={in_x_channels}, out={out_channels}"
            )
        if in_skip_channels % groups_skip != 0 or out_channels % groups_skip != 0:
            raise ValueError(
                f"invalid groups_skip={groups_skip} for in_skip={in_skip_channels}, out={out_channels}"
            )

        self.pointwise_r_x = Conv2d(
            in_x_channels, out_channels, kernel_size=1, bias=False, groups=groups_x
        )
        self.pointwise_i_x = Conv2d(
            in_x_channels, out_channels, kernel_size=1, bias=False, groups=groups_x
        )
        self.pointwise_r_skip = Conv2d(
            in_skip_channels, out_channels, kernel_size=1, bias=False, groups=groups_skip
        )
        self.pointwise_i_skip = Conv2d(
            in_skip_channels, out_channels, kernel_size=1, bias=False, groups=groups_skip
        )

        self.depthwise_r = ConvTranspose2d(
            out_channels, out_channels, kernel_size, stride,
            padding, output_padding, groups=out_channels,
            bias=bias, dilation=dilation
        )
        self.depthwise_i = ConvTranspose2d(
            out_channels, out_channels, kernel_size, stride,
            padding, output_padding, groups=out_channels,
            bias=bias, dilation=dilation
        )

    def forward(self, x, skip):
        pw_x = apply_complex(self.pointwise_r_x, self.pointwise_i_x, x)
        pw_skip = apply_complex(self.pointwise_r_skip, self.pointwise_i_skip, skip)
        point_out = pw_x + pw_skip
        return apply_complex(self.depthwise_r, self.depthwise_i, point_out)
