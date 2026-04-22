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


class ComplexPReLU(Module):
    def __init__(self):
        super().__init__()
        self.r_prelu = PReLU(dtype=torch.float32)        
        self.i_prelu = PReLU(dtype=torch.float32)


    def forward(self, input):
        result = self.r_prelu(input.real) + 1j*self.i_prelu(input.imag)
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


class NaiveComplexBatchNorm1d(Module):
    '''
    Naive approach to complex batch norm, perform batch norm independently on real and imaginary part.
    '''

    def __init__(self, num_features, eps=1e-5, momentum=0.1, affine=True,
                 track_running_stats=True):
        super().__init__()
        self.bn_r = BatchNorm1d(
            num_features, eps, momentum, affine, track_running_stats)
        self.bn_i = BatchNorm1d(
            num_features, eps, momentum, affine, track_running_stats)

    def forward(self, input):
        return self.bn_r(input.real).type(torch.complex64) + 1j*self.bn_i(input.imag).type(torch.complex64)


class NaiveComplexBatchNorm2d(Module):
    '''
    Naive approach to complex batch norm, perform batch norm independently on real and imaginary part.
    '''

    def __init__(self, num_features, eps=1e-5, momentum=0.1, affine=True,
                 track_running_stats=True):
        super().__init__()
        self.bn_r = BatchNorm2d(
            num_features, eps, momentum, affine, track_running_stats)
        self.bn_i = BatchNorm2d(
            num_features, eps, momentum, affine, track_running_stats)

    def forward(self, input):
        return self.bn_r(input.real).type(torch.complex64) + 1j*self.bn_i(input.imag).type(torch.complex64)


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

def _mha_attention_forward(mha, x, attn_mask, top_k, activation_quantizer=None,
                           fp16_accumulator=False):
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
    """
    B, T, E = x.shape
    H = mha.num_heads
    D = E // H
    orig_dtype = x.dtype

    if activation_quantizer is not None:
        x_q = activation_quantizer(x)
    else:
        x_q = x

    qkv = F.linear(x_q, mha.in_proj_weight, mha.in_proj_bias)
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

    # QK^T / sqrt(D). Hardware datapath: fp16 accumulator when enabled.
    attn = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(D)

    if attn_mask is not None:
        mask = attn_mask.to(attn.dtype) if fp16_accumulator else attn_mask
        if mask.dim() == 2:
            attn = attn + mask  # broadcasts over (B, H)
        else:
            attn = attn + mask

    attn = F.softmax(attn, dim=-1)

    if top_k is not None and top_k < T:
        # Top-k hard masking. scatter_ has a well-defined gradient, STE-like
        # behavior emerges naturally: gradients flow through the kept entries.
        topk_val, topk_idx = torch.topk(attn, k=top_k, dim=-1)
        mask = torch.zeros_like(attn)
        mask.scatter_(-1, topk_idx, 1.0)
        attn = attn * mask
        attn = attn / (attn.sum(dim=-1, keepdim=True) + 1e-8)

    # Score × V
    out = torch.matmul(attn, v)  # (B, H, T, D)
    out = out.transpose(1, 2).contiguous().reshape(B, T, E)

    if fp16_accumulator:
        out = out.to(orig_dtype)

    out = mha.out_proj(out)
    return out


class ComplexMultiheadAttention(Module):
    def __init__(self, embed_dim=512, num_heads=8, batch_first=True):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        print(f"ComplexMultiheadAttention: embed_dim={embed_dim}, num_heads={num_heads}, batch_first={batch_first}")

        self.MultiheadAttention_re = MultiheadAttention(self.embed_dim, self.num_heads, batch_first=batch_first)
        self.MultiheadAttention_im = MultiheadAttention(self.embed_dim, self.num_heads, batch_first=batch_first)

        # Mixed-precision QAT hooks (set externally by enable_mixed_precision_qat)
        self.top_k = None
        self.use_custom_forward = False
        self.activation_quantizer = None
        self.fp16_accumulator = False


    def forward(self, x, attn_mask=None):
        if self.use_custom_forward:
            real      = _mha_attention_forward(self.MultiheadAttention_re, x.real, attn_mask,
                                               self.top_k, self.activation_quantizer,
                                               self.fp16_accumulator) \
                      - _mha_attention_forward(self.MultiheadAttention_im, x.imag, attn_mask,
                                               self.top_k, self.activation_quantizer,
                                               self.fp16_accumulator)
            imaginary = _mha_attention_forward(self.MultiheadAttention_re, x.imag, attn_mask,
                                               self.top_k, self.activation_quantizer,
                                               self.fp16_accumulator) \
                      + _mha_attention_forward(self.MultiheadAttention_im, x.real, attn_mask,
                                               self.top_k, self.activation_quantizer,
                                               self.fp16_accumulator)
            return torch.complex(real, imaginary)

        real      = self._forward_real(x, attn_mask)
        imaginary = self._forward_imaginary(x, attn_mask)
        output = torch.complex(real, imaginary)

        return output

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