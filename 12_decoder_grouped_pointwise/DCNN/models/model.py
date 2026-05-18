import math

import torch
import torch.nn as nn
from DCNN.feature_extractors import IStft, Stft

import DCNN.utils.complexPyTorch.complexLayers as torch_complex
from DCNN.utils.show import show_params, show_model

from DCNN.utils.apply_mask import apply_mask



class DCNN(nn.Module):
    def __init__(
            self,
            rnn_layers=2, rnn_units=128,
            win_len=128, win_inc=80, fft_len=128, win_type='hann',
            masking_mode='E', use_clstm=False,
            kernel_size=5, 
            kernel_num=[16, 32, 64, 128, 256, 256], 
            # kernel_num = [ 8, 16, 32, 64, 128, 128],
            bidirectional=False, embed_dim=512, num_heads=32, use_depthwise=True, attn_win=None, **kwargs
    ):
        ''' 
            rnn_layers: the number of lstm layers in the crn,
            rnn_units: for clstm, rnn_units = real+imag
        '''

        super().__init__()

        # for fft
        self.win_len = win_len
        self.win_inc = win_inc  # TODO: Rename to hop_size
        self.fft_len = fft_len
        print(f"win_len: {self.win_len}, win_inc: {self.win_inc}, fft_len: {self.fft_len}")

        self.rnn_units = rnn_units
        self.hidden_layers = rnn_layers
        self.kernel_size = kernel_size
        # self.kernel_num = [2, 8, 16, 32, 128, 128, 128]
        # self.kernel_num = [2, 16, 32, 64, 128, 256, 256]
        self.kernel_num = [2] + kernel_num

        print(f"kernel_num: {self.kernel_num}, kernel_size: {self.kernel_size}")
        self.masking_mode = masking_mode
        self.use_clstm = use_clstm

        self.stft = Stft(self.fft_len, self.win_inc, self.win_len)
        self.istft = IStft(self.fft_len, self.win_inc, self.win_len)
        

        self.num_heads = num_heads
        self.embed_dim = embed_dim

        self.attn_win = attn_win

        print(f"num_heads: {self.num_heads}, embed_dim: {self.embed_dim}")

        self.use_depthwise = use_depthwise
        # hidden_dim = self.fft_len // (2 ** (len(self.kernel_num) + 1))
        # self.mattn = MultiAttnBlock(input_size=1024,
        #                             hidden_size=self.rnn_units,
        #                             embed_dim=self.embed_dim,
        #                             num_heads=self.num_heads,
        #                             batch_first=True)

        base_flat = (self.kernel_num[-1] // 2) * (self.fft_len // 2**(len(self.kernel_num)))
        # 若 config 有 binaural=True → 乘 2
        self.is_binaural = kwargs.get("binaural", False)
        flatten_size = base_flat * (2 if self.is_binaural else 1)
        # breakpoint()
        # print(f"flatten_size: {flatten_size}")
        self.tied_mha = kwargs.get("tied_mha", False)
        self.attention_softmax = kwargs.get("attention_softmax", {})
        self.mattn = MultiAttnBlock(
            input_size=flatten_size,  # 512 or 1024
            hidden_size=self.rnn_units,
            embed_dim=self.embed_dim, # 512  (跟論文一樣比 flatten_size 小沒問題)
            num_heads=self.num_heads,
            batch_first=True,
            window_size=self.attn_win,  # None or int
            tied_mha=self.tied_mha,
            attention_softmax=self.attention_softmax,
        )

        # self.encoder = Encoder(self.kernel_num, kernel_size)
        # self.decoder = Decoder(self.kernel_num, self.kernel_size)
        self.encoder = Encoder(self.kernel_num, kernel_size, use_depthwise=self.use_depthwise)  
        self.decoder_grouped_pointwise = kwargs.get("decoder_grouped_pointwise", {})
        self.decoder = Decoder(
            self.kernel_num,
            self.kernel_size,
            use_depthwise=self.use_depthwise,
            decoder_grouped_pointwise=self.decoder_grouped_pointwise,
        )
        print(f"depthwise: {self.use_depthwise}")
        # show_model(self)
        # show_params(self)
        # self._flatten_parameters()

    def forward(self, inputs):

        # 0. Extract STFT
        x = cspecs = self.stft(inputs)
        x = x.unsqueeze(1)  # Add a dummy channel

        # x=self.attn(x)

        encoder_out = self.encoder(x)
        x = encoder_out[-1]
        x = self.mattn(x)
        x = self.decoder(x, encoder_out)
        out_spec = apply_mask(x[:, 0], cspecs, self.masking_mode)
        out_wav = self.istft(out_spec)
        return out_wav  # out_spec, out_wav

    def get_params(self, weight_decay=0.0):
        # add L2 penalty
        weights, biases = [], []
        for name, param in self.named_parameters():
            if 'bias' in name:
                biases += [param]
            else:
                weights += [param]
        params = [{
            'params': weights,
            'weight_decay': weight_decay,
        }, {
            'params': biases,
            'weight_decay': 0.0,
        }]
        return params

class Encoder(nn.Module):
    def __init__(self, kernel_num, kernel_size, use_depthwise=True):
        super().__init__()

        self.kernel_num = list(kernel_num)
        if self.kernel_num[0] != 2:
            self.kernel_num = [2] + self.kernel_num
        self.kernel_size = kernel_size
        self.use_depthwise = use_depthwise

        self.model = nn.ModuleList()
        for idx in range(len(self.kernel_num) - 1):
            if self.use_depthwise:
                # Use depthwise separable convolution
                conv_layer = torch_complex.ComplexDepthwiseSeparableConv2d(
                    self.kernel_num[idx]//2,
                    self.kernel_num[idx + 1]//2,
                    # kernel_size=(self.kernel_size, 2),
                    # stride=(2, 1),
                    # padding=(2, 1)
                    kernel_size=(self.kernel_size, 1),
                    stride=(2, 1),
                    padding=(2, 0)
                )
            else:
                # Use regular convolution (existing code)
                conv_layer = torch_complex.ComplexConv2d(
                    self.kernel_num[idx]//2,
                    self.kernel_num[idx + 1]//2,
                    kernel_size=(self.kernel_size, 2),
                    stride=(2, 1),
                    padding=(2, 1)
                )
            
            self.model.append(
                nn.Sequential(
                    conv_layer,
                    torch_complex.NaiveComplexBatchNorm2d(
                        self.kernel_num[idx + 1]//2),
                    torch_complex.ComplexPReLU()
                )
            )

    def forward(self, x):
        output = []
        for idx, layer in enumerate(self.model):
            x = layer(x)
            output.append(x)
        return output


def _effective_groups(requested_groups, in_channels, out_channels):
    """Largest valid group count <= requested_groups for this 1x1 branch."""
    requested_groups = max(1, int(requested_groups))
    max_groups = math.gcd(in_channels, out_channels)
    for groups in range(min(requested_groups, max_groups), 0, -1):
        if max_groups % groups == 0:
            return groups
    return 1


def _skip_keep_count(full_channels, ratio, groups):
    """Group-aware skip-channel keep count.

    Keeping the same number of channels per group preserves the grouped
    pointwise branch layout. This is important for warm-starting from P1 g42.
    """
    if ratio >= 1.0:
        return full_channels
    if not (0.0 < ratio <= 1.0):
        raise ValueError(f"skip_select_ratio must be in (0, 1], got {ratio}")
    if full_channels % groups != 0:
        raise ValueError(f"groups={groups} does not divide skip channels={full_channels}")
    group_size = full_channels // groups
    keep_per_group = max(1, min(group_size, int(round(group_size * ratio))))
    return keep_per_group * groups


def _first_n_skip_indices(full_channels, ratio, groups):
    keep = _skip_keep_count(full_channels, ratio, groups)
    if keep == full_channels:
        return torch.arange(full_channels, dtype=torch.long)
    group_size = full_channels // groups
    keep_per_group = keep // groups
    pieces = []
    for group_idx in range(groups):
        start = group_idx * group_size
        pieces.append(torch.arange(start, start + keep_per_group, dtype=torch.long))
    return torch.cat(pieces, dim=0)


def _stage_indices_from_payload(payload, stage):
    if isinstance(payload, (list, tuple)):
        return payload[stage]
    if isinstance(payload, dict):
        if "indices" in payload:
            return _stage_indices_from_payload(payload["indices"], stage)
        for key in (stage, str(stage), f"stage_{stage}", f"decoder_stage_{stage}"):
            if key in payload:
                return payload[key]
    raise KeyError(f"could not find skip indices for decoder stage {stage}")


def _validate_skip_indices(indices, full_channels, ratio, groups, stage):
    indices = torch.as_tensor(indices, dtype=torch.long).flatten()
    keep = _skip_keep_count(full_channels, ratio, groups)
    if indices.numel() != keep:
        raise ValueError(
            f"decoder stage {stage}: got {indices.numel()} skip indices, "
            f"expected {keep} for full_channels={full_channels}, ratio={ratio}, groups={groups}"
        )
    if indices.numel() == 0:
        raise ValueError(f"decoder stage {stage}: empty skip index set")
    if indices.min().item() < 0 or indices.max().item() >= full_channels:
        raise ValueError(
            f"decoder stage {stage}: skip indices out of range 0..{full_channels - 1}"
        )
    if indices.unique().numel() != indices.numel():
        raise ValueError(f"decoder stage {stage}: duplicate skip indices")

    if groups > 1:
        group_size = full_channels // groups
        keep_per_group = keep // groups
        for group_idx in range(groups):
            chunk = indices[group_idx * keep_per_group:(group_idx + 1) * keep_per_group]
            lo = group_idx * group_size
            hi = lo + group_size
            if not torch.all((chunk >= lo) & (chunk < hi)):
                raise ValueError(
                    f"decoder stage {stage}: skip indices must be stored group-wise; "
                    f"group {group_idx} expected [{lo}, {hi})"
                )
    return indices


class _DecoderBlockSplit(nn.Module):
    """Decoder block wrapper with stable numeric state-dict keys.

    The split conv is registered as submodule "0" so depthwise / BN / PReLU
    keys stay aligned with the legacy Sequential block wherever possible.
    """
    def __init__(self, split_conv, bn=None, prelu=None):
        super().__init__()
        self.add_module("0", split_conv)
        if bn is not None:
            self.add_module("1", bn)
        if prelu is not None:
            self.add_module("2", prelu)

    @property
    def split_conv(self):
        return self._modules["0"]

    def forward(self, x, skip):
        x = self._modules["0"](x, skip)
        if "1" in self._modules:
            x = self._modules["1"](x)
        if "2" in self._modules:
            x = self._modules["2"](x)
        return x


class Decoder(nn.Module):
    def __init__(self, kernel_num, kernel_size, use_depthwise=True,
                 decoder_grouped_pointwise=None):
        super().__init__()

        self.kernel_num = list(kernel_num)
        if self.kernel_num[0] != 2:
            self.kernel_num = [2] + self.kernel_num
        self.kernel_size = kernel_size
        self.use_depthwise = use_depthwise
        cfg = decoder_grouped_pointwise or {}
        self.grouped_pw_mode = cfg.get("mode", "off")
        if self.grouped_pw_mode not in ("off", "split", "grouped"):
            raise ValueError(f"unknown decoder_grouped_pointwise.mode={self.grouped_pw_mode}")
        self.groups_x = int(cfg.get("groups_x", 1))
        self.groups_skip = int(cfg.get("groups_skip", 1))
        self.channel_shuffle = bool(cfg.get("channel_shuffle", False))
        self.skip_select_ratio = float(cfg.get("skip_select_ratio", 1.0))
        self.skip_select_mode = cfg.get("skip_select_mode", "topk_l1")
        self.skip_indices_path = cfg.get("skip_indices_path", None)
        if self.skip_indices_path in ("", "None", "none", "null"):
            self.skip_indices_path = None

        if self.channel_shuffle:
            raise NotImplementedError("decoder channel_shuffle is deferred by the spec")
        if not (0.0 < self.skip_select_ratio <= 1.0):
            raise ValueError(f"skip_select_ratio must be in (0, 1], got {self.skip_select_ratio}")
        self.skip_select_active = self.skip_select_ratio < 1.0
        if self.skip_select_active and self.grouped_pw_mode == "off":
            raise ValueError("skip_select_ratio < 1.0 requires split/grouped decoder pointwise")
        if self.skip_select_mode not in ("topk_l1", "first_n"):
            raise ValueError(f"unknown skip_select_mode={self.skip_select_mode}")
        if self.skip_select_active and self.skip_select_mode == "topk_l1" and self.skip_indices_path is None:
            raise ValueError("topk_l1 skip selection requires decoder_grouped_pointwise.skip_indices_path")
        if self.grouped_pw_mode != "off" and not self.use_depthwise:
            raise ValueError("decoder_grouped_pointwise requires depthwise-separable decoder blocks")

        self._skip_index_payload = None
        if self.skip_indices_path is not None:
            self._skip_index_payload = torch.load(self.skip_indices_path, map_location="cpu")

        self.model = nn.ModuleList()
        for idx in range(len(self.kernel_num) - 1, 0, -1):
            stage = len(self.model)
            if self.use_depthwise and self.grouped_pw_mode == "off":
                # Use depthwise separable transposed convolution
                conv_layer = torch_complex.ComplexDepthwiseSeparableConvTranspose2d(
                    self.kernel_num[idx],
                    self.kernel_num[idx - 1]//2,
                    # kernel_size=(self.kernel_size, 2),
                    # stride=(2, 1),
                    # padding=(2, 1),
                    kernel_size=(self.kernel_size, 1),
                    stride=(2, 1),
                    padding=(2, 0),
                    output_padding=(1, 0)
                )
            elif self.use_depthwise:
                in_x_channels = self.kernel_num[idx] // 2
                full_skip_channels = self.kernel_num[idx] // 2
                out_channels = self.kernel_num[idx - 1] // 2
                if self.grouped_pw_mode == "split":
                    groups_x_eff = 1
                    select_groups = 1
                else:
                    groups_x_eff = _effective_groups(self.groups_x, in_x_channels, out_channels)
                    select_groups = _effective_groups(
                        self.groups_skip, full_skip_channels, out_channels
                    )

                if self.skip_select_active:
                    if self._skip_index_payload is not None:
                        skip_indices = _stage_indices_from_payload(self._skip_index_payload, stage)
                    else:
                        skip_indices = _first_n_skip_indices(
                            full_skip_channels, self.skip_select_ratio, select_groups
                        )
                    skip_indices = _validate_skip_indices(
                        skip_indices, full_skip_channels, self.skip_select_ratio,
                        select_groups, stage
                    )
                    self.register_buffer(
                        f"skip_indices_{stage}", skip_indices, persistent=True
                    )
                    in_skip_channels = int(skip_indices.numel())
                else:
                    in_skip_channels = full_skip_channels

                if self.grouped_pw_mode == "split":
                    groups_skip_eff = 1
                else:
                    groups_skip_eff = _effective_groups(
                        self.groups_skip, in_skip_channels, out_channels
                    )
                print(
                    "decoder_grouped_pointwise "
                    f"layer={stage} idx={idx} mode={self.grouped_pw_mode} "
                    f"in_x={in_x_channels} in_skip={in_skip_channels} out={out_channels} "
                    f"groups_x={groups_x_eff} groups_skip={groups_skip_eff} "
                    f"skip_select_ratio={self.skip_select_ratio}"
                )
                conv_layer = torch_complex.ComplexDepthwiseSeparableConvTranspose2dSplit(
                    in_x_channels,
                    in_skip_channels,
                    out_channels,
                    kernel_size=(self.kernel_size, 1),
                    stride=(2, 1),
                    padding=(2, 0),
                    output_padding=(1, 0),
                    groups_x=groups_x_eff,
                    groups_skip=groups_skip_eff,
                )
            else:
                # Use regular transposed convolution (existing code)
                conv_layer = torch_complex.ComplexConvTranspose2d(
                    self.kernel_num[idx],
                    self.kernel_num[idx - 1]//2,
                    # kernel_size=(self.kernel_size, 2),
                    # stride=(2, 1),
                    # padding=(2, 1),
                    kernel_size=(self.kernel_size, 1),
                    stride=(2, 1),
                    padding=(2, 0),
                    output_padding=(1, 0)
                )
            
            block = [conv_layer]
            
            if idx != 1:
                block.append(torch_complex.NaiveComplexBatchNorm2d(
                    self.kernel_num[idx - 1]//2))
                block.append(torch_complex.ComplexPReLU())
            
            if self.grouped_pw_mode == "off":
                self.model.append(nn.Sequential(*block))
            else:
                bn = block[1] if len(block) > 1 else None
                prelu = block[2] if len(block) > 2 else None
                self.model.append(_DecoderBlockSplit(conv_layer, bn=bn, prelu=prelu))

    def forward(self, x, encoder_out):
        for idx in range(len(self.model)):
            skip = encoder_out[-1 - idx]
            if self.skip_select_active:
                skip_indices = getattr(self, f"skip_indices_{idx}")
                skip = skip.index_select(1, skip_indices)
            if self.grouped_pw_mode == "off":
                x = torch.cat([x, skip], 1)
                x = self.model[idx](x)
            else:
                x = self.model[idx](x, skip)
        return x



class MultiAttnBlock(nn.Module):
    def __init__(self, input_size, hidden_size, embed_dim=512, num_heads=32,
                 batch_first=True, window_size=None, tied_mha=False,
                 attention_softmax=None):
        super().__init__()
        # Project to embed_dim first (exactly as in paper)
        self.input_proj = nn.Linear(input_size, embed_dim, dtype=torch.complex64)

        self.window_size = window_size
        if isinstance(attention_softmax, dict):
            self.softmax_mode = attention_softmax.get("mode", "softmax")
        else:
            self.softmax_mode = attention_softmax or "softmax"

        print(f"MultiAttnBlock: input_size={input_size}, hidden_size={hidden_size}, embed_dim={embed_dim}, num_heads={num_heads}, window_size={window_size}, tied_mha={tied_mha}, softmax_mode={self.softmax_mode}")
        # Use the right number of heads as in paper
        self.mattn = torch_complex.ComplexMultiheadAttention(
            embed_dim=embed_dim, num_heads=num_heads, batch_first=batch_first,
            tied=tied_mha, softmax_mode=self.softmax_mode)
        # Project back to original dimension
        self.output_proj = nn.Linear(embed_dim, input_size, dtype=torch.complex64)

        # print(f"MultiAttnBlock: input_size={input_size}, hidden_size={hidden_size}, embed_dim={embed_dim}, num_heads={num_heads}")
    def forward(self, x):
        batch_size, channels, freqs, time_bins = x.shape       
        x = x.flatten(start_dim=1, end_dim=2)
        x = x.transpose(1, 2)
        x = self.input_proj(x)
        # x = self.mattn(x)
        # ---------- causal mask ----------
        T = x.size(1)
        if self.window_size is None:
            # 舊版：完整因果遮罩
            attn_mask = torch.triu(
                torch.full((T, T), float("-inf"), device=x.device), diagonal=1
            )
        else:
            # 新版：只保留 [i-window_size, i] 的 key/value
            mask = torch.triu(
                torch.full((T, T), float("-inf"), device=x.device), diagonal=1
            )
            idx  = torch.arange(T, device=x.device)
            dist = idx.unsqueeze(1) - idx.unsqueeze(0)   
            mask[dist > self.window_size] = float("-inf")
            attn_mask = mask
            # print(attn_mask)
    #         print("softmax 可見 key 數：",
    #   (self.mattn.MultiheadAttention_re(x.real, x.real, x.real,
    #                                     attn_mask=attn_mask)[1][0]
    #    .isfinite()).sum(dim=-1)[:10])
            
        # ---------- self attention ----------
        x = self.mattn(x, attn_mask=attn_mask)
        x = self.output_proj(x)
        x = x.unflatten(-1, (channels, freqs))
        x = x.movedim(1, -1)
        return x
