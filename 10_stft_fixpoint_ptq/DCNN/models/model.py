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

        fixed_point_cfg = kwargs.get("stft_fixed_point", None)
        float_precision_cfg = kwargs.get("stft_float_precision", None)
        self.stft = Stft(self.fft_len, self.win_inc, self.win_len,
                         fixed_point=fixed_point_cfg,
                         float_precision=float_precision_cfg)
        self.istft = IStft(self.fft_len, self.win_inc, self.win_len,
                           fixed_point=fixed_point_cfg,
                           float_precision=float_precision_cfg)
        if fixed_point_cfg and fixed_point_cfg.get("enabled", False):
            print(f"STFT/iSTFT fixed-point PTQ enabled: {fixed_point_cfg}")
        if float_precision_cfg and float_precision_cfg.get("enabled", False):
            print(f"STFT/iSTFT float-precision PTQ enabled: {float_precision_cfg}")
        

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
        self.mattn = MultiAttnBlock(
            input_size=flatten_size,  # 512 or 1024
            hidden_size=self.rnn_units,
            embed_dim=self.embed_dim, # 512  (跟論文一樣比 flatten_size 小沒問題)
            num_heads=self.num_heads,
            batch_first=True,
            window_size=self.attn_win,  # None or int
            tied_mha=self.tied_mha,
        )

        # self.encoder = Encoder(self.kernel_num, kernel_size)
        # self.decoder = Decoder(self.kernel_num, self.kernel_size)
        self.encoder = Encoder(self.kernel_num, kernel_size, use_depthwise=self.use_depthwise)  
        self.decoder = Decoder(self.kernel_num, self.kernel_size, use_depthwise=self.use_depthwise)
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

        self.kernel_num = kernel_num
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


class Decoder(nn.Module):
    def __init__(self, kernel_num, kernel_size, use_depthwise=True):
        super().__init__()

        self.kernel_num = kernel_num
        self.kernel_size = kernel_size
        self.use_depthwise = use_depthwise

        self.model = nn.ModuleList()
        for idx in range(len(self.kernel_num) - 1, 0, -1):
            if self.use_depthwise:
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
            
            self.model.append(nn.Sequential(*block))

    def forward(self, x, encoder_out):
        for idx in range(len(self.model)):
            x = torch.cat([x, encoder_out[-1 - idx]], 1)
            x = self.model[idx](x)
        return x



class MultiAttnBlock(nn.Module):
    def __init__(self, input_size, hidden_size, embed_dim=512, num_heads=32, batch_first=True, window_size=None, tied_mha=False):
        super().__init__()
        # Project to embed_dim first (exactly as in paper)
        self.input_proj = nn.Linear(input_size, embed_dim, dtype=torch.complex64)

        self.window_size = window_size

        print(f"MultiAttnBlock: input_size={input_size}, hidden_size={hidden_size}, embed_dim={embed_dim}, num_heads={num_heads}, window_size={window_size}, tied_mha={tied_mha}")
        # Use the right number of heads as in paper
        self.mattn = torch_complex.ComplexMultiheadAttention(
            embed_dim=embed_dim, num_heads=num_heads, batch_first=batch_first, tied=tied_mha)
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
