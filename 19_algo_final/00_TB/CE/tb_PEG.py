"""
test_peg_fp16.py - Conv Encoder Pattern Generator (IEEE FP16 格式)

根據實際架構:
1. FFT_LEN=128 → 64 complex freq bins (移除 DC)
2. Kernel=(5,1) 頻率維度卷積, stride=(2,1), padding=(2,0)
3. 左右耳共用同一組 Encoder/Decoder 權重
4. 6 層 encoder: 1→8→16→32→64→128→128 complex channels
5. apply_complex 運算規則
6. 生成 IEEE FP16 格式供 256 Real MACs 硬體驗證

FP16 格式說明:
  - Total: 16 bits
  - Sign: 1 bit
  - Exponent (WXP): 5 bits (bias=15)
  - Fraction (FRAC): 10 bits
  - Format: [sign(1)][exp(5)][frac(10)]
"""

import numpy as np
import struct
import os
from pathlib import Path
from typing import Tuple, Dict, List


class FP16Converter:
    """IEEE FP16 格式轉換器"""
    
    @staticmethod
    def float32_to_fp16_hex(value):
        """
        Float32 → IEEE FP16 hex
        FP16 格式: 1 sign + 5 exp + 10 frac = 16 bits
        """
        if np.isnan(value) or np.isinf(value):
            value = 0.0
        
        # 使用 numpy 的 FP16 轉換
        fp16_val = np.float16(value)
        fp16_bits = np.frombuffer(fp16_val.tobytes(), dtype=np.uint16)[0]
        
        return f"{fp16_bits:04x}"
    
    @staticmethod
    def fp16_hex_to_float32(hex_str):
        """IEEE FP16 hex → Float32"""
        fp16_bits = int(hex_str, 16)
        fp16_bytes = struct.pack('H', fp16_bits)
        fp16_val = np.frombuffer(fp16_bytes, dtype=np.float16)[0]
        return float(fp16_val)
    
    @staticmethod
    def get_fp16_info():
        """返回 FP16 格式信息"""
        return {
            'total_bits': 16,
            'sign_bits': 1,
            'exp_bits': 5,      # WXP
            'frac_bits': 10,    # FRAC
            'exp_bias': 15,
            'max_exp': 31,
            'format': '[sign(1)][exp(5)][frac(10)]'
        }


class ComplexDepthwiseSeparableConv1D:
    """
    Complex Depthwise Separable Conv1D (頻率維度)
    實現 PyTorch complexLayers.py 的 apply_complex 規則
    """
    
    def __init__(self, 
                 in_channels: int,
                 out_channels: int, 
                 kernel_size: int = 5,
                 stride: int = 2,
                 padding: int = 2):
        """
        Args:
            in_channels: 輸入複數 channels
            out_channels: 輸出複數 channels
            kernel_size: 卷積核大小 (頻率維度)
            stride: 步長 (頻率維度)
            padding: 填充 (頻率維度)
        """
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        
    def depthwise_conv1d(self, input_data, dw_weight):
        """
        Depthwise convolution (每個 channel 獨立)
        input_data: [C, F] complex
        dw_weight: [C, K] complex
        Returns: [C, F_out] complex
        """
        C, F = input_data.shape
        K = self.kernel_size
        
        # Padding
        if self.padding > 0:
            padded = np.pad(input_data, ((0,0), (self.padding, self.padding)), 
                          mode='constant', constant_values=0)
        else:
            padded = input_data
        
        F_out = (F + 2*self.padding - K) // self.stride + 1
        output = np.zeros((C, F_out), dtype=np.complex64)
        
        # 分離實虛部
        input_r = padded.real
        input_i = padded.imag
        weight_r = dw_weight.real
        weight_i = dw_weight.imag
        
        for c in range(C):
            for f_out in range(F_out):
                f_start = f_out * self.stride
                
                # 取 kernel 範圍
                patch_r = input_r[c, f_start:f_start+K]
                patch_i = input_i[c, f_start:f_start+K]
                
                # apply_complex: (Wr*Xr - Wi*Xi) + j(Wr*Xi + Wi*Xr)
                acc_r = np.sum(weight_r[c] * patch_r - weight_i[c] * patch_i)
                acc_i = np.sum(weight_r[c] * patch_i + weight_i[c] * patch_r)
                
                output[c, f_out] = acc_r + 1j * acc_i
        
        return output
    
    def pointwise_conv1d(self, input_data, pw_weight):
        """
        Pointwise convolution (1×1 跨 channel)
        input_data: [C_in, F] complex
        pw_weight: [C_out, C_in] complex
        Returns: [C_out, F] complex
        """
        C_in, F = input_data.shape
        C_out = self.out_channels
        
        output = np.zeros((C_out, F), dtype=np.complex64)
        
        input_r = input_data.real
        input_i = input_data.imag
        weight_r = pw_weight.real
        weight_i = pw_weight.imag
        
        for c_out in range(C_out):
            for f in range(F):
                acc_r = 0.0
                acc_i = 0.0
                for c_in in range(C_in):
                    # apply_complex
                    acc_r += weight_r[c_out, c_in] * input_r[c_in, f] - \
                             weight_i[c_out, c_in] * input_i[c_in, f]
                    acc_i += weight_r[c_out, c_in] * input_i[c_in, f] + \
                             weight_i[c_out, c_in] * input_r[c_in, f]
                
                output[c_out, f] = acc_r + 1j * acc_i
        
        return output
    
    def forward(self, input_data, dw_weight, pw_weight):
        """完整的 Depthwise Separable Conv"""
        dw_out = self.depthwise_conv1d(input_data, dw_weight)
        pw_out = self.pointwise_conv1d(dw_out, pw_weight)
        return dw_out, pw_out


class ConvEncoderPatternGenerator:
    """
    生成 Conv Encoder 測試 patterns
    
    架構: 左右耳共用權重
    - 單一組 Encoder/Decoder
    - 左右聲道在 STFT 後於 channel 維度處理
    """
    
    # 根據 config: kernel_num = [2, 16, 32, 64, 128, 256, 256]
    # 複數 channels (//2): [1, 8, 16, 32, 64, 128, 128]
    LAYER_CONFIGS = [
        {'in_ch': 1,   'out_ch': 8,   'freq_in': 64, 'name': 'Layer0_1to8'},
        {'in_ch': 8,   'out_ch': 16,  'freq_in': 32, 'name': 'Layer1_8to16'},
        {'in_ch': 16,  'out_ch': 32,  'freq_in': 16, 'name': 'Layer2_16to32'},
        {'in_ch': 32,  'out_ch': 64,  'freq_in': 8,  'name': 'Layer3_32to64'},
        {'in_ch': 64,  'out_ch': 128, 'freq_in': 4,  'name': 'Layer4_64to128'},
        {'in_ch': 128, 'out_ch': 128, 'freq_in': 2,  'name': 'Layer5_128to128'},
    ]
    
    def __init__(self, 
                 kernel_size: int = 5,
                 stride: int = 2,
                 padding: int = 2):
        """
        Args:
            kernel_size: 卷積核大小 (頻率維度)
            stride: 步長
            padding: 填充
        """
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.converter = FP16Converter()
        
        fp16_info = self.converter.get_fp16_info()
        
        print(f"\n{'='*80}")
        print(f"Binaural Conv Encoder Pattern Generator (IEEE FP16)")
        print(f"{'='*80}")
        print(f"數值格式:")
        print(f"  - Format: IEEE 754 Half Precision (FP16)")
        print(f"  - Total bits: {fp16_info['total_bits']}")
        print(f"  - WXP (Exponent): {fp16_info['exp_bits']} bits (bias={fp16_info['exp_bias']})")
        print(f"  - FRAC (Fraction): {fp16_info['frac_bits']} bits")
        print(f"  - Bit layout: {fp16_info['format']}")
        print(f"\n架構特性:")
        print(f"  - STFT: FFT_LEN=128 → 64 complex freq bins (DC removed)")
        print(f"  - 左右耳共用同一組 Encoder/Decoder 權重")
        print(f"  - Kernel: ({kernel_size}, 1), Stride: ({stride}, 1), Padding: ({padding}, 0)")
        print(f"  - 6 層 Depthwise Separable Conv")
        print(f"{'='*80}\n")
    
    def generate_random_complex(self, shape, data_range=(-1.0, 1.0)):
        """生成隨機複數數據"""
        real = np.random.uniform(data_range[0], data_range[1], shape).astype(np.float32)
        imag = np.random.uniform(data_range[0], data_range[1], shape).astype(np.float32)
        return real + 1j * imag
    
    def compute_layer_macs(self, in_ch, out_ch, freq, kernel_size):
        """計算單層運算量"""
        # Depthwise
        dw_macs = in_ch * kernel_size * freq
        # Pointwise  
        pw_macs = out_ch * in_ch * freq
        total_complex = dw_macs + pw_macs
        total_real = total_complex * 4  # apply_complex: 4 real MACs per complex MAC
        
        return {
            'dw_complex': dw_macs,
            'pw_complex': pw_macs,
            'total_complex': total_complex,
            'total_real': total_real,
            'cycles_256macs': total_real / 256
        }
    
    def complex_to_hex(self, complex_val):
        """複數 → 32-bit hex [imag_16][real_16]"""
        real_hex = self.converter.float32_to_fp16_hex(complex_val.real)
        imag_hex = self.converter.float32_to_fp16_hex(complex_val.imag)
        return f"{imag_hex}{real_hex}"
    
    def save_dat(self, data, filename, description=""):
        """保存為 .dat 格式 (純數據，無註解)"""
        with open(filename, 'w') as f:
            flat = data.flatten()
            for val in flat:
                f.write(f"{self.complex_to_hex(val)}\n")
    
    def generate_single_layer(self, layer_idx: int, output_dir: str):
        """生成單層測試 pattern"""
        config = self.LAYER_CONFIGS[layer_idx]
        in_ch = config['in_ch']
        out_ch = config['out_ch']
        freq_in = config['freq_in']
        layer_name = config['name']
        
        freq_out = (freq_in + 2*self.padding - self.kernel_size) // self.stride + 1
        
        # 計算運算量
        macs = self.compute_layer_macs(in_ch, out_ch, freq_out, self.kernel_size)
        
        print(f"\n{'='*80}")
        print(f"{layer_name}")
        print(f"{'='*80}")
        print(f"Input:  [{in_ch:3d} ch, {freq_in:3d} freq] (單聲道等效)")
        print(f"Output: [{out_ch:3d} ch, {freq_out:3d} freq]")
        print(f"\nMACs per time frame:")
        print(f"  Depthwise:  {macs['dw_complex']:>8,} complex = {macs['dw_complex']*4:>10,} real")
        print(f"  Pointwise:  {macs['pw_complex']:>8,} complex = {macs['pw_complex']*4:>10,} real")
        print(f"  Total:      {macs['total_complex']:>8,} complex = {macs['total_real']:>10,} real")
        print(f"\n256 Real MACs: ~{macs['cycles_256macs']:.0f} cycles per frame")
        print(f"\n註: 左右耳共用此組權重,實際處理時雙倍數據量")
        
        # 生成數據
        print(f"\n生成測試數據...")
        input_data = self.generate_random_complex((in_ch, freq_in), data_range=(-2.0, 2.0))
        dw_weight = self.generate_random_complex((in_ch, self.kernel_size), data_range=(-1.0, 1.0))
        pw_weight = self.generate_random_complex((out_ch, in_ch), data_range=(-1.0, 1.0))
        
        # 執行卷積
        print(f"計算 Golden output...")
        conv = ComplexDepthwiseSeparableConv1D(in_ch, out_ch, self.kernel_size, self.stride, self.padding)
        dw_output, final_output = conv.forward(input_data, dw_weight, pw_weight)
        
        # 保存
        layer_dir = f"{output_dir}/{layer_name}"
        Path(layer_dir).mkdir(parents=True, exist_ok=True)
        
        print(f"保存 patterns...")
        self.save_dat(input_data, f"{layer_dir}/input.dat")
        self.save_dat(dw_weight, f"{layer_dir}/dw_weight.dat")
        self.save_dat(pw_weight, f"{layer_dir}/pw_weight.dat")
        self.save_dat(dw_output, f"{layer_dir}/dw_output.dat")
        self.save_dat(final_output, f"{layer_dir}/final_output.dat")
        
        # Config
        fp16_info = self.converter.get_fp16_info()
        
        with open(f"{layer_dir}/config.txt", 'w') as f:
            f.write(f"="*60 + "\n")
            f.write(f"Layer Configuration: {layer_name}\n")
            f.write(f"="*60 + "\n\n")
            
            f.write(f"Number Format:\n")
            f.write(f"  Type: IEEE 754 Half Precision (FP16)\n")
            f.write(f"  Total bits: {fp16_info['total_bits']}\n")
            f.write(f"  WXP (Exponent): {fp16_info['exp_bits']} bits\n")
            f.write(f"  FRAC (Fraction): {fp16_info['frac_bits']} bits\n")
            f.write(f"  Exponent bias: {fp16_info['exp_bias']}\n")
            f.write(f"  Bit layout: {fp16_info['format']}\n\n")
            
            f.write(f"Architecture:\n")
            f.write(f"  Type: Shared Weights (左右耳共用)\n")
            f.write(f"  Description: 左右聲道使用相同的 Encoder/Decoder 權重\n\n")
            
            f.write(f"Layer Dimensions:\n")
            f.write(f"  in_channels:  {in_ch} (complex)\n")
            f.write(f"  out_channels: {out_ch} (complex)\n")
            f.write(f"  freq_in:      {freq_in}\n")
            f.write(f"  freq_out:     {freq_out}\n\n")
            
            f.write(f"Convolution Parameters:\n")
            f.write(f"  kernel_size: ({self.kernel_size}, 1)\n")
            f.write(f"  stride:      ({self.stride}, 1)\n")
            f.write(f"  padding:     ({self.padding}, 0)\n\n")
            
            f.write(f"Data Files:\n")
            f.write(f"  input.dat:\n")
            f.write(f"    - Shape: [{in_ch}, {freq_in}]\n")
            f.write(f"    - Size: {in_ch * freq_in} complex numbers\n")
            f.write(f"    - Description: 輸入特徵 (單聲道等效)\n")
            f.write(f"    - Format: [imag_fp16][real_fp16] per line (32-bit)\n\n")
            
            f.write(f"  dw_weight.dat:\n")
            f.write(f"    - Shape: [{in_ch}, {self.kernel_size}]\n")
            f.write(f"    - Size: {in_ch * self.kernel_size} complex numbers\n")
            f.write(f"    - Description: Depthwise 卷積權重 (左右耳共用)\n")
            f.write(f"    - Format: [imag_fp16][real_fp16] per line (32-bit)\n\n")
            
            f.write(f"  pw_weight.dat:\n")
            f.write(f"    - Shape: [{out_ch}, {in_ch}]\n")
            f.write(f"    - Size: {out_ch * in_ch} complex numbers\n")
            f.write(f"    - Description: Pointwise 卷積權重 (左右耳共用)\n")
            f.write(f"    - Format: [imag_fp16][real_fp16] per line (32-bit)\n\n")
            
            f.write(f"  dw_output.dat:\n")
            f.write(f"    - Shape: [{in_ch}, {freq_out}]\n")
            f.write(f"    - Size: {in_ch * freq_out} complex numbers\n")
            f.write(f"    - Description: Depthwise 輸出 (中間驗證用)\n")
            f.write(f"    - Format: [imag_fp16][real_fp16] per line (32-bit)\n\n")
            
            f.write(f"  final_output.dat:\n")
            f.write(f"    - Shape: [{out_ch}, {freq_out}]\n")
            f.write(f"    - Size: {out_ch * freq_out} complex numbers\n")
            f.write(f"    - Description: 最終輸出 (Golden reference)\n")
            f.write(f"    - Format: [imag_fp16][real_fp16] per line (32-bit)\n\n")
            
            f.write(f"Computational Complexity (per time frame, 單聲道):\n")
            f.write(f"  Depthwise:\n")
            f.write(f"    - Complex MACs: {macs['dw_complex']}\n")
            f.write(f"    - Real MACs:    {macs['dw_complex']*4}\n")
            f.write(f"  Pointwise:\n")
            f.write(f"    - Complex MACs: {macs['pw_complex']}\n")
            f.write(f"    - Real MACs:    {macs['pw_complex']*4}\n")
            f.write(f"  Total:\n")
            f.write(f"    - Complex MACs: {macs['total_complex']}\n")
            f.write(f"    - Real MACs:    {macs['total_real']}\n")
            f.write(f"    - Cycles @ 256 Real MACs: {macs['cycles_256macs']:.0f}\n\n")
            
            f.write(f"Binaural Processing:\n")
            f.write(f"  - 左右耳使用相同權重\n")
            f.write(f"  - 可循序處理 (L→R) 或並行處理\n")
            f.write(f"  - 循序處理: {macs['cycles_256macs']:.0f} cycles/channel\n")
            f.write(f"  - 並行處理: {macs['cycles_256macs']:.0f} cycles/frame (需2組PEG)\n\n")
            
            f.write(f"="*60 + "\n")
        
        print(f"✓ 完成: {layer_dir}/")
        print(f"{'='*80}")
        
        return macs
    
    def generate_all_layers(self, output_dir="./pattern_fp16"):
        """生成所有層的 patterns"""
        print(f"\n{'='*80}")
        print(f"開始生成所有層的測試 Patterns (IEEE FP16)")
        print(f"{'='*80}")
        
        total_macs = []
        
        for i in range(len(self.LAYER_CONFIGS)):
            macs = self.generate_single_layer(i, output_dir)
            total_macs.append(macs)
        
        # 生成硬體配置說明
        self.save_hardware_config(output_dir, total_macs)
        
        # 總結
        print(f"\n{'='*80}")
        print(f"所有層 Patterns 生成完成!")
        print(f"{'='*80}")
        print(f"\n層運算量總結 (per time frame, 單聲道):")
        print(f"{'Layer':<20} {'Complex MACs':<16} {'Real MACs':<16} {'Cycles@256'}")
        print(f"{'-'*80}")
        for i, config in enumerate(self.LAYER_CONFIGS):
            macs = total_macs[i]
            print(f"{config['name']:<20} {macs['total_complex']:>13,}   "
                  f"{macs['total_real']:>13,}   {macs['cycles_256macs']:>10.0f}")
        print(f"{'-'*80}")
        
        total_cycles = sum(m['cycles_256macs'] for m in total_macs)
        print(f"{'Total per frame:':<20} {'':<16} {'':<16} {total_cycles:>10.0f}")
        print(f"\n雙聲道說明:")
        print(f"  - 左右耳使用 SAME 權重 (參數共用)")
        print(f"  - 可循序處理 (L→R) 或並行處理")
        print(f"  - 循序: ~{total_cycles:.0f} cycles/channel = ~{total_cycles*2:.0f} cycles/frame")
        print(f"  - 並行: ~{total_cycles:.0f} cycles/frame (需要雙倍硬體)")
        print(f"{'='*80}\n")
    
    def save_hardware_config(self, output_dir, total_macs: List[Dict]):
        """生成硬體配置說明"""
        config_file = f"{output_dir}/HARDWARE_CONFIG.txt"
        
        total_cycles = sum(m['cycles_256macs'] for m in total_macs)
        fp16_info = self.converter.get_fp16_info()
        
        with open(config_file, 'w') as f:
            f.write("="*80 + "\n")
            f.write("Hardware Configuration for Binaural Conv Encoder (IEEE FP16)\n")
            f.write("="*80 + "\n\n")
            
            f.write("數值格式:\n")
            f.write(f"  Format: IEEE 754 Half Precision (FP16)\n")
            f.write(f"  Total bits: {fp16_info['total_bits']}\n")
            f.write(f"  WXP (Exponent): {fp16_info['exp_bits']} bits (bias={fp16_info['exp_bias']})\n")
            f.write(f"  FRAC (Fraction): {fp16_info['frac_bits']} bits\n")
            f.write(f"  Bit layout: {fp16_info['format']}\n")
            f.write(f"  Range: ~6e-5 to ~6.5e4\n")
            f.write(f"  Precision: ~3 decimal digits\n\n")
            
            f.write("架構特性:\n")
            f.write("  - 左右耳共用同一組 Encoder/Decoder 權重\n")
            f.write("  - STFT Output: 64 complex frequency bins\n")
            f.write("  - Kernel Size: (5, 1) - 頻率維度卷積\n")
            f.write("  - Stride: (2, 1), Padding: (2, 0)\n")
            f.write("  - Total MACs: 256 Real = 64 Complex\n\n")
            
            f.write("處理策略:\n")
            f.write("  Option 1: 循序處理 (Sequential)\n")
            f.write("    - 先處理左耳，再處理右耳\n")
            f.write(f"    - 總延遲: ~{total_cycles*2:.0f} cycles/frame\n")
            f.write("    - 硬體需求: 1 組 PEG (256 MACs)\n")
            f.write("    - 優點: 最小面積\n\n")
            
            f.write("  Option 2: 並行處理 (Parallel)\n")
            f.write("    - 同時處理左右耳\n")
            f.write(f"    - 總延遲: ~{total_cycles:.0f} cycles/frame\n")
            f.write("    - 硬體需求: 2 組 PEG (512 MACs)\n")
            f.write("    - 優點: 最低延遲\n\n")
            
            f.write("推薦配置 (循序處理):\n")
            f.write("  MAC Units: 256 Real MACs = 64 Complex MACs (FP16)\n")
            f.write("  SRAM: 8 Banks × 32-bit × 20 depth\n")
            f.write("    - Total: 5,120 bits (160 complex weights)\n")
            f.write("    - Bandwidth: 256 bits/cycle\n")
            f.write("    - Format: [imag_fp16(16)][real_fp16(16)]\n\n")
            
            f.write("Weight 存取策略:\n")
            f.write("  共用權重:\n")
            f.write("    - Depthwise: 5 weights per channel\n")
            f.write("    - Pointwise: varies by layer\n")
            f.write("    - 左右耳重用同一組 SRAM weights\n\n")
            
            f.write("  Bank Interleaving:\n")
            f.write("    Depthwise: W[0]→Bank0, W[1]→Bank1, ..., W[4]→Bank4\n")
            f.write("    Pointwise: 按 output channel 分組\n\n")
            
            f.write("性能估算 (@ 16kHz, hop=80 samples):\n")
            f.write(f"  Frame interval: 5ms = 80,000 cycles @ 16MHz\n")
            f.write(f"  Processing time: {total_cycles*2:.0f} cycles (循序)\n")
            f.write(f"  Utilization: {(total_cycles*2/80000)*100:.2f}%\n")
            f.write(f"  結論: 256 MACs 綽綽有餘\n\n")
            
            f.write("FP16 vs BFloat16 比較:\n")
            f.write("  FP16 優勢:\n")
            f.write("    - 更高精度 (10-bit frac vs 7-bit frac)\n")
            f.write("    - 更適合音訊處理的動態範圍\n")
            f.write("    - IEEE 標準，工具鏈支援完整\n")
            f.write("  BFloat16 優勢:\n")
            f.write("    - 更大數值範圍 (8-bit exp vs 5-bit exp)\n")
            f.write("    - 與 FP32 轉換簡單 (直接截斷)\n\n")
            
            f.write("="*80 + "\n")
        
        print(f"\n✓ 硬體配置已保存: {config_file}")


def main():
    """主函數"""
    
    # 設定隨機種子以便重現
    np.random.seed(42)
    
    generator = ConvEncoderPatternGenerator(
        kernel_size=5,
        stride=2,
        padding=2
    )
    
    # 生成所有層
    generator.generate_all_layers(output_dir="./pattern_fp16")
    
    print("\n" + "="*80)
    print("使用說明")
    print("="*80)
    print("\nIEEE FP16 格式規格:")
    print("  ✓ WXP (Exponent):  5 bits")
    print("  ✓ FRAC (Fraction): 10 bits")
    print("  ✓ Total:           16 bits")
    print("  ✓ Layout: [sign(1)][exp(5)][frac(10)]")
    print("\n生成的文件:")
    print("  每層目錄 (Layer0_1to8/ ~ Layer5_128to128/):")
    print("    - input.dat:       輸入特徵 (FP16 complex)")
    print("    - dw_weight.dat:   Depthwise 權重 (左右耳共用)")
    print("    - pw_weight.dat:   Pointwise 權重 (左右耳共用)")
    print("    - dw_output.dat:   中間結果 (可選驗證)")
    print("    - final_output.dat: Golden 輸出")
    print("    - config.txt:      層配置信息")
    print("\n  HARDWARE_CONFIG.txt: 完整硬體架構說明")
    print("\n資料格式:")
    print("  ✓ 每行 32-bit hex: [imag_fp16(16)][real_fp16(16)]")
    print("  ✓ 複數實部在低 16 bits")
    print("  ✓ 複數虛部在高 16 bits")
    print("\n重要:")
    print("  ✓ 左右耳使用 SAME 權重")
    print("  ✓ 可循序處理或並行處理")
    print("  ✓ 循序處理只需 1 組 PEG (推薦)")
    print("  ✓ 並行處理需要 2 組 PEG")
    print("\n" + "="*80 + "\n")


if __name__ == "__main__":
    main()