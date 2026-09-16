import numpy as np
import os
from pathlib import Path
import math

class FP16Converter:
    """
    IEEE 754 FP16 格式轉換器
    """
    @staticmethod
    def float32_to_fp16_hex(value):
        if np.isnan(value) or np.isinf(value): value = 0.0
        fp16_val = np.float16(value)
        fp16_bits = fp16_val.view(np.uint16)
        return f"{fp16_bits:04x}"

    @staticmethod
    def complex_to_hex(complex_val):
        real_hex = FP16Converter.float32_to_fp16_hex(complex_val.real)
        imag_hex = FP16Converter.float32_to_fp16_hex(complex_val.imag)
        return f"{imag_hex}{real_hex}"

class ConvPatternGenerator:
    def __init__(self, layer_idx=4):
        self.output_dir = "./pattern_banked_fp16_strict"
        Path(self.output_dir).mkdir(parents=True, exist_ok=True)
        
        # (In_Ch, In_Freq, Out_Ch)
        self.layer_configs = [
            (1, 64, 8), (8, 32, 16), (16, 16, 32), 
            (32, 8, 64), (64, 4, 128), (128, 2, 128)
        ]
        
        self.cfg = self.layer_configs[layer_idx]
        self.cin, self.fin, self.cout = self.cfg
        self.fout = math.floor((self.fin + 4 - 5) / 2) + 1
        
        self.num_banks = 16
        self.words_per_bank = 4   
        self.total_words_per_line = self.num_banks * self.words_per_bank
        
        print(f"=== Layer {layer_idx} (Strict FP16 Calc) ===")
        print(f"In: ({self.cin}, {self.fin}), Out: ({self.cout}, {self.fout})")

    def generate_random_complex(self, shape):
        # 初始生成就限制在 float16
        real = np.random.uniform(-1.0, 1.0, shape).astype(np.float16)
        imag = np.random.uniform(-1.0, 1.0, shape).astype(np.float16)
        return real + 1j * imag

    # --- 關鍵修正：嚴格 FP16 複數運算 ---
    def strict_fp16_complex_mul_add(self, a, b, acc=0j):
        """
        模擬硬體行為：
        (ar + j*ai) * (br + j*bi) + acc
        = (ar*br - ai*bi + acc_r) + j(ar*bi + ai*br + acc_i)
        每一步都強制轉型回 float16
        """
        # 拆解實虛部 (確保輸入已是 fp16)
        ar, ai = a.real, a.imag
        br, bi = b.real, b.imag
        acc_r, acc_i = acc.real, acc.imag
        
        # 1. 乘法 (立即截斷)
        t1 = np.float16(ar * br)
        t2 = np.float16(ai * bi)
        t3 = np.float16(ar * bi)
        t4 = np.float16(ai * br)
        
        # 2. 加減法 (立即截斷)
        # Real part: t1 - t2 + acc_r
        res_r = np.float16(t1 - t2)
        res_r = np.float16(res_r + acc_r)
        
        # Imag part: t3 + t4 + acc_i
        res_i = np.float16(t3 + t4)
        res_i = np.float16(res_i + acc_i)
        
        return res_r + 1j * res_i

    def complex_conv1d_depthwise(self, x, weight):
        # x: (Cin, Fin), weight: (Cin, K=5)
        # Pad with FP16 zeros
        x_pad = np.pad(x, ((0,0), (2,2)), mode='constant')
        out = np.zeros((self.cin, self.fout), dtype=np.complex64) # 暫存容器，但內容會是 fp16
        
        # 慢速迴圈：為了模擬嚴格 FP16 行為
        print("Computing Depthwise (Strict FP16)...")
        for c in range(self.cin):
            for f in range(self.fout):
                acc = np.float16(0.0) + 1j*np.float16(0.0)
                # Kernel loop
                for k in range(5):
                    val = x_pad[c, f*2 + k]
                    w   = weight[c, k]
                    # 呼叫嚴格運算函數
                    acc = self.strict_fp16_complex_mul_add(val, w, acc)
                out[c, f] = acc
        return out

    def complex_pointwise(self, x, weight):
        # x: (Cin, Fout), weight: (Cout, Cin)
        out = np.zeros((self.cout, self.fout), dtype=np.complex64)
        
        print("Computing Pointwise (Strict FP16)...")
        for f in range(self.fout):
            for cout_idx in range(self.cout):
                acc = np.float16(0.0) + 1j*np.float16(0.0)
                # Accumulate over Cin
                for cin_idx in range(self.cin):
                    val = x[cin_idx, f]
                    w   = weight[cout_idx, cin_idx]
                    acc = self.strict_fp16_complex_mul_add(val, w, acc)
                out[cout_idx, f] = acc
        return out

    # --- Bank 切分與存檔 (與之前相同) ---
    def pack_chunk_hex(self, chunk_4_words):
        if len(chunk_4_words) < 4:
            chunk_4_words = np.concatenate([chunk_4_words, np.zeros(4 - len(chunk_4_words), dtype=np.complex64)])
        hex_list = [FP16Converter.complex_to_hex(val) for val in chunk_4_words]
        return "".join(reversed(hex_list))

    def save_banked_files(self, flat_data_all, prefix_name):
        remainder = len(flat_data_all) % self.total_words_per_line
        if remainder != 0:
            flat_data_all = np.concatenate([flat_data_all, np.zeros(self.total_words_per_line - remainder, dtype=np.complex64)])
        
        num_lines = len(flat_data_all) // self.total_words_per_line
        reshaped = flat_data_all.reshape(num_lines, self.num_banks, self.words_per_bank)
        
        for bank_idx in range(self.num_banks):
            bank_data = reshaped[:, bank_idx, :] 
            lines_hex = [self.pack_chunk_hex(row) for row in bank_data]
            filename = f"{prefix_name}_bank{bank_idx}.dat"
            self.save_file(lines_hex, filename)

    def save_file(self, lines, filename):
        filepath = os.path.join(self.output_dir, filename)
        with open(filepath, 'w') as f:
            for line in lines:
                f.write(f"{line}\n")

    def run(self):
        # 1. Gen Data (FP16)
        input_feat = self.generate_random_complex((self.cin, self.fin))
        dw_weight = self.generate_random_complex((self.cin, 5))
        pw_weight = self.generate_random_complex((self.cout, self.cin))
        
        # 2. Compute (Strict FP16)
        dw_out = self.complex_conv1d_depthwise(input_feat, dw_weight)
        final_out = self.complex_pointwise(dw_out, pw_weight)
        
        # 3. Save
        self.save_banked_files(input_feat.T.flatten(), "ifbuf")
        
        dw_flat = dw_weight.T.flatten()
        pw_flat = pw_weight.flatten()
        wbuf_flat = np.concatenate([dw_flat, pw_flat])
        self.save_banked_files(wbuf_flat, "wbuf")
        
        self.save_banked_files(dw_out.T.flatten(), "dw_output_golden")
        self.save_banked_files(final_out.T.flatten(), "ofbuf_golden")
        
        print(f"\nFiles saved to {self.output_dir}")

if __name__ == "__main__":
    np.random.seed(42)
    gen = ConvPatternGenerator(layer_idx=4)
    gen.run()