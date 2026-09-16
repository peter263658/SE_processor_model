import numpy as np
import struct
import os
from pathlib import Path

class FP16Converter:
    """IEEE FP16 格式轉換工具"""
    @staticmethod
    def float32_to_fp16_hex(value):
        if np.isnan(value) or np.isinf(value): value = 0.0
        fp16_val = np.float16(value)
        fp16_bits = np.frombuffer(fp16_val.tobytes(), dtype=np.uint16)[0]
        return f"{fp16_bits:04x}"

    @staticmethod
    def complex_to_hex(complex_val):
        """
        複數 → 32-bit hex
        根據你的硬體需求，若發現實虛部相反，請在此交換
        預設: [imag_16][real_16] (高位是虛部)
        """
        real_hex = FP16Converter.float32_to_fp16_hex(complex_val.real)
        imag_hex = FP16Converter.float32_to_fp16_hex(complex_val.imag)
        # return f"{real_hex}{imag_hex}" # 若需要 Real 在高位請用這行
        return f"{real_hex}{imag_hex}"  

class AdderTreePatternGenerator:
    def __init__(self, num_transactions=16, num_inputs=64):
        self.num_transactions = num_transactions
        self.num_inputs = num_inputs
        self.output_dir = "./pattern_at_fp16"
        Path(self.output_dir).mkdir(parents=True, exist_ok=True)
        
        print(f"{'='*60}")
        print(f"Adder Tree Pattern Generator (FP16)")
        print(f"Transactions: {self.num_transactions}")
        print(f"Inputs per transaction: {self.num_inputs}")
        print(f"{'='*60}")

    def generate_random_data(self):
        shape = (self.num_transactions, self.num_inputs)
        # 範圍設小一點 (-2~2)，方便肉眼檢查，也避免 FP16 溢位
        real = np.random.uniform(-2.0, 2.0, shape).astype(np.float32)
        imag = np.random.uniform(-2.0, 2.0, shape).astype(np.float32)
        return real + 1j * imag

    def save_hex_file(self, data, filename):
        filepath = os.path.join(self.output_dir, filename)
        with open(filepath, 'w') as f:
            for row in data:
                for val in row:
                    f.write(f"{FP16Converter.complex_to_hex(val)}\n")
        print(f"Saved Hex: {filepath}")

    def save_decimal_file(self, data, filename, description="Input Data"):
        """儲存人類可讀的 Input Decimal 檔案"""
        filepath = os.path.join(self.output_dir, filename)
        with open(filepath, 'w') as f:
            f.write(f"Description: {description}\n")
            f.write(f"Format: (Real, Imag)\n")
            f.write("="*80 + "\n")
            
            # data shape: (num_transactions, num_inputs)
            for t in range(self.num_transactions):
                f.write(f"Transaction #{t}:\n")
                row = data[t]
                # 每行印 4 個複數，方便閱讀
                for i in range(0, len(row), 4):
                    chunk = row[i:i+4]
                    line_str = ", ".join([f"({x.real:6.3f}, {x.imag:6.3f})" for x in chunk])
                    f.write(f"  [{i:02d}-{min(i+3, len(row)-1):02d}] {line_str}\n")
                f.write("-" * 40 + "\n")
        print(f"Saved Decimal: {filepath}")

    def save_answer_key(self, output, mode_name):
        filepath = os.path.join(self.output_dir, f"answer_key_output_{mode_name}.txt")
        with open(filepath, 'w') as f:
            f.write(f"Mode: {mode_name} Output Key\n")
            f.write("="*60 + "\n")
            for t in range(self.num_transactions):
                f.write(f"Transaction #{t}:\n")
                if mode_name == "GLAC":
                    # GLAC reshape 後是 (16, 1)，取 [0] 變 scalar
                    val = output[t][0] 
                    f.write(f"  Golden Output: ({val.real:.4f}, {val.imag:.4f})\n")
                else:
                    f.write(f"  Golden Output (First 4): {output[t][:4]}\n")
                    f.write(f"  Golden Output (Last 4):  {output[t][-4:]}\n")
                f.write("-" * 20 + "\n")
        print(f"Saved Output Key: {filepath}")

    def generate_glac_patterns(self):
        print("\n[Genering GLAC Patterns...]")
        data0 = self.generate_random_data()
        data1 = np.zeros_like(data0) # GLAC 不用 data1

        # Golden Calculation (Sum over axis 1)
        golden_out = np.sum(data0, axis=1).reshape(-1, 1)

        # Save Files
        self.save_hex_file(data0, "glac_input0.dat")
        self.save_hex_file(data1, "glac_input1.dat")
        self.save_hex_file(golden_out, "glac_golden_hex.dat")
        
        # Save Decimal for Debugging
        self.save_decimal_file(data0, "glac_input0_decimal.txt", "GLAC Input Data 0")
        self.save_answer_key(golden_out, "GLAC")

    def generate_psum_patterns(self):
        print("\n[Genering PSUM Patterns...]")
        data0 = self.generate_random_data()
        data1 = self.generate_random_data()
        golden_out = data0 + data1

        self.save_hex_file(data0, "psum_input0.dat")
        self.save_hex_file(data1, "psum_input1.dat")
        self.save_hex_file(golden_out, "psum_golden_hex.dat")
        # PSUM Input 比較多，有需要再打開，避免檔案太大
        self.save_decimal_file(data0, "psum_input0_decimal.txt", "PSUM Input Data 0")
        self.save_answer_key(golden_out, "PSUM")

if __name__ == "__main__":
    np.random.seed(123)
    gen = AdderTreePatternGenerator(num_transactions=16, num_inputs=64)
    gen.generate_glac_patterns()
    gen.generate_psum_patterns() # 暫時註解掉，專注測 GLAC
    print("\nDone! Check './pattern_at_fp16' folder.")