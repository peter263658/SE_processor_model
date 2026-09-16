import numpy as np
import os

def float_to_hex_fp16(f):
    """將浮點數轉換為 16-bit 十六進位字串 (FP16)"""
    h = np.array([f], dtype=np.float16).view(np.uint16)[0]
    return f"{h:04x}"

def gen_mem_files(name, banks, width, depth):
    """
    name: Buffer 名稱 (wbuf, ifbuf, etc.)
    banks: Bank 數量
    width: 每個 Bank 的位元寬度 (128 bit)
    depth: 每個 Bank 的深度
    """
    print(f"Generating patterns for {name}...")
    folder = f"./patterns/{name}"
    if not os.path.exists(folder):
        os.makedirs(folder)

    # 每個 128-bit 字組包含 8 個 FP16 (128/16 = 8)
    fp16_per_word = width // 16

    for b in range(banks):
        file_path = f"{folder}/bank_{b}.mem"
        with open(file_path, "w") as f:
            for d in range(depth):
                # 生成 8 個隨機 FP16 並拼成一個 128-bit hex string
                word_hex = ""
                for _ in range(fp16_per_word):
                    rand_val = np.random.uniform(-1.0, 1.0)
                    word_hex = float_to_hex_fp16(rand_val) + word_hex # 小端對齊順序
                f.write(word_hex + "\n")

if __name__ == "__main__":
    # 配置參數
    specs = [
        {"name": "wbuf",  "banks": 16, "width": 128, "depth": 256},
        {"name": "ifbuf", "banks": 16, "width": 128, "depth": 8},
        {"name": "ofbuf", "banks": 16, "width": 128, "depth": 8},
        {"name": "kvbuf", "banks": 32, "width": 128, "depth": 512},
    ]

    for s in specs:
        gen_mem_files(s["name"], s["banks"], s["width"], s["depth"])
    
    print("Pattern generation complete. Files are in ./patterns/")