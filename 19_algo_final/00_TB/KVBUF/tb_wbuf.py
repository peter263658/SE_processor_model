import os
import random

def generate_patterns():
    # Parameters according to Param_def.sv and Mem_def.sv
    WBUF_BANKS = 16
    WBUF_DEPTH = 256
    CHIP_I_WIDTH = 128
    FP8_NUM = CHIP_I_WIDTH // 8  # 16 FP8 values per cycle

    # Determine directory paths
    current_dir = os.path.dirname(os.path.abspath(__file__))
    pattern_dir = os.path.join(current_dir, "pattern")
    os.makedirs(pattern_dir, exist_ok=True)
    
    in_file = os.path.join(pattern_dir, "wbuf_in.dat")
    
    # The SRAM total bit width is: 16 banks * 256 depth * 64 bits = 262144 bits.
    # The CHIP_I_WIDTH is 128 bits.
    # Total lines = 262144 / 128 = 2048 lines.
    TOTAL_LINES = (WBUF_BANKS * WBUF_DEPTH * 64) // CHIP_I_WIDTH
    
    with open(in_file, "w") as f:
        # Each line is a 128-bit hex string (32 hex characters)
        for line_idx in range(TOTAL_LINES):
            # Byte 0: bank or line info for debugging
            # Byte 2-15: random fp8 values
            hex_str = ""
            for i in range(FP8_NUM):
                if i == 0:
                    val = (line_idx >> 8) & 0xFF
                elif i == 1:
                    val = line_idx & 0xFF
                else:
                    val = random.randint(0, 255)
                # Prepend to build MSB to LSB
                hex_str = f"{val:02x}" + hex_str 
            f.write(f"{hex_str}\n")
                
    print(f"=========================================================")
    print(f"Generated {TOTAL_LINES} patterns in {in_file}")
    print(f"Each line contains 16 FP8 values (128 bits).")
    print(f"=========================================================")

if __name__ == "__main__":
    generate_patterns()
