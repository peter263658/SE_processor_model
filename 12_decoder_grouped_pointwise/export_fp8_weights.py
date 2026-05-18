"""
Export FP8 Weights from QAT-trained Checkpoint
===============================================
Reads a PyTorch Lightning checkpoint, converts every learnable parameter
to float8_e4m3fn (or e5m2), and writes layer-by-layer text files containing
both the HEX (raw 1-byte FP8 encoding) and DECIMAL (dequantized FP32 value)
representations side-by-side for hardware verification.

Output structure:
    <output_dir>/
        summary.txt                        # layer list with shapes
        <sanitised_layer_name>.txt         # per-layer file (hex + decimal)
        <sanitised_layer_name>.csv         # per-layer CSV for easy parsing

Usage:
    python export_fp8_weights.py \
        --checkpoint <path_to_ckpt> \
        --output_dir ./fp8_weights \
        --fp8_dtype e4m3
"""

import argparse
import os
import sys
import csv

import torch
import numpy as np


# ─────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────

def get_fp8_dtype(dtype_str: str):
    if dtype_str.lower() == "e4m3":
        return torch.float8_e4m3fn
    elif dtype_str.lower() == "e5m2":
        return torch.float8_e5m2
    else:
        raise ValueError(f"Unsupported: {dtype_str}. Use 'e4m3' or 'e5m2'.")


def tensor_to_fp8_hex_decimal(tensor: torch.Tensor, fp8_dtype):
    """
    Convert a real-valued FP32/FP16 tensor to FP8.
    Returns:
        hex_list:     list of hex strings   (e.g. "0x3C")
        decimal_list: list of float strings (e.g. "1.0")
        fp8_tensor:   the FP8 tensor itself
    """
    flat = tensor.detach().cpu().float().flatten()

    # Dynamic per-tensor scaling  (same logic as QAT)
    max_representable = 448.0 if fp8_dtype == torch.float8_e4m3fn else 57344.0
    amax = flat.abs().max().clamp(min=1e-12)
    scale = max_representable / amax

    scaled   = flat * scale
    fp8_vals = scaled.to(fp8_dtype)                     # actual FP8 bytes
    deq_vals = fp8_vals.to(torch.float32) / scale       # dequantized decimal

    # Raw byte view — float8 is 1 byte, same width as uint8
    raw_bytes = fp8_vals.view(torch.uint8)

    hex_list     = [f"0x{b:02X}" for b in raw_bytes.tolist()]
    decimal_list = [f"{v:.8e}" for v in deq_vals.tolist()]

    return hex_list, decimal_list, fp8_vals, scale.item()


def sanitize_name(name: str) -> str:
    """Convert 'encoder.model.0.0.depthwise_r.weight' → safe filename."""
    return name.replace(".", "_").replace("/", "_")


# ─────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Export QAT checkpoint weights to FP8 (hex + decimal)")
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="Path to PyTorch Lightning .ckpt file")
    parser.add_argument("--output_dir", type=str, default="./fp8_weights",
                        help="Directory to write exported weight files")
    parser.add_argument("--fp8_dtype", type=str, default="e4m3",
                        choices=["e4m3", "e5m2"],
                        help="FP8 format (default: e4m3)")
    args = parser.parse_args()

    fp8_dtype = get_fp8_dtype(args.fp8_dtype)
    os.makedirs(args.output_dir, exist_ok=True)

    # ── Load checkpoint ──
    print(f"📂 Loading checkpoint: {args.checkpoint}")
    ckpt = torch.load(args.checkpoint, map_location="cpu")
    state_dict = ckpt.get("state_dict", ckpt)

    # Strip 'model.' prefix added by PyTorch Lightning
    cleaned = {}
    for k, v in state_dict.items():
        clean_key = k.replace("model.model.", "model.").replace("model.", "", 1)

        # Skip QAT parametrization wrapper keys
        # e.g.  "parametrizations.weight.original" → treat as "weight"
        if "parametrizations" in clean_key:
            # Only keep the "original" base tensor
            if "original" not in clean_key:
                continue
            clean_key = clean_key.replace("parametrizations.", "").replace(".original", "")

        cleaned[clean_key] = v

    state_dict = cleaned

    # ── Summary file ──
    summary_path = os.path.join(args.output_dir, "summary.txt")
    summary_lines = []
    summary_lines.append(f"FP8 Weight Export Summary")
    summary_lines.append(f"========================")
    summary_lines.append(f"Checkpoint : {args.checkpoint}")
    summary_lines.append(f"FP8 Format : {args.fp8_dtype.upper()}")
    summary_lines.append(f"Total keys : {len(state_dict)}")
    summary_lines.append(f"")
    summary_lines.append(f"{'Layer Name':<70} {'Shape':<25} {'Dtype':<15} {'#Params':>10}")
    summary_lines.append("-" * 120)

    layer_count = 0
    total_params = 0

    for name, param in sorted(state_dict.items()):
        shape_str = str(list(param.shape))
        dtype_str = str(param.dtype)
        n_params = param.numel()
        total_params += n_params
        summary_lines.append(f"{name:<70} {shape_str:<25} {dtype_str:<15} {n_params:>10}")

        # ── Skip non-floating and non-complex (e.g. BN num_batches_tracked) ──
        if not (param.is_floating_point() or param.is_complex()):
            layer_count += 1
            # Still write a small note file
            note_path = os.path.join(args.output_dir, sanitize_name(name) + ".txt")
            with open(note_path, "w") as f:
                f.write(f"# Layer: {name}\n")
                f.write(f"# Shape: {shape_str}\n")
                f.write(f"# Dtype: {dtype_str} (integer/bool — not quantized)\n")
                f.write(f"# Values:\n")
                for val in param.flatten().tolist():
                    f.write(f"{val}\n")
            continue

        layer_count += 1
        safe_name = sanitize_name(name)

        if param.is_complex():
            # ── Complex parameter: export real and imag separately ──
            for part_label, part_tensor in [("real", param.real), ("imag", param.imag)]:
                hex_list, dec_list, _, scale = tensor_to_fp8_hex_decimal(
                    part_tensor, fp8_dtype)

                # .txt file
                txt_path = os.path.join(args.output_dir,
                                        f"{safe_name}_{part_label}.txt")
                with open(txt_path, "w") as f:
                    f.write(f"# Layer : {name} ({part_label} part)\n")
                    f.write(f"# Shape : {list(part_tensor.shape)}\n")
                    f.write(f"# Scale : {scale}\n")
                    f.write(f"# Format: FP8 {args.fp8_dtype.upper()}\n")
                    f.write(f"# Total : {len(hex_list)} values\n")
                    f.write(f"#\n")
                    f.write(f"# {'Index':<10} {'Hex':<10} {'Decimal':>15}\n")
                    f.write(f"# {'-'*10} {'-'*10} {'-'*15}\n")
                    for i, (h, d) in enumerate(zip(hex_list, dec_list)):
                        f.write(f"  {i:<10} {h:<10} {d:>15}\n")

                # .csv file
                csv_path = os.path.join(args.output_dir,
                                        f"{safe_name}_{part_label}.csv")
                with open(csv_path, "w", newline="") as f:
                    writer = csv.writer(f)
                    writer.writerow(["index", "hex", "decimal"])
                    for i, (h, d) in enumerate(zip(hex_list, dec_list)):
                        writer.writerow([i, h, d])

                print(f"  ✅ {name} ({part_label}): {len(hex_list)} values → "
                      f"{txt_path}")

        else:
            # ── Real parameter ──
            hex_list, dec_list, _, scale = tensor_to_fp8_hex_decimal(
                param, fp8_dtype)

            # .txt file
            txt_path = os.path.join(args.output_dir, f"{safe_name}.txt")
            with open(txt_path, "w") as f:
                f.write(f"# Layer : {name}\n")
                f.write(f"# Shape : {list(param.shape)}\n")
                f.write(f"# Scale : {scale}\n")
                f.write(f"# Format: FP8 {args.fp8_dtype.upper()}\n")
                f.write(f"# Total : {len(hex_list)} values\n")
                f.write(f"#\n")
                f.write(f"# {'Index':<10} {'Hex':<10} {'Decimal':>15}\n")
                f.write(f"# {'-'*10} {'-'*10} {'-'*15}\n")
                for i, (h, d) in enumerate(zip(hex_list, dec_list)):
                    f.write(f"  {i:<10} {h:<10} {d:>15}\n")

            # .csv file
            csv_path = os.path.join(args.output_dir, f"{safe_name}.csv")
            with open(csv_path, "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(["index", "hex", "decimal"])
                for i, (h, d) in enumerate(zip(hex_list, dec_list)):
                    writer.writerow([i, h, d])

            print(f"  ✅ {name}: {len(hex_list)} values → {txt_path}")

    # ── Write summary ──
    summary_lines.append("-" * 120)
    summary_lines.append(f"Total layers: {layer_count}    Total parameters: {total_params}")
    with open(summary_path, "w") as f:
        f.write("\n".join(summary_lines) + "\n")

    print(f"\n📋 Summary written to: {summary_path}")
    print(f"📁 All weight files in: {args.output_dir}/")
    print(f"🎯 Total: {layer_count} layers, {total_params:,} parameters exported.")


if __name__ == "__main__":
    main()
