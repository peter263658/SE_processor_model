# Debug Reference

This folder is for inspection and numerical comparison:

- `layer_io_fp32/`: fp32/complex64 module input/output tensors from the same
  golden waveform used to generate the block-float testbench activations (incl.
  Stft/iSTFT Q1.15 DSP I/O and all pre-quant outputs, which are NOT in tb_pattern).
- `export_reference_debug/`: readable fp32/fp16/raw_s8/hex/dec views copied from
  the full export.

Do not use this folder as the source of packed testbench weights.
