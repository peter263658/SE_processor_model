// ISTFT directed test: STFT_TOP (i_stft_mode=ISTFT) reusing FFT128 as IFFT.
// packages
../../01_RTL/Define/Param_def.sv
../../01_RTL/Define/Mem_def.sv

// testbench
./tb_istft.sv

// design (full STFT_TOP chain; ISTFT reuses FFT128 + adds conj_sym_feeder + istft_reorder)
../../01_RTL/engine/FFT/STFT_TOP.sv
../../01_RTL/engine/FFT/FFT128.sv
../../01_RTL/engine/FFT/SdfUnit.sv
../../01_RTL/engine/FFT/SdfUnit2.sv
../../01_RTL/engine/FFT/Butterfly.sv
../../01_RTL/engine/FFT/DelayBuffer.sv
../../01_RTL/engine/FFT/Multiply.sv
../../01_RTL/engine/FFT/Twiddle128.sv
../../01_RTL/engine/FFT/Hamming.sv
../../01_RTL/engine/FFT/Bit_Reorder.sv
../../01_RTL/engine/FFT/conj_sym_feeder.sv
../../01_RTL/engine/FFT/istft_reorder.sv
// STFT_TOP also instantiates the e4m3 cast (unused in ISTFT mode but must elaborate)
../../01_RTL/common/Designware/DW_fp_i2flt_inst.sv
../../01_RTL/common/Designware/DW_fp_mult_inst.sv
../../01_RTL/engine/CE/ce_fp_util.sv
../../01_RTL/common/utils/fp16_to_fp8.sv
../../01_RTL/engine/FFT/fxp_to_e4m3.sv
../../01_RTL/engine/FFT/stft_fp8_cast.sv

// vcs flags + DesignWare
-debug_access+all +v2k +notimingcheck -sverilog -timescale=1ns/10ps
-y /usr/cad/synopsys/synthesis/cur/dw/sim_ver +libext+.v
+incdir+/usr/cad/synopsys/synthesis/cur/dw/sim_ver+
