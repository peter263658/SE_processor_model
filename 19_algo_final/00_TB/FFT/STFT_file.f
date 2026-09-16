// Parameter / config. Definition
// -----------------------------------------------------------------------------
../../01_RTL/Define/Param_def.sv
../../01_RTL/Define/Mem_def.sv

// testbench
// -----------------------------------------------------------------------------
./tb_stft.sv

// DesignWare
// -----------------------------------------------------------------------------
// ../../01_RTL/Designware/DW_fp_add_inst.sv
// ../../01_RTL/Designware/DW_fp_addsub_inst.sv
// ../../01_RTL/Designware/DW_fp_sub_inst.sv
// ../../01_RTL/Designware/DW_fp_cmp_inst.sv
// ../../01_RTL/Designware/DW_fp_exp_inst.sv
// ../../01_RTL/Designware/DW_fp_ln_inst.sv
// ../../01_RTL/Designware/DW_fp_mult_inst.sv

// design files
// -----------------------------------------------------------------------------
// source rtl
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

// ISTFT path (reuses FFT128 as IFFT): conj-sym feeder + reorder/synth-window
../../01_RTL/engine/FFT/conj_sym_feeder.sv
../../01_RTL/engine/FFT/istft_reorder.sv

// STFT -> e4m3 cast (block-float for UFBUF) + its fp helpers + DW wrappers
../../01_RTL/common/Designware/DW_fp_i2flt_inst.sv
../../01_RTL/common/Designware/DW_fp_mult_inst.sv
../../01_RTL/engine/CE/ce_fp_util.sv
../../01_RTL/common/utils/fp16_to_fp8.sv
../../01_RTL/engine/FFT/fxp_to_e4m3.sv
../../01_RTL/engine/FFT/stft_fp8_cast.sv

// For vcs
-debug_access+all +v2k +notimingcheck -sverilog -timescale=1ns/10ps
-y /usr/cad/synopsys/synthesis/cur/dw/sim_ver +libext+.v
+incdir+/usr/cad/synopsys/synthesis/cur/dw/sim_ver+
-P /usr/cad/synopsys/verdi/cur/share/PLI/VCS/LINUX64/novas.tab
    /usr/cad/synopsys/verdi/cur/share/PLI/VCS/LINUX64/pli.a

// // For ncverilog
// +access+r -timescale 1ns/10ps -notimingcheck 
// -y /home/cad/synopsys/synthesis/2021.06-sp2/dw/sim_ver +libext+.v
// +incdir+/home/cad/synopsys/synthesis/2021.06-sp2/dw/sim_ver+