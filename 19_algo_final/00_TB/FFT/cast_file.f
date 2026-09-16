// STFT(Q1.15) -> e4m3 cast: fxp_to_e4m3 vs encoder.model.0.0.depthwise e4m3 golden.
// (DW_fp_i2flt / DW_fp_mult resolved from -y sim_ver on the vcs command line.)

// package first
../../01_RTL/Define/Param_def.sv

// DesignWare wrappers
../../01_RTL/common/Designware/DW_fp_i2flt_inst.sv
../../01_RTL/common/Designware/DW_fp_mult_inst.sv

// fp helpers: fp16_scale2 (ce_fp_util) + fp16_to_fp8 (e4m3 RNE cast)
../../01_RTL/engine/CE/ce_fp_util.sv
../../01_RTL/common/utils/fp16_to_fp8.sv

// DUT
../../01_RTL/engine/FFT/fxp_to_e4m3.sv

// testbench
./tb_fxp_to_e4m3.sv
