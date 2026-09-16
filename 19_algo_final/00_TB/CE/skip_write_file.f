// tb_skip_write — encoder skip-WRITE capture + STFT spec region (T2 #8).
+incdir+../../01_RTL/engine/CE
../../01_RTL/Define/Param_def.sv
../../01_RTL/Define/Mem_def.sv
../../01_RTL/engine/CE/ce_subop_pkg.sv
// CE datapath spine
../../01_RTL/common/Designware/DW_fp_add_inst.sv
../../01_RTL/engine/CE/ce_fp_util.sv
../../01_RTL/engine/CE/PEG.sv
../../01_RTL/engine/CE/oacc.sv
../../01_RTL/engine/CE/ppu_fma.sv
../../01_RTL/engine/CE/CE_core.sv
// CE control
../../01_RTL/engine/CE/mem_controller.sv
../../01_RTL/engine/CE/inst_decoder.sv
// engine
../../01_RTL/engine/CE/ce_operand_gather.sv
../../01_RTL/engine/CE/ppu_const_cache.sv
../../01_RTL/common/utils/fp8_to_fp16.sv
../../01_RTL/engine/CE/ce_kv_cache.sv
../../01_RTL/engine/Softmax/Softmax_v3.sv
../../01_RTL/engine/CE/ce_attn_sfu.sv
../../01_RTL/engine/CE/ce_engine.sv
// mask_apply (phase 2 end-use) + deps
../../01_RTL/common/Designware/DW_fp_i2flt_inst.sv
../../01_RTL/common/Designware/DW_fp_flt2i_inst.sv
../../01_RTL/common/Designware/DW_fp_mult_inst.sv
../../01_RTL/common/Designware/DW_fp_addsub_inst.sv
../../01_RTL/common/utils/fp16_to_fp32.sv
../../01_RTL/engine/FFT/mask_apply.sv
// testbench
tb_skip_write.sv
