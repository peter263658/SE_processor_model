// ce_engine (Step 2-5) elaborate filelist — full W4A8 engine (behavioral WBUF/UF).
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
// engine: gather + const cache + softmax + top
../../01_RTL/engine/CE/ce_operand_gather.sv
../../01_RTL/engine/CE/ppu_const_cache.sv
../../01_RTL/common/utils/fp8_to_fp16.sv
../../01_RTL/engine/CE/ce_kv_cache.sv
../../01_RTL/engine/Softmax/Softmax_v3.sv
../../01_RTL/engine/CE/ce_attn_sfu.sv
../../01_RTL/engine/CE/ce_engine.sv
