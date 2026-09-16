// tb_ce_engine_golden — CE W4A8 per-layer GOLDEN numeric sim (task 4/9).
// Full ce_engine (control + behavioral WBUF/UF + gather + dealer + CE_core) vs the REAL
// deploy image + fp32-reference-derived golden for encoder.model.0.0.pointwise (sub-op 1).
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
// testbench
tb_ce_engine_proj.sv
