// CE_core (W4A8 datapath spine) — compile + smoke
// packages first
../../01_RTL/Define/Param_def.sv
../../01_RTL/Define/Mem_def.sv
// DW fp16 adder wrapper
../../01_RTL/common/Designware/DW_fp_add_inst.sv
// CE compute datapath
../../01_RTL/engine/CE/ce_fp_util.sv
../../01_RTL/engine/CE/PEG.sv
../../01_RTL/engine/CE/oacc.sv
../../01_RTL/engine/CE/ppu_fma.sv
../../01_RTL/engine/CE/CE_core.sv
// tb
./tb_ce_core_smoke.sv
