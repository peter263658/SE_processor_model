// inst_decoder (Step 2-2) compile + smoke filelist.
// packages first; ce_subop_rom.svh is `included by ce_subop_pkg -> +incdir the CE dir.
+incdir+../../01_RTL/engine/CE
../../01_RTL/Define/Param_def.sv
../../01_RTL/Define/Mem_def.sv
../../01_RTL/engine/CE/ce_subop_pkg.sv
../../01_RTL/engine/CE/inst_decoder.sv
// testbench (self-checking smoke) — last
./tb_inst_decoder_smoke.sv
