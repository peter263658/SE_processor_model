// Parameter / config. Definition
// -----------------------------------------------------------------------------
../../01_RTL/Define/Param_def.sv
../../01_RTL/Define/Mem_def.sv

// testbench
// -----------------------------------------------------------------------------
./tb_CE.sv

// DesignWare
// -----------------------------------------------------------------------------
../../01_RTL/Designware/DW_fp_add_inst.sv
../../01_RTL/Designware/DW_fp_addsub_inst.sv
// ../../01_RTL/Designware/DW_fp_sub_inst.sv
// ../../01_RTL/Designware/DW_fp_cmp_inst.sv
// ../../01_RTL/Designware/DW_fp_exp_inst.sv
// ../../01_RTL/Designware/DW_fp_ln_inst.sv
../../01_RTL/Designware/DW_fp_mult_inst.sv
// ../../01_RTL/Designware/DW_fp_flt2i_inst.sv
// ../../01_RTL/Designware/DW_fp_i2flt_inst.sv


// design files
// -----------------------------------------------------------------------------
../../01_RTL/CE/CE_AT.sv
../../01_RTL/CE/CE_PEG.sv
../../01_RTL/CE/CE.sv



// // For vcs
-debug_access+all +v2k +notimingcheck -sverilog -timescale=1ns/10ps 
-y /home/cad/synopsys/synthesis/2021.06-sp2/dw/sim_ver +libext+.v
+incdir+/home/cad/synopsys/synthesis/2021.06-sp2/dw/sim_ver+
-P /usr/cad/synopsys/verdi/cur/share/PLI/VCS/LINUX64/novas.tab
    /usr/cad/synopsys/verdi/cur/share/PLI/VCS/LINUX64/pli.a

// // For ncverilog
// +access+r -timescale 1ns/10ps -notimingcheck 
// -y /home/cad/synopsys/synthesis/2021.06-sp2/dw/sim_ver +libext+.v
// +incdir+/home/cad/synopsys/synthesis/2021.06-sp2/dw/sim_ver+