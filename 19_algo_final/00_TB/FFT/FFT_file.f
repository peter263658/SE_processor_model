// Parameter / config. Definition
// -----------------------------------------------------------------------------
../../01_RTL/Define/Param_def.sv

// testbench
// -----------------------------------------------------------------------------
./TB128.v

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
../../01_RTL/engine/FFT/FFT128.sv
../../01_RTL/engine/FFT/SdfUnit.sv
../../01_RTL/engine/FFT/SdfUnit2.sv
../../01_RTL/engine/FFT/Butterfly.sv
../../01_RTL/engine/FFT/DelayBuffer.sv
../../01_RTL/engine/FFT/Multiply.sv
../../01_RTL/engine/FFT/Twiddle128.sv
// ../../01_RTL/engine/FFT/TwiddleConvert8.v


// For vcs
-debug_access+all +v2k +notimingcheck -sverilog -timescale=1ns/10ps 
-y /usr/cad/synopsys/synthesis/cur/dw/sim_ver +libext+.v
+incdir+/usr/cad/synopsys/synthesis/cur/dw/sim_ver+
-P /usr/cad/synopsys/verdi/cur/share/PLI/VCS/LINUX64/novas.tab
    /usr/cad/synopsys/verdi/cur/share/PLI/VCS/LINUX64/pli.a

// // For ncverilog
// +access+r -timescale 1ns/10ps -notimingcheck 
// -y /usr/cad/synopsys/synthesis/cur/dw/sim_ver +libext+.v
// +incdir+/usr/cad/synopsys/synthesis/cur/dw/sim_ver+