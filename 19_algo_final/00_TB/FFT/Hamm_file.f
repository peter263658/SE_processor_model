// Parameter / config. Definition
// -----------------------------------------------------------------------------
../../01_RTL/Define/Param_def.sv
../../01_RTL/Define/Mem_def.sv

// testbench
// -----------------------------------------------------------------------------
./tb_hamming.sv


// RTL
../../01_RTL/engine/FFT/Hamming.sv

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