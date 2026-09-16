// W4A8 lane multiplier (INT4 x FP8 -> FP16) unit test
// -----------------------------------------------------------------------------

// testbench
./tb_mult_int4_fp8.sv

// design under test
../../01_RTL/engine/CE/mult_int4_fp8.sv

// VCS @ gpu01
// -debug_access+all +v2k +notimingcheck -sverilog -timescale=1ns/10ps
// -P /usr/cad/synopsys/verdi/cur/share/PLI/VCS/LINUX64/novas.tab
//    /usr/cad/synopsys/verdi/cur/share/PLI/VCS/LINUX64/pli.a
