// PEG i_preshift directed test — scale-equivariance (preshift=s == ldexp(preshift=0, s))
//                                + FTZ/saturating-tree boundary checks.
// No pe_lane reference here (golden has no preshift); this verifies preshift != 0.

// packages (first)
../../01_RTL/Define/Param_def.sv
../../01_RTL/Define/Mem_def.sv

// DesignWare FP16 adder wrapper (DW_fp_add via -y sim_ver)
../../01_RTL/common/Designware/DW_fp_add_inst.sv

// design under test: pe_pair (+ i_preshift) feeding PEG (32-way saturating tree)
../../01_RTL/engine/CE/PEG.sv

// testbench (last)
./tb_peg_preshift.sv
