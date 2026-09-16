// PEG (W4A8 Design-B) — group-level test: pe_pair (fused 2x8-bit slices)
//                        + 32-way FP16 saturating adder tree
// NOTE: packages MUST be compiled before any module that imports them.
// -----------------------------------------------------------------------------

// packages (first)
../../01_RTL/Define/Param_def.sv
../../01_RTL/Define/Mem_def.sv

// DesignWare FP16 adder wrapper (DW_fp_add via -y sim_ver)
../../01_RTL/common/Designware/DW_fp_add_inst.sv

// design under test
//   PEG.sv provides: PEG, adder_tree_fp16, fp16_add_sat, and shared helpers
//                    pe_fp_unpack / pe_int4_csd.
//   (pe_lane + pe_slice_addsub were MOVED to pe_lane_ref.sv — TB-only golden)
../../01_RTL/engine/CE/PEG.sv

// golden reference (TB-only): pe_lane + pe_slice_addsub (moved out of PEG.sv)
./pe_lane_ref.sv

// testbench (includes PEG_ref reference) — last
./tb_PEG_w4a8.sv
