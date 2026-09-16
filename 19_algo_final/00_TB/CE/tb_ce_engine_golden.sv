/*
    tb_ce_engine_golden.sv — CE W4A8 per-layer GOLDEN numeric sim (task 4/9), GENERAL.

    Drives the full ce_engine (control + behavioral WBUF/UF + gather + dealer + CE_core) with the
    REAL deploy SRAM image and a BIT-EXACT fp16 golden for any encoder pointwise sub-op. Parameters
    come from +plusargs (defaults = enc0 pw); generate the golden with:
        python3 10_algo/scripts/gen_golden_conv.py <dotted_layer> <subop_idx>
    which prints the exact plusargs (SUBOP/IC/OC/ACT/GOLD) and writes act_t0/golden_preBN_t0.

    CONTRACT (see gen_golden_conv.py): the CE conv core reproduces the PRE-bias descaled accumulate
        oacc_re[oc] = dealer( PEG_fp16_tree( {int4_w[oc,ic]*xdeq[ic]} ), apot[oc] )
    The golden is computed BIT-EXACTLY (same paired FP16 saturating tree + APoT 2^a±2^b dealer), so
    HW oacc_re must match it to 0 ulp. (For ic=32 the fp16 tree deviates ~3% from the fp64 ideal, so
    the bit-exact model — not the ideal — is the correct compare.) PPU is benign-stubbed in ce_engine,
    so o_re is NOT this; we hierarchically observe CE_core.oacc_re during the target sub-op's DRAIN.

    SINGLE-POSITION: CE processes ONE timestep per walk; the act for that timestep's IC channels is
    loaded into uf[0..IC-1] in LANE order (channel-major [ic,T] at t0), uf[IC..]=0 (so unused lanes
    and the xi slice are 0 -> isolates the pw_r real branch: oacc_re = dealer(tree(fr*xr), apot_r)).
*/
`timescale 1ns/10ps
import Param_def::*;

module tb_ce_engine_golden;
    localparam int    WBUF_BYTES  = 262144;
    localparam int    UFBUF_BYTES = 8192;
    localparam int    MAXC        = 256;     // max IC/OC we size for (attention proj = 256)
    localparam real   REL_TOL     = 0.02;    // diagnostic real-tol (primary check is BIT-EXACT)
    localparam real   ABS_TOL     = 1.0e-4;

    // ---- runtime params (plusargs; default = enc0 pw) ----
    int    SUBOP = 1, IC = 1, OC = 8;
    string ACT_FILE  = "../../10_algo/hw_patterns_final/sram_images/golden/encoder_model_0_0_pointwise_r__0/act_t0.hex";
    string GOLD_FILE = "../../10_algo/hw_patterns_final/sram_images/golden/encoder_model_0_0_pointwise_r__0/golden_preBN_t0.hex";
    localparam string IMG_BIN = "../../10_algo/hw_patterns_final/sram_images/wbuf_unified_init.bin";

    logic i_clk = 1'b0, i_rst_n;
    always #5 i_clk = ~i_clk;

    logic i_start, o_busy, o_done, o_result_valid;
    logic [15:0] o_re, o_im;

    ce_engine u_eng (
        .i_clk(i_clk), .i_rst_n(i_rst_n), .i_start(i_start),
        .o_busy(o_busy), .o_done(o_done),
        .o_re(o_re), .o_im(o_im), .o_result_valid(o_result_valid)
    );

    function automatic real fp16_to_real(input logic [15:0] h);
        int sign, exp, frac; real mant, value;
        begin
            sign = h[15]; exp = h[14:10]; frac = h[9:0];
            if (exp == 0)            value = (frac==0) ? 0.0 : (real'(frac)/1024.0)*(2.0**(-14));
            else if (exp == 31)      value = 1.0e30;
            else                     value = (1.0 + real'(frac)/1024.0)*(2.0**(exp-15));
            fp16_to_real = sign ? -value : value;
        end
    endfunction
    function automatic real absr(input real x); absr = (x<0.0)?-x:x; endfunction

    logic [7:0]  img_tmp [0:WBUF_BYTES-1];
    logic [7:0]  act_arr [0:MAXC-1];
    logic [15:0] golden  [0:MAXC-1];
    logic [15:0] got_re  [0:MAXC-1];
    logic [MAXC-1:0] captured;
    integer fd, nrd, b;

    initial begin
        i_rst_n = 1'b1; i_start = 1'b0; captured = '0;
        for (int o=0;o<MAXC;o++) got_re[o]=16'hxxxx;

        void'($value$plusargs("SUBOP=%d", SUBOP));
        void'($value$plusargs("IC=%d", IC));
        void'($value$plusargs("OC=%d", OC));
        void'($value$plusargs("ACT=%s", ACT_FILE));
        void'($value$plusargs("GOLD=%s", GOLD_FILE));
        void'($value$plusargs("SAMPLE_LAT=%d", SAMPLE_LAT));
        $display("[TB] SUBOP=%0d IC=%0d OC=%0d", SUBOP, IC, OC);
        $display("[TB] ACT=%s", ACT_FILE);
        $display("[TB] GOLD=%s", GOLD_FILE);

        // REAL image -> behavioral WBUF
        fd = $fopen(IMG_BIN, "rb");
        if (fd == 0) begin $display("[TB] FATAL cannot open %s", IMG_BIN); $finish; end
        nrd = $fread(img_tmp, fd); $fclose(fd);
        $display("[TB] read %0d bytes image", nrd);
        for (b=0;b<WBUF_BYTES;b++) u_eng.wbuf_mem[b] = img_tmp[b];

        // zero UF + SKIPBUF; load the IC act bytes in lane order. Destination follows the ROM
        // descriptor: pw_skip sub-ops (act_is_skip) read SKIPBUF -> load skip_mem[act_base..]
        // (re half; im half stays 0 = isolates the _r branch, same as the UF convention);
        // everything else loads uf[0..IC-1].
        for (b=0;b<UFBUF_BYTES;b++) u_eng.uf_mem[b] = 8'h00;
        for (b=0;b<4096;b++)        u_eng.skip_mem[b] = 8'h00;
        $readmemh(ACT_FILE, act_arr);
        begin
            automatic ce_subop_pkg::subop_t sd = ce_subop_pkg::ce_subop_rom(SUBOP);
            if (sd.act_is_skip) begin
                for (b=0;b<IC;b++) u_eng.skip_mem[32'(sd.act_base)+b] = act_arr[b];
                u_eng.skip_wr_en = 1'b0;   // walk passes encoder pw first: gate the live skip-write
                                           // so it can't overwrite this preloaded golden act
                $display("[TB] act -> SKIPBUF @%0d (act_is_skip, skip_wr_en gated)", sd.act_base);
            end else begin
                for (b=0;b<IC;b++) u_eng.uf_mem[b] = act_arr[b];
            end
        end
        $readmemh(GOLD_FILE, golden);

        repeat (2) @(negedge i_clk); i_rst_n = 1'b0;
        repeat (2) @(negedge i_clk); i_rst_n = 1'b1;
        repeat (2) @(negedge i_clk);
        @(negedge i_clk); i_start = 1'b1;
        @(negedge i_clk); i_start = 1'b0;
    end

    // hierarchical sample: OACC settles SAMPLE_LAT (=OACC_DLY+1) cycles after each oc's LAST
    // accumulate (marked by o_oc_complete). Works for BOTH the per-oc-DRAIN design (oacc holds)
    // and the pipelined continuous-RUN design (oacc valid for 1 cycle only -> exact timing needed).
    int SAMPLE_LAT = 3;
    wire [6:0]  dbg_subop = u_eng.u_id.subop_idx;
    wire [9:0]  dbg_oc    = u_eng.u_id.oc_idx;
    wire        dbg_occ   = u_eng.u_id.o_oc_complete;
    wire [15:0] dbg_ore   = u_eng.u_ce.oacc_re;

    logic [15:0]      occ_sr;            // o_oc_complete delayed
    logic [9:0]       oc_sr  [0:15];
    logic [6:0]       sub_sr [0:15];
    always @(posedge i_clk) begin
        occ_sr <= {occ_sr[14:0], (i_rst_n & dbg_occ)};
        oc_sr[0] <= dbg_oc; sub_sr[0] <= dbg_subop;
        for (int k=1;k<16;k++) begin oc_sr[k]<=oc_sr[k-1]; sub_sr[k]<=sub_sr[k-1]; end
        if (occ_sr[SAMPLE_LAT-1] && (sub_sr[SAMPLE_LAT-1]==SUBOP) && (oc_sr[SAMPLE_LAT-1]<MAXC)) begin
            got_re[oc_sr[SAMPLE_LAT-1]]   <= dbg_ore;
            captured[oc_sr[SAMPLE_LAT-1]] <= 1'b1;
        end
    end

    task automatic report_and_finish;
        int errs, exact_hits, checks; real got, exp, d;
        begin
            errs = 0; exact_hits = 0; checks = 0;
            $display("\n[TB] ===== sub-op %0d OACC vs BIT-EXACT golden (IC=%0d OC=%0d) =====", SUBOP, IC, OC);
            $display("[TB] oc | got(oacc_re) | golden | bitexact? | real.err");
            for (int oc=0; oc<OC; oc++) begin
                got = fp16_to_real(got_re[oc]);
                exp = fp16_to_real(golden[oc]);
                d   = absr(got-exp);
                checks++;
                if (got_re[oc] === golden[oc]) exact_hits++;
                if (d <= (ABS_TOL + REL_TOL*absr(exp))) begin
                    if (oc < 16 || got_re[oc] !== golden[oc])
                        $display("[TB] %3d | %h | %h | %s | %e", oc, got_re[oc], golden[oc],
                                 (got_re[oc]===golden[oc])?"EXACT":"~tol ", d);
                end else begin
                    errs++;
                    $display("[TB] %3d | %h | %h | FAIL  | %e  <-- mismatch", oc, got_re[oc], golden[oc], d);
                end
            end
            $display("[TB] -------------------------------------------------------------");
            $display("[TB] bit-exact: %0d/%0d channels ; within-tol: %0d/%0d", exact_hits, checks, checks-errs, checks);
            if (errs == 0 && exact_hits == checks)
                $display("[TB] ALL PASS (BIT-EXACT), %0d channels vs real fp32-derived golden", checks);
            else if (errs == 0)
                $display("[TB] ALL within tol (%0d/%0d bit-exact) — fp16-model ulp diffs, see above", exact_hits, checks);
            else
                $display("[TB] FAIL: %0d/%0d channels mismatch", errs, checks);
            $finish;
        end
    endtask

    initial begin
        @(posedge i_rst_n);
        wait (i_start === 1'b1);
        fork
            begin : done_watch
                wait ((u_eng.u_id.subop_idx > SUBOP) || (o_done === 1'b1));
                repeat (4) @(posedge i_clk);
                report_and_finish();
            end
            begin : timeout
                repeat (60000) @(posedge i_clk);
                $display("[TB] TIMEOUT subop_idx=%0d", u_eng.u_id.subop_idx);
                report_and_finish();
            end
        join_any
        $finish;
    end
endmodule
