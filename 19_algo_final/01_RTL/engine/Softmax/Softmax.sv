/*
    Module:     SoftMax function
    Note:       Fully pipelined design (with 2*NUM_PEG FP_EXP & FP_SUB, NUM_PEG FP_ADD),
                elements to compute depends on i_SM_ndat
                With output registers
*/


module Softmax
    import Param_def::*;
    import Mem_def::*;
(
    input   logic   i_clk,
    input   logic   i_rst_n,
    input   logic                                               i_SM_valid,
    input   logic   [MAX_SEQ_LEN-1 : 0] [FP_FULL_WIDTH-1   : 0] i_SM_data,
    input   logic                       [MAX_SEQ_LEN_BIT   : 0] i_SM_ndat,

    output  logic                                               o_SM_valid,
    output  logic   [MAX_SEQ_LEN-1 : 0] [FP_FULL_WIDTH-1   : 0] o_SM_data
);
    // Input Reg
        logic                                               i_SM_valid_w, i_SM_valid_r;
        logic   [MAX_SEQ_LEN-1 : 0] [FP_FULL_WIDTH-1   : 0] i_SM_data_w, i_SM_data_r;
        logic                       [MAX_SEQ_LEN_BIT   : 0] i_SM_ndat_w, i_SM_ndat_r;

        assign i_SM_valid_w = i_SM_valid;
        assign i_SM_data_w  = (i_SM_valid) ? i_SM_data : i_SM_data_r;
        assign i_SM_ndat_w  = (i_SM_valid) ? i_SM_ndat : i_SM_ndat_r;

        always_ff @(posedge i_clk) begin
            if(!i_rst_n) begin
                i_SM_valid_r <= 0;
                i_SM_data_r  <= 0;
                i_SM_ndat_r  <= 0;
            end else begin
                i_SM_valid_r <= i_SM_valid_w;
                i_SM_data_r  <= i_SM_data_w;
                i_SM_ndat_r  <= i_SM_ndat_w;
            end
        end
    //

    // Find X_max
        logic                                               i_SM_CP_valid;
        logic   [MAX_SEQ_LEN-1 : 0] [FP_FULL_WIDTH-1   : 0] i_SM_CP_data;
        logic                       [MAX_SEQ_LEN_BIT   : 0] i_SM_CP_ndat;
        logic                                               o_SM_CP_valid;
        logic                       [FP_FULL_WIDTH-1   : 0] o_SM_CP_data;
        logic                       [MAX_SEQ_LEN_BIT   : 0] o_SM_CP_ndat;
        logic                                               o_SM_CP_fvld;
        assign	i_SM_CP_valid   = i_SM_valid_r;
        assign  i_SM_CP_data    = i_SM_data_r;
        assign  i_SM_CP_ndat    = i_SM_ndat_r;
        SM_COMP u_SM_COMP(.*);

        logic                                               i_SM_F_fvld0;
        logic                                               i_SM_F_valid0;
        logic   [MAX_SEQ_LEN-1 : 0] [FP_FULL_WIDTH-1   : 0] i_SM_F_data0;
        logic                       [MAX_SEQ_LEN_BIT   : 0] i_SM_F_ndat0;
        logic                                               o_SM_F_valid0;
        logic   [MAX_SEQ_LEN-1 : 0] [FP_FULL_WIDTH-1   : 0] o_SM_F_data0;
        logic                       [MAX_SEQ_LEN_BIT   : 0] o_SM_F_ndat0;
        assign  i_SM_F_fvld0    = o_SM_CP_fvld;
        assign  i_SM_F_valid0   = i_SM_valid_r;
        assign  i_SM_F_data0    = i_SM_data_r;
        assign  i_SM_F_ndat0    = i_SM_ndat_r;
        SM_FIFO u_SM_FIFO0(
        	.i_clk        (i_clk        ),
            .i_rst_n      (i_rst_n      ),
            .i_SM_F_fvld  (i_SM_F_fvld0 ),
            .i_SM_F_valid (i_SM_F_valid0),
            .i_SM_F_data  (i_SM_F_data0 ),
            .i_SM_F_ndat  (i_SM_F_ndat0 ),
            .o_SM_F_valid (o_SM_F_valid0),
            .o_SM_F_data  (o_SM_F_data0 ),
            .o_SM_F_ndat  (o_SM_F_ndat0 )
        );
    
    // X-X_max
        logic                                               i_SM_SUB_valid0;
        logic   [MAX_SEQ_LEN-1 : 0] [FP_FULL_WIDTH-1   : 0] i_SM_SUB_data0;
        logic                       [FP_FULL_WIDTH-1   : 0] i_SM_SUB_dsub0;
        logic                       [MAX_SEQ_LEN_BIT   : 0] i_SM_SUB_ndat0;
        logic                                               o_SM_SUB_valid0;
        logic   [MAX_SEQ_LEN-1 : 0] [FP_FULL_WIDTH-1   : 0] o_SM_SUB_data0;
        logic                       [MAX_SEQ_LEN_BIT   : 0] o_SM_SUB_ndat0;
        assign  i_SM_SUB_valid0 = o_SM_CP_valid;
        assign  i_SM_SUB_data0  = o_SM_F_data0;
        assign  i_SM_SUB_dsub0  = o_SM_CP_data;
        assign  i_SM_SUB_ndat0  = o_SM_F_ndat0;
        SM_SUB u_SM_SUB0(
            .i_clk          (i_clk          ),
            .i_rst_n        (i_rst_n        ),
            .i_SM_SUB_valid (i_SM_SUB_valid0),
            .i_SM_SUB_data  (i_SM_SUB_data0 ),
            .i_SM_SUB_dsub  (i_SM_SUB_dsub0 ),
            .i_SM_SUB_ndat  (i_SM_SUB_ndat0 ),
            .o_SM_SUB_valid (o_SM_SUB_valid0),
            .o_SM_SUB_data  (o_SM_SUB_data0 ),
            .o_SM_SUB_ndat  (o_SM_SUB_ndat0 )
        );  

    // Exp(X-Xmax)
        logic                                               i_SM_EXP_valid0;
        logic   [MAX_SEQ_LEN-1 : 0] [FP_FULL_WIDTH-1   : 0] i_SM_EXP_data0;
        logic                       [MAX_SEQ_LEN_BIT   : 0] i_SM_EXP_ndat0;
        logic                                               o_SM_EXP_valid0;
        logic   [MAX_SEQ_LEN-1 : 0] [FP_FULL_WIDTH-1   : 0] o_SM_EXP_data0;
        logic                       [MAX_SEQ_LEN_BIT   : 0] o_SM_EXP_ndat0;
        assign  i_SM_EXP_valid0  = o_SM_SUB_valid0;
        assign  i_SM_EXP_data0   = o_SM_SUB_data0;
        assign  i_SM_EXP_ndat0   = o_SM_SUB_ndat0;
        SM_EXP u_SM_EXP0(
        	.i_clk          (i_clk          ),
            .i_rst_n        (i_rst_n        ),
            .i_SM_EXP_valid (i_SM_EXP_valid0),
            .i_SM_EXP_data  (i_SM_EXP_data0 ),
            .i_SM_EXP_ndat  (i_SM_EXP_ndat0 ),
            .o_SM_EXP_valid (o_SM_EXP_valid0),
            .o_SM_EXP_data  (o_SM_EXP_data0 ),
            .o_SM_EXP_ndat  (o_SM_EXP_ndat0 )
        );   
        
    // Sum and Ln
        logic                                               i_SM_SL_valid;
        logic   [MAX_SEQ_LEN-1 : 0] [FP_FULL_WIDTH-1   : 0] i_SM_SL_data;
        logic                       [MAX_SEQ_LEN_BIT   : 0] i_SM_SL_ndat;
        logic                                               o_SM_SL_valid;
        logic                       [FP_FULL_WIDTH-1   : 0] o_SM_SL_data;
        logic                       [MAX_SEQ_LEN_BIT   : 0] o_SM_SL_ndat;
        logic                                               o_SM_SL_fvld;
        assign	i_SM_SL_valid   = o_SM_EXP_valid0;
        assign	i_SM_SL_data    = o_SM_EXP_data0;
        assign	i_SM_SL_ndat    = o_SM_EXP_ndat0;
        SM_SUM_LN u_SM_SUM_LN(.*);

        logic                                               i_SM_F_fvld1;
        logic                                               i_SM_F_valid1;
        logic   [MAX_SEQ_LEN-1 : 0] [FP_FULL_WIDTH-1   : 0] i_SM_F_data1;
        logic                       [MAX_SEQ_LEN_BIT   : 0] i_SM_F_ndat1;
        logic                                               o_SM_F_valid1;
        logic   [MAX_SEQ_LEN-1 : 0] [FP_FULL_WIDTH-1   : 0] o_SM_F_data1;
        logic                       [MAX_SEQ_LEN_BIT   : 0] o_SM_F_ndat1;
        assign  i_SM_F_fvld1    = o_SM_SL_fvld;
        assign  i_SM_F_valid1   = o_SM_SUB_valid0;
        assign  i_SM_F_data1    = o_SM_SUB_data0;
        assign  i_SM_F_ndat1    = o_SM_SUB_ndat0;
        SM_FIFO u_SM_FIFO1(
        	.i_clk        (i_clk        ),
            .i_rst_n      (i_rst_n      ),
            .i_SM_F_fvld  (i_SM_F_fvld1 ),
            .i_SM_F_valid (i_SM_F_valid1),
            .i_SM_F_data  (i_SM_F_data1 ),
            .i_SM_F_ndat  (i_SM_F_ndat1 ),
            .o_SM_F_valid (o_SM_F_valid1),
            .o_SM_F_data  (o_SM_F_data1 ),
            .o_SM_F_ndat  (o_SM_F_ndat1 )
        );

    // X-X_max - Ln(Sum)
        logic                                               i_SM_SUB_valid1;
        logic   [MAX_SEQ_LEN-1 : 0] [FP_FULL_WIDTH-1   : 0] i_SM_SUB_data1;
        logic                       [FP_FULL_WIDTH-1   : 0] i_SM_SUB_dsub1;
        logic                       [MAX_SEQ_LEN_BIT   : 0] i_SM_SUB_ndat1;
        logic                                               o_SM_SUB_valid1;
        logic   [MAX_SEQ_LEN-1 : 0] [FP_FULL_WIDTH-1   : 0] o_SM_SUB_data1;
        logic                       [MAX_SEQ_LEN_BIT   : 0] o_SM_SUB_ndat1;
        assign  i_SM_SUB_valid1 = o_SM_SL_valid;
        assign  i_SM_SUB_data1  = o_SM_F_data1;
        assign  i_SM_SUB_dsub1  = o_SM_SL_data;
        assign  i_SM_SUB_ndat1  = o_SM_F_ndat1;
        SM_SUB u_SM_SUB1(
            .i_clk          (i_clk          ),
            .i_rst_n        (i_rst_n        ),
            .i_SM_SUB_valid (i_SM_SUB_valid1),
            .i_SM_SUB_data  (i_SM_SUB_data1 ),
            .i_SM_SUB_dsub  (i_SM_SUB_dsub1 ),
            .i_SM_SUB_ndat  (i_SM_SUB_ndat1 ),
            .o_SM_SUB_valid (o_SM_SUB_valid1),
            .o_SM_SUB_data  (o_SM_SUB_data1 ),
            .o_SM_SUB_ndat  (o_SM_SUB_ndat1 )
        ); 

    // Exp
        logic                                               i_SM_EXP_valid1;
        logic   [MAX_SEQ_LEN-1 : 0] [FP_FULL_WIDTH-1   : 0] i_SM_EXP_data1;
        logic                       [MAX_SEQ_LEN_BIT   : 0] i_SM_EXP_ndat1;
        logic                                               o_SM_EXP_valid1;
        logic   [MAX_SEQ_LEN-1 : 0] [FP_FULL_WIDTH-1   : 0] o_SM_EXP_data1;
        logic                       [MAX_SEQ_LEN_BIT   : 0] o_SM_EXP_ndat1;
        assign  i_SM_EXP_valid1   = o_SM_SUB_valid1;
        assign  i_SM_EXP_data1   = o_SM_SUB_data1;
        assign  i_SM_EXP_ndat1   = o_SM_SUB_ndat1;
        SM_EXP u_SM_EXP1(
        	.i_clk          (i_clk          ),
            .i_rst_n        (i_rst_n        ),
            .i_SM_EXP_valid (i_SM_EXP_valid1),
            .i_SM_EXP_data  (i_SM_EXP_data1 ),
            .i_SM_EXP_ndat  (i_SM_EXP_ndat1 ),
            .o_SM_EXP_valid (o_SM_EXP_valid1),
            .o_SM_EXP_data  (o_SM_EXP_data1 ),
            .o_SM_EXP_ndat  (o_SM_EXP_ndat1 )
        );
    
    assign	o_SM_valid  = o_SM_EXP_valid1;
    assign  o_SM_data   = o_SM_EXP_data1;

    
endmodule

// Pipelined submodules

module SM_SUB
    import Param_def::*;
(
    input   logic   i_clk,
    input   logic   i_rst_n,
    input   logic                                               i_SM_SUB_valid,
    input   logic   [MAX_SEQ_LEN-1 : 0] [FP_FULL_WIDTH-1   : 0] i_SM_SUB_data,
    input   logic                       [FP_FULL_WIDTH-1   : 0] i_SM_SUB_dsub,  // data to be sub.
    input   logic                       [MAX_SEQ_LEN_BIT   : 0] i_SM_SUB_ndat,
    output  logic                                               o_SM_SUB_valid,
    output  logic   [MAX_SEQ_LEN-1 : 0] [FP_FULL_WIDTH-1   : 0] o_SM_SUB_data,
    output  logic                       [MAX_SEQ_LEN_BIT   : 0] o_SM_SUB_ndat
);
    // Input reg
        logic                                               i_SM_SUB_valid_w, i_SM_SUB_valid_r;
        logic                       [FP_FULL_WIDTH-1   : 0] i_SM_SUB_dsub_w, i_SM_SUB_dsub_r;
        logic                       [MAX_SEQ_LEN_BIT   : 0] i_SM_SUB_ndat_w, i_SM_SUB_ndat_r;

        assign  i_SM_SUB_valid_w = i_SM_SUB_valid;
        assign  i_SM_SUB_dsub_w  = (i_SM_SUB_valid) ? i_SM_SUB_dsub : i_SM_SUB_dsub_r;
        assign  i_SM_SUB_ndat_w  = (i_SM_SUB_valid) ? i_SM_SUB_ndat : i_SM_SUB_ndat_r;

        always_ff @( posedge i_clk or negedge i_rst_n ) begin
            if(!i_rst_n) begin
                i_SM_SUB_valid_r <= 0;
                i_SM_SUB_dsub_r  <= 0;
                i_SM_SUB_ndat_r  <= 0;
            end else begin
                i_SM_SUB_valid_r <= i_SM_SUB_valid_w;
                i_SM_SUB_dsub_r  <= i_SM_SUB_dsub_w;
                i_SM_SUB_ndat_r  <= i_SM_SUB_ndat_w;
            end
        end
    //

    // Output reg
        logic                                               o_SM_SUB_valid_w, o_SM_SUB_valid_r;
        logic   [MAX_SEQ_LEN-1 : 0] [FP_FULL_WIDTH-1   : 0] o_SM_SUB_data_w, o_SM_SUB_data_r;
        logic                       [MAX_SEQ_LEN_BIT   : 0] o_SM_SUB_ndat_w, o_SM_SUB_ndat_r;

        assign  o_SM_SUB_valid = o_SM_SUB_valid_r;
        assign  o_SM_SUB_data  = o_SM_SUB_data_r;
        assign	o_SM_SUB_ndat  = o_SM_SUB_ndat_r;

        always_ff @( posedge i_clk or negedge i_rst_n ) begin
            if(!i_rst_n) begin
                o_SM_SUB_valid_r <= 0;
                o_SM_SUB_data_r  <= 0;
                o_SM_SUB_ndat_r  <= 0;
            end else begin
                o_SM_SUB_valid_r <= o_SM_SUB_valid_w;
                o_SM_SUB_data_r  <= o_SM_SUB_data_w;
                o_SM_SUB_ndat_r  <= o_SM_SUB_ndat_w;
            end
        end
    //

    // FSM
        typedef enum logic {
            SM_SUB_IDLE,
            SM_SUB_CALC
        } SM_SUB_mode;

        SM_SUB_mode SM_SUB_mode_w, SM_SUB_mode_r;
        logic [MAX_SEQ_LEN_BIT : 0] ndat_acc_w, ndat_acc_r; // Accumulator for number of calculated data

        always_comb begin
            case(SM_SUB_mode_r)
                SM_SUB_IDLE: begin
                    // Next state
                    SM_SUB_mode_w = (i_SM_SUB_valid_r) ? SM_SUB_CALC : SM_SUB_IDLE;
                    // Support logic
                    ndat_acc_w = 0;
                end
                SM_SUB_CALC: begin
                    // Next state
                    SM_SUB_mode_w = (ndat_acc_r + NUM_PEG >= i_SM_SUB_ndat_r) ? SM_SUB_IDLE : SM_SUB_CALC;
                    // Support logic
                    ndat_acc_w = ndat_acc_r + NUM_PEG;
                end
                default: begin
                    // Next state
                    SM_SUB_mode_w = SM_SUB_IDLE;
                    // Support logic
                    ndat_acc_w = 0;
                end
            endcase
        end

        always_ff @( posedge i_clk or negedge i_rst_n ) begin
            if(!i_rst_n) begin
                SM_SUB_mode_r <= SM_SUB_IDLE;
                ndat_acc_r    <= 0;
            end else begin
                SM_SUB_mode_r <= SM_SUB_mode_w;
                ndat_acc_r    <= ndat_acc_w;
            end
        end
    // 

    // Instantiate DW FP modules
        parameter ADDR = 64;
        logic   [ADDR-1 : 0] [NUM_PEG-1 : 0] [FP_FULL_WIDTH-1 : 0]  data_view;
        logic   [NUM_PEG-1 : 0] [FP_FULL_WIDTH-1 : 0]   i_sub;
        logic   [NUM_PEG-1 : 0] [FP_FULL_WIDTH-1 : 0]   o_sub;
        
        // Re-arrange data
        always_comb begin
            for (int i=0; i<ADDR; i++)
                data_view[i] = i_SM_SUB_data[NUM_PEG*i +: NUM_PEG];

            i_sub = data_view[ndat_acc_r[MAX_SEQ_LEN_BIT : 2]];
        end

        generate
            for (genvar gi=0; gi<NUM_PEG; gi++) begin
                DW_fp_sub_inst #(
                    .sig_width       (FP_FRAC_WIDTH),
                    .exp_width       (FP_EXPN_WIDTH),
                    .ieee_compliance (1)
                )
                u_DW_fp_sub_inst(
                    .inst_a      (i_sub        [gi]),
                    .inst_b      (i_SM_SUB_dsub_r  ),
                    .inst_rnd    (3'b000           ),
                    .z_inst      (o_sub        [gi]),
                    .status_inst ()
                );
            end
        endgenerate
    //

    // Output assignment
        always_comb begin
            case(SM_SUB_mode_r)
                SM_SUB_IDLE: begin
                    o_SM_SUB_valid_w = 0;
                    o_SM_SUB_data_w  = o_SM_SUB_data_r;
                    o_SM_SUB_ndat_w  = (i_SM_SUB_valid_r) ? i_SM_SUB_ndat_r : 0;
                end
                SM_SUB_CALC: begin
                    o_SM_SUB_valid_w = (ndat_acc_r + NUM_PEG >= i_SM_SUB_ndat_r) ? 1 : 0;

                    o_SM_SUB_data_w  = o_SM_SUB_data_r;
                    for (int i=0; i<NUM_PEG; i++)
                        o_SM_SUB_data_w[ndat_acc_r + i] = o_sub[i];
                    
                    o_SM_SUB_ndat_w = o_SM_SUB_ndat_r;
                end
                default: begin
                    o_SM_SUB_valid_w = 0;
                    o_SM_SUB_data_w  = 0;
                    o_SM_SUB_ndat_w  = 0;
                end
            endcase
        end
    //
endmodule

module SM_EXP
    import Param_def::*;
(
    input   logic   i_clk,
    input   logic   i_rst_n,
    input   logic                                               i_SM_EXP_valid,
    input   logic   [MAX_SEQ_LEN-1 : 0] [FP_FULL_WIDTH-1   : 0] i_SM_EXP_data,
    input   logic                       [MAX_SEQ_LEN_BIT   : 0] i_SM_EXP_ndat,
    output  logic                                               o_SM_EXP_valid,
    output  logic   [MAX_SEQ_LEN-1 : 0] [FP_FULL_WIDTH-1   : 0] o_SM_EXP_data,
    output  logic                       [MAX_SEQ_LEN_BIT   : 0] o_SM_EXP_ndat
);
    // Input reg
        logic                                               i_SM_EXP_valid_w, i_SM_EXP_valid_r;
        logic                       [MAX_SEQ_LEN_BIT   : 0] i_SM_EXP_ndat_w, i_SM_EXP_ndat_r;

        assign  i_SM_EXP_valid_w = i_SM_EXP_valid;
        assign  i_SM_EXP_ndat_w  = (i_SM_EXP_valid) ? i_SM_EXP_ndat : i_SM_EXP_ndat_r;

        always_ff @( posedge i_clk or negedge i_rst_n ) begin
            if(!i_rst_n) begin
                i_SM_EXP_valid_r <= 0;
                i_SM_EXP_ndat_r  <= 0;
            end else begin
                i_SM_EXP_valid_r <= i_SM_EXP_valid_w;
                i_SM_EXP_ndat_r  <= i_SM_EXP_ndat_w;
            end
        end
    //

    // Output reg
        logic                                               o_SM_EXP_valid_w, o_SM_EXP_valid_r;
        logic   [MAX_SEQ_LEN-1 : 0] [FP_FULL_WIDTH-1   : 0] o_SM_EXP_data_w, o_SM_EXP_data_r;
        logic                       [MAX_SEQ_LEN_BIT   : 0] o_SM_EXP_ndat_w, o_SM_EXP_ndat_r;

        assign  o_SM_EXP_valid = o_SM_EXP_valid_r;
        assign  o_SM_EXP_data  = o_SM_EXP_data_r;
        assign	o_SM_EXP_ndat  = o_SM_EXP_ndat_r;

        always_ff @( posedge i_clk or negedge i_rst_n ) begin
            if(!i_rst_n) begin
                o_SM_EXP_valid_r <= 0;
                o_SM_EXP_data_r  <= 0;
                o_SM_EXP_ndat_r  <= 0;
            end else begin
                o_SM_EXP_valid_r <= o_SM_EXP_valid_w;
                o_SM_EXP_data_r  <= o_SM_EXP_data_w;
                o_SM_EXP_ndat_r  <= o_SM_EXP_ndat_w;
            end
        end
    // 

    // FSM
        typedef enum logic {
            SM_EXP_IDLE,
            SM_EXP_CALC
        } SM_EXP_mode;

        SM_EXP_mode SM_EXP_mode_w, SM_EXP_mode_r;
        logic [MAX_SEQ_LEN_BIT : 0] ndat_acc_w, ndat_acc_r; // Accumulator for number of calculated data

        always_comb begin
            case(SM_EXP_mode_r)
                SM_EXP_IDLE: begin
                    // Next state
                    SM_EXP_mode_w = (i_SM_EXP_valid_r) ? SM_EXP_CALC : SM_EXP_IDLE;
                    // Support logic
                    ndat_acc_w = 0;
                end
                SM_EXP_CALC: begin
                    // Next state
                    SM_EXP_mode_w = (ndat_acc_r + NUM_PEG >= i_SM_EXP_ndat_r) ? SM_EXP_IDLE : SM_EXP_CALC;
                    // Support logic
                    ndat_acc_w = ndat_acc_r + NUM_PEG;
                end
                default: begin
                    // Next state
                    SM_EXP_mode_w = SM_EXP_IDLE;
                    // Support logic
                    ndat_acc_w = 0;
                end
            endcase
        end

        always_ff @( posedge i_clk or negedge i_rst_n ) begin
            if(!i_rst_n) begin
                SM_EXP_mode_r <= SM_EXP_IDLE;
                ndat_acc_r    <= 0;
            end else begin
                SM_EXP_mode_r <= SM_EXP_mode_w;
                ndat_acc_r    <= ndat_acc_w;
            end
        end
    //

    // Instantiate DW FP modules
        parameter ADDR = 64;
        logic   [ADDR-1 : 0] [NUM_PEG-1 : 0] [FP_FULL_WIDTH-1 : 0]  data_view;
        logic   [NUM_PEG-1 : 0] [FP_FULL_WIDTH-1 : 0]   i_exp;
        logic   [NUM_PEG-1 : 0] [FP_FULL_WIDTH-1 : 0]   o_exp;

        // Re-arrange data
        always_comb begin
            for (int i=0; i<ADDR; i++)
                data_view[i] = i_SM_EXP_data[NUM_PEG*i +: NUM_PEG];

            i_exp = data_view[ndat_acc_r[MAX_SEQ_LEN_BIT : 2]];
        end

        generate
            for (genvar gi=0; gi<NUM_PEG; gi++) begin
                DW_fp_exp_inst #(
                    .inst_sig_width (FP_FRAC_WIDTH),
                    .inst_exp_width (FP_EXPN_WIDTH),
                    .inst_ieee_compliance (1)
                )
                u_DW_fp_exp_inst(
                	.inst_a      (i_exp        [gi]),
                    .z_inst      (o_exp        [gi]),
                    .status_inst (                 )
                );
            end
        endgenerate
    //

    // Output assignment
        always_comb begin
            case(SM_EXP_mode_r)
                SM_EXP_IDLE: begin
                    o_SM_EXP_valid_w = 0;
                    o_SM_EXP_data_w  = o_SM_EXP_data_r;
                    o_SM_EXP_ndat_w  = (i_SM_EXP_valid_r) ? i_SM_EXP_ndat_r : 0;
                end
                SM_EXP_CALC: begin
                    o_SM_EXP_valid_w = (ndat_acc_r + NUM_PEG >= i_SM_EXP_ndat_r) ? 1 : 0;

                    o_SM_EXP_data_w  = o_SM_EXP_data_r;
                    for (int i=0; i<NUM_PEG; i++)
                        o_SM_EXP_data_w[ndat_acc_r + i] = o_exp[i];
                    
                    o_SM_EXP_ndat_w  = o_SM_EXP_ndat_r;
                end
                default: begin
                    o_SM_EXP_valid_w = 0;
                    o_SM_EXP_data_w  = 0;
                    o_SM_EXP_ndat_w  = 0;
                end
            endcase
        end
    //
endmodule

module SM_SUM_LN
    import Param_def::*;
(
    input   logic   i_clk,
    input   logic   i_rst_n,
    input   logic                                               i_SM_SL_valid,
    input   logic   [MAX_SEQ_LEN-1 : 0] [FP_FULL_WIDTH-1   : 0] i_SM_SL_data,
    input   logic                       [MAX_SEQ_LEN_BIT   : 0] i_SM_SL_ndat,
    output  logic                                               o_SM_SL_valid,
    output  logic                       [FP_FULL_WIDTH-1   : 0] o_SM_SL_data,
    output  logic                       [MAX_SEQ_LEN_BIT   : 0] o_SM_SL_ndat,
    output  logic                                               o_SM_SL_fvld    // signal to inform FIFO, a cycle before valid
);
    // Input reg
        logic                                               i_SM_SL_valid_w, i_SM_SL_valid_r;
        logic                       [MAX_SEQ_LEN_BIT   : 0] i_SM_SL_ndat_w, i_SM_SL_ndat_r;

        assign  i_SM_SL_valid_w = i_SM_SL_valid;
        assign  i_SM_SL_ndat_w  = (i_SM_SL_valid) ? i_SM_SL_ndat : i_SM_SL_ndat_r;

        always_ff @( posedge i_clk or negedge i_rst_n ) begin
            if(!i_rst_n) begin
                i_SM_SL_valid_r <= 0;
                i_SM_SL_ndat_r  <= 0;
            end else begin
                i_SM_SL_valid_r <= i_SM_SL_valid_w;
                i_SM_SL_ndat_r  <= i_SM_SL_ndat_w;
            end
        end
    //
    
    // Output reg
        logic                           o_SM_SL_valid_w, o_SM_SL_valid_r;
        logic   [FP_FULL_WIDTH-1   : 0] o_SM_SL_data_w, o_SM_SL_data_r;
        logic   [MAX_SEQ_LEN_BIT   : 0] o_SM_SL_ndat_w, o_SM_SL_ndat_r;
        logic                           o_SM_SL_fvld_w, o_SM_SL_fvld_r;

        assign  o_SM_SL_valid = o_SM_SL_valid_r;
        assign  o_SM_SL_data  = o_SM_SL_data_r;
        assign  o_SM_SL_ndat  = o_SM_SL_ndat_r;
        assign  o_SM_SL_fvld  = o_SM_SL_fvld_r;
        
        always_ff @( posedge i_clk or negedge i_rst_n ) begin
            if(!i_rst_n) begin
                o_SM_SL_valid_r <= 0;
                o_SM_SL_data_r  <= 0;
                o_SM_SL_ndat_r  <= 0;
                o_SM_SL_fvld_r  <= 0;
            end else begin
                o_SM_SL_valid_r <= o_SM_SL_valid_w;
                o_SM_SL_data_r  <= o_SM_SL_data_w;
                o_SM_SL_ndat_r  <= o_SM_SL_ndat_w;
                o_SM_SL_fvld_r  <= o_SM_SL_fvld_w;
            end
        end
    //

    // FSM
        typedef enum logic {
            SM_SL_IDLE,
            SM_SL_CALC
        } SM_SL_mode;

        SM_SL_mode SM_SL_mode_w, SM_SL_mode_r;
        logic [MAX_SEQ_LEN_BIT : 0] ndat_acc_w, ndat_acc_r; // Accumulator for number of calculated data

        always_comb begin
            case(SM_SL_mode_r)
                SM_SL_IDLE: begin
                    // Next state
                    SM_SL_mode_w = (i_SM_SL_valid_r) ? SM_SL_CALC : SM_SL_IDLE;
                    // Support logic
                    ndat_acc_w = 0;
                end
                SM_SL_CALC: begin
                    // Next state
                    SM_SL_mode_w = (o_SM_SL_valid_w) ? SM_SL_IDLE : SM_SL_CALC;
                    // Support logic
                    ndat_acc_w = ndat_acc_r + NUM_PEG;
                end
                default: begin
                    // Next state
                    SM_SL_mode_w = SM_SL_IDLE;
                    // Support logic
                    ndat_acc_w = 0;
                end
            endcase
        end

        always_ff @( posedge i_clk or negedge i_rst_n ) begin
            if(!i_rst_n) begin
                SM_SL_mode_r <= SM_SL_IDLE;
                ndat_acc_r   <= 0;
            end else begin
                SM_SL_mode_r <= SM_SL_mode_w;
                ndat_acc_r   <= ndat_acc_w;
            end
        end
    // 

    // Instantiate DW FP modules
        // Adder tree pipeline
        parameter ADDR = 64;
        logic   [ADDR-1 : 0] [NUM_PEG-1 : 0] [FP_FULL_WIDTH-1 : 0]  data_view;
        logic   [NUM_PEG-1    : 0] [FP_FULL_WIDTH-1 : 0] i_sl;
        logic   [NUM_PEG/2 -1 : 0] [FP_FULL_WIDTH-1 : 0] inter_dat0_w, inter_dat0_r;
        logic                      [FP_FULL_WIDTH-1 : 0] inter_dat1_w, inter_dat1_r;
        logic                      [FP_FULL_WIDTH-1 : 0] partial_sum_w, partial_sum_r;
        logic                      [FP_FULL_WIDTH-1 : 0] o_ln;
        logic   [2            : 0] [MAX_SEQ_LEN_BIT : 0] ndat_pipe_w, ndat_pipe_r;
        logic   [2            : 0]                       input_valid_pipe_w, input_valid_pipe_r;

        // Re-arrange data
        always_comb begin
            for (int i=0; i<ADDR; i++)
                data_view[i] = i_SM_SL_data[NUM_PEG*i +: NUM_PEG];

            i_sl = data_view[ndat_acc_r[MAX_SEQ_LEN_BIT : 2]];
        end

        generate
            for (genvar gi=0; gi<NUM_PEG/2  ; gi++) begin
                DW_fp_add_inst #(
                    .sig_width   (FP_FRAC_WIDTH),
                    .exp_width   (FP_EXPN_WIDTH),
                    .ieee_compliance (1)
                )u_DW_fp_add_inst(
                	.inst_a      (i_sl[gi            ]),
                    .inst_b      (i_sl[gi + NUM_PEG/2]),
                    .inst_rnd    (3'b000          ),
                    .z_inst      (inter_dat0_w[gi]),
                    .status_inst ()
                );
            end
        endgenerate

        DW_fp_add_inst #(
            .sig_width   (FP_FRAC_WIDTH),
            .exp_width   (FP_EXPN_WIDTH),
            .ieee_compliance (1)
        )u_DW_fp_add_inst0(
            .inst_a      (inter_dat0_r[0]),
            .inst_b      (inter_dat0_r[1]),
            .inst_rnd    (3'b000         ),
            .z_inst      (inter_dat1_w   ),
            .status_inst ()
        );

        DW_fp_add_inst #(
            .sig_width   (FP_FRAC_WIDTH),
            .exp_width   (FP_EXPN_WIDTH),
            .ieee_compliance (1)
        )u_DW_fp_add_inst1(
            .inst_a      (inter_dat1_r   ),
            .inst_b      (partial_sum_r  ),
            .inst_rnd    (3'b000         ),
            .z_inst      (partial_sum_w  ),
            .status_inst ()
        );

        DW_fp_ln_inst #(
            .inst_sig_width       (FP_FRAC_WIDTH),
            .inst_exp_width       (FP_EXPN_WIDTH),
            .inst_ieee_compliance (1)
        )u_DW_fp_ln_inst(
        	.inst_a      (partial_sum_r ),
            .z_inst      (o_ln          ),
            .status_inst ()
        );

        always_comb begin
            case (SM_SL_mode_r)
                SM_SL_IDLE: begin
                    ndat_pipe_w = 0;

                    input_valid_pipe_w[0] = i_SM_SL_valid_r;
                    input_valid_pipe_w[1] = 0;
                    input_valid_pipe_w[2] = 0;
                end
                SM_SL_CALC: begin
                    ndat_pipe_w[0] = ndat_acc_r;
                    ndat_pipe_w[1] = ndat_pipe_r[0];
                    ndat_pipe_w[2] = ndat_pipe_r[1];

                    input_valid_pipe_w[0] = input_valid_pipe_r[0];
                    input_valid_pipe_w[1] = input_valid_pipe_r[0];
                    input_valid_pipe_w[2] = input_valid_pipe_r[1];
                end
                default: begin
                    ndat_pipe_w = 0;
                    input_valid_pipe_w = 0;
                end
            endcase
        end

        always_ff @( posedge i_clk or negedge i_rst_n ) begin
            if(!i_rst_n) begin
                inter_dat0_r       <= 0;
                inter_dat1_r       <= 0;
                partial_sum_r      <= 0;
                ndat_pipe_r        <= 0;
                input_valid_pipe_r <= 0;
            end else begin
                inter_dat0_r       <= inter_dat0_w;
                inter_dat1_r       <= inter_dat1_w;
                partial_sum_r      <= partial_sum_w;
                ndat_pipe_r        <= ndat_pipe_w;
                input_valid_pipe_r <= input_valid_pipe_w;
            end
        end
    // 
        
    // Output assignment
        always_comb begin
            case(SM_SL_mode_r)
                SM_SL_IDLE: begin
                    o_SM_SL_valid_w = 0;
                    o_SM_SL_data_w  = 0;
                    o_SM_SL_ndat_w  = (i_SM_SL_valid_r) ? i_SM_SL_ndat_r : 0;
                    o_SM_SL_fvld_w  = 0;
                end
                SM_SL_CALC: begin
                    o_SM_SL_valid_w = (ndat_pipe_r[2] + NUM_PEG >= i_SM_SL_ndat_r) && (input_valid_pipe_r[2]) ? 1 : 0;
                    o_SM_SL_data_w  = o_ln;
                    o_SM_SL_ndat_w  = o_SM_SL_ndat_r;
                    o_SM_SL_fvld_w  = (ndat_pipe_r[1] + NUM_PEG >= i_SM_SL_ndat_r) && (input_valid_pipe_r[1]) ? 1 : 0;
                end
                default: begin
                    o_SM_SL_valid_w = 0;
                    o_SM_SL_data_w  = 0;
                    o_SM_SL_ndat_w  = 0;
                    o_SM_SL_fvld_w  = 0;
                end
            endcase
        end
    //
endmodule

module SM_COMP
    import Param_def::*;
(
    input   logic   i_clk,
    input   logic   i_rst_n,
    input   logic                                               i_SM_CP_valid,
    input   logic   [MAX_SEQ_LEN-1 : 0] [FP_FULL_WIDTH-1   : 0] i_SM_CP_data,
    input   logic                       [MAX_SEQ_LEN_BIT   : 0] i_SM_CP_ndat,
    output  logic                                               o_SM_CP_valid,
    output  logic                       [FP_FULL_WIDTH-1   : 0] o_SM_CP_data,
    output  logic                       [MAX_SEQ_LEN_BIT   : 0] o_SM_CP_ndat,
    output  logic                                               o_SM_CP_fvld    // signal to inform FIFO, a cycle before valid
);
    // Input reg
        logic                                               i_SM_CP_valid_w, i_SM_CP_valid_r;
        logic                       [MAX_SEQ_LEN_BIT   : 0] i_SM_CP_ndat_w, i_SM_CP_ndat_r;

        assign  i_SM_CP_valid_w = i_SM_CP_valid;
        assign  i_SM_CP_ndat_w  = (i_SM_CP_valid) ? i_SM_CP_ndat : i_SM_CP_ndat_r;

        always_ff @( posedge i_clk or negedge i_rst_n ) begin
            if(!i_rst_n) begin
                i_SM_CP_valid_r <= 0;
                i_SM_CP_ndat_r  <= 0;
            end else begin
                i_SM_CP_valid_r <= i_SM_CP_valid_w;
                i_SM_CP_ndat_r  <= i_SM_CP_ndat_w;
            end
        end
    //

    // Output reg
        logic                           o_SM_CP_valid_w, o_SM_CP_valid_r;
        logic   [FP_FULL_WIDTH-1   : 0] o_SM_CP_data_w, o_SM_CP_data_r;
        logic   [MAX_SEQ_LEN_BIT   : 0] o_SM_CP_ndat_w, o_SM_CP_ndat_r;
        logic                           o_SM_CP_fvld_w, o_SM_CP_fvld_r;

        assign  o_SM_CP_valid = o_SM_CP_valid_r;
        assign  o_SM_CP_data  = o_SM_CP_data_r;
        assign  o_SM_CP_ndat  = o_SM_CP_ndat_r;
        assign  o_SM_CP_fvld  = o_SM_CP_fvld_r;
        
        always_ff @( posedge i_clk or negedge i_rst_n ) begin
            if(!i_rst_n) begin
                o_SM_CP_valid_r <= 0;
                o_SM_CP_data_r  <= 0;
                o_SM_CP_ndat_r  <= 0;
                o_SM_CP_fvld_r  <= 0;
            end else begin
                o_SM_CP_valid_r <= o_SM_CP_valid_w;
                o_SM_CP_data_r  <= o_SM_CP_data_w;
                o_SM_CP_ndat_r  <= o_SM_CP_ndat_w;
                o_SM_CP_fvld_r  <= o_SM_CP_fvld_w;
            end
        end
    // 

    // FSM
        typedef enum logic {
            SM_CP_IDLE,
            SM_CP_CALC
        } SM_CP_mode;

        SM_CP_mode SM_CP_mode_w, SM_CP_mode_r;
        logic [MAX_SEQ_LEN_BIT : 0] ndat_acc_w, ndat_acc_r; // Accumulator for number of calculated data

        always_comb begin
            case(SM_CP_mode_r)
                SM_CP_IDLE: begin
                    // Next state
                    SM_CP_mode_w = (i_SM_CP_valid_r) ? SM_CP_CALC : SM_CP_IDLE;
                    // Support logic
                    ndat_acc_w = 0;
                end
                SM_CP_CALC: begin
                    // Next state
                    SM_CP_mode_w = (o_SM_CP_valid_w) ? SM_CP_IDLE : SM_CP_CALC;
                    // Support logic
                    ndat_acc_w = ndat_acc_r + NUM_PEG;
                end
                default: begin
                    // Next state
                    SM_CP_mode_w = SM_CP_IDLE;
                    // Support logic
                    ndat_acc_w = 0;
                end
            endcase
        end

        always_ff @( posedge i_clk or negedge i_rst_n ) begin
            if(!i_rst_n) begin
                SM_CP_mode_r <= SM_CP_IDLE;
                ndat_acc_r   <= 0;
            end else begin
                SM_CP_mode_r <= SM_CP_mode_w;
                ndat_acc_r   <= ndat_acc_w;
            end
        end
    //

    // Instantiate DW FP modules
        // Comparator tree pipeline
        parameter ADDR = 64;
        logic   [ADDR-1 : 0] [NUM_PEG-1 : 0] [FP_FULL_WIDTH-1 : 0]  data_view;
        logic   [NUM_PEG-1     : 0] [FP_FULL_WIDTH-1 : 0] i_cp;
        logic   [NUM_PEG/2  -1 : 0] [FP_FULL_WIDTH-1 : 0] inter_dat0_w, inter_dat0_r;
        logic                       [FP_FULL_WIDTH-1 : 0] inter_dat1_w, inter_dat1_r;
        logic                       [FP_FULL_WIDTH-1 : 0] current_max_w, current_max_r;
        logic   [2             : 0] [MAX_SEQ_LEN_BIT : 0] ndat_pipe_w, ndat_pipe_r;
        logic   [2             : 0]                       input_valid_pipe_w, input_valid_pipe_r;

        // Re-arrange data
        always_comb begin
            for (int i=0; i<ADDR; i++)
                data_view[i] = i_SM_CP_data[NUM_PEG*i +: NUM_PEG];

            i_cp = data_view[ndat_acc_r[MAX_SEQ_LEN_BIT : 2]];
        end

        generate
            for (genvar gi=0; gi<NUM_PEG/2  ; gi++) begin
                DW_fp_cmp_inst #(
                    .sig_width       (FP_FRAC_WIDTH),
                    .exp_width       (FP_EXPN_WIDTH),
                    .ieee_compliance (1)
                )u_DW_fp_cmp_inst(
                	.inst_a         (i_cp[gi            ]),
                    .inst_b         (i_cp[gi + NUM_PEG/2]),
                    .inst_zctr      (1'b0                ),
                    .aeqb_inst      (),
                    .altb_inst      (),
                    .agtb_inst      (),
                    .unordered_inst (),
                    .z0_inst        (),
                    .z1_inst        (inter_dat0_w[gi    ]),
                    .status0_inst   (),
                    .status1_inst   ()
                );
            end
        endgenerate

        DW_fp_cmp_inst #(
            .sig_width       (FP_FRAC_WIDTH),
            .exp_width       (FP_EXPN_WIDTH),
            .ieee_compliance (1)
        )u_DW_fp_cmp_inst0(
            .inst_a         (inter_dat0_r[0]),
            .inst_b         (inter_dat0_r[1]),
            .inst_zctr      (1'b0           ),
            .aeqb_inst      (),
            .altb_inst      (),
            .agtb_inst      (),
            .unordered_inst (),
            .z0_inst        (),
            .z1_inst        (inter_dat1_w   ),
            .status0_inst   (),
            .status1_inst   ()
        );

        DW_fp_cmp_inst #(
            .sig_width       (FP_FRAC_WIDTH),
            .exp_width       (FP_EXPN_WIDTH),
            .ieee_compliance (1)
        )u_DW_fp_cmp_inst1(
            .inst_a         (inter_dat1_r ),
            .inst_b         (current_max_r),
            .inst_zctr      (1'b0         ),
            .aeqb_inst      (),
            .altb_inst      (),
            .agtb_inst      (),
            .unordered_inst (),
            .z0_inst        (),
            .z1_inst        (current_max_w),
            .status0_inst   (),
            .status1_inst   ()
        );

        always_comb begin
            case (SM_CP_mode_r)
                SM_CP_IDLE: begin
                    ndat_pipe_w = 0;
                    input_valid_pipe_w[0] = i_SM_CP_valid_r;
                    input_valid_pipe_w[1] = 0;
                    input_valid_pipe_w[2] = 0;
                end
                SM_CP_CALC: begin
                    ndat_pipe_w[0] = ndat_acc_r;
                    ndat_pipe_w[1] = ndat_pipe_r[0];
                    ndat_pipe_w[2] = ndat_pipe_r[1];

                    input_valid_pipe_w[0] = input_valid_pipe_r[0];
                    input_valid_pipe_w[1] = input_valid_pipe_r[0];
                    input_valid_pipe_w[2] = input_valid_pipe_r[1];
                end
                default: begin
                    ndat_pipe_w = 0;
                    input_valid_pipe_w = 0;
                end
            endcase
        end

        always_ff @( posedge i_clk or negedge i_rst_n ) begin
            if(!i_rst_n) begin
                inter_dat0_r       <= 0;
                inter_dat1_r       <= 0;
                current_max_r      <= 0;
                ndat_pipe_r        <= 0;
                input_valid_pipe_r <= 0;
            end else begin
                inter_dat0_r       <= inter_dat0_w;
                inter_dat1_r       <= inter_dat1_w;
                current_max_r      <= current_max_w;
                ndat_pipe_r        <= ndat_pipe_w;
                input_valid_pipe_r <= input_valid_pipe_w;
            end
        end
    // 
        
    // Output assignment
        always_comb begin
            case(SM_CP_mode_r)
                SM_CP_IDLE: begin
                    o_SM_CP_valid_w = 0;
                    o_SM_CP_data_w  = 0;
                    o_SM_CP_ndat_w  = (i_SM_CP_valid_r) ? i_SM_CP_ndat_r : 0;
                    o_SM_CP_fvld_w  = 0;
                end
                SM_CP_CALC: begin
                    o_SM_CP_valid_w = (ndat_pipe_r[2] + NUM_PEG >= i_SM_CP_ndat_r) && (input_valid_pipe_r[2]) ? 1 : 0;
                    o_SM_CP_data_w  = current_max_r;
                    o_SM_CP_ndat_w  = o_SM_CP_ndat_r;
                    o_SM_CP_fvld_w  = (ndat_pipe_r[1] + NUM_PEG >= i_SM_CP_ndat_r) && (input_valid_pipe_r[1]) ? 1 : 0;
                end
                default: begin
                    o_SM_CP_valid_w = 0;
                    o_SM_CP_data_w  = 0;
                    o_SM_CP_ndat_w  = 0;
                    o_SM_CP_fvld_w  = 0;
                end
            endcase
        end
    //
endmodule

module SM_FIFO
    import Param_def::*;
(
    input   logic   i_clk,
    input   logic   i_rst_n,
    input   logic                                               i_SM_F_fvld,    // Inquiry for data
    input   logic                                               i_SM_F_valid,
    input   logic   [MAX_SEQ_LEN-1 : 0] [FP_FULL_WIDTH-1   : 0] i_SM_F_data,
    input   logic                       [MAX_SEQ_LEN_BIT   : 0] i_SM_F_ndat,
    output  logic                                               o_SM_F_valid,
    output  logic   [MAX_SEQ_LEN-1 : 0] [FP_FULL_WIDTH-1   : 0] o_SM_F_data,
    output  logic                       [MAX_SEQ_LEN_BIT   : 0] o_SM_F_ndat
);
    // Output reg
        logic   o_SM_F_valid_w, o_SM_F_valid_r;
        assign  o_SM_F_valid  = o_SM_F_valid_r;

        always_ff @( posedge i_clk or negedge i_rst_n ) begin
            if(!i_rst_n)    o_SM_F_valid_r   <= 0;
            else            o_SM_F_valid_r   <= o_SM_F_valid_w;
        end
    //

    // FIFO reg
        localparam PTR_BIT = $clog2(SM_FIFO_SIZE);
        logic   [PTR_BIT-1 : 0]                                                   W_ptr_w    , W_ptr_r;   // Write pointer
        logic   [PTR_BIT-1 : 0]                                                   R_ptr_w    , R_ptr_r;   // Read  pointer
        logic   [SM_FIFO_SIZE-1 : 0] [MAX_SEQ_LEN-1 : 0] [FP_FULL_WIDTH-1   : 0]  data_buff_w, data_buff_r;
        logic   [SM_FIFO_SIZE-1 : 0]                     [MAX_SEQ_LEN_BIT   : 0]  ndat_buff_w, ndat_buff_r;

        always_ff @( posedge i_clk or negedge i_rst_n ) begin
            if(!i_rst_n) begin
                W_ptr_r     <= 0;
                R_ptr_r     <= 0;
                data_buff_r <= 0;
                ndat_buff_r <= 0;
            end else begin
                W_ptr_r     <= W_ptr_w;
                R_ptr_r     <= R_ptr_w;
                data_buff_r <= data_buff_w;
                ndat_buff_r <= ndat_buff_w;
            end
        end
    // 
    
    always_comb begin
        o_SM_F_valid_w = 0;

        W_ptr_w     = W_ptr_r;
        R_ptr_w     = R_ptr_r;
        data_buff_w = data_buff_r;
        ndat_buff_w = ndat_buff_r;

        if (i_SM_F_valid)    W_ptr_w              = W_ptr_r + 1;
        if (i_SM_F_valid)    data_buff_w[W_ptr_r] = i_SM_F_data;
        if (i_SM_F_valid)    ndat_buff_w[W_ptr_r] = i_SM_F_ndat;

        if (i_SM_F_fvld)    o_SM_F_valid_w        = 1          ;
        if (o_SM_F_valid_r)  R_ptr_w              = R_ptr_r + 1;

        o_SM_F_data = data_buff_r[R_ptr_r];
        o_SM_F_ndat = ndat_buff_r[R_ptr_r];
    end
endmodule