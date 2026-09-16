//----------------------------------------------------------------------
//	TB: FftTop Testbench
//----------------------------------------------------------------------
`timescale	1ns/1ns
module TB #(
	parameter	N = 128
);

localparam		NN = log2(N);	//	Count Bit Width of FFT Point

//	log2 constant function
function integer log2;
	input integer x;
	integer value;
	begin
		value = x-1;
		for (log2=0; value>0; log2=log2+1)
			value = value>>1;
	end
endfunction

//	Internal Regs and Nets
reg			i_clk;
reg			i_rst_n;
reg			di_en;
reg	[15:0]	di_re;
reg	[15:0]	di_im;
wire		do_en;
wire[15:0]	do_re;
wire[15:0]	do_im;

reg	[15:0]	imem[0:2*N-1];
reg	[15:0]	omem[0:2*N-1];

//----------------------------------------------------------------------
//	Clock and Reset
//----------------------------------------------------------------------
always begin
	i_clk = 0; #10;
	i_clk = 1; #10;
end

initial begin                 //  active-low async reset (assert = 0)
	i_rst_n = 1; #20;         //  inactive
	i_rst_n = 0; #100;        //  assert
	i_rst_n = 1;              //  release
end

//----------------------------------------------------------------------
//	Functional Blocks
//----------------------------------------------------------------------

//	Input Control Initialize
initial begin
	wait (i_rst_n == 0);
	di_en = 0;
end

//	Output Data Capture
initial begin : OCAP
	integer		n;
	forever begin
		n = 0;
		while (do_en !== 1) @(negedge i_clk);
		while ((do_en == 1) && (n < N)) begin
			omem[2*n  ] = do_re;
			omem[2*n+1] = do_im;
			n = n + 1;
			@(negedge i_clk);
		end
	end
end

//----------------------------------------------------------------------
//	Tasks
//----------------------------------------------------------------------
task LoadInputData;
	input[80*8:1]	filename;
begin
	$readmemh(filename, imem);
end
endtask

task GenerateInputWave;
	integer	n;
begin
	di_en <= 1;
	for (n = 0; n < N; n = n + 1) begin
		di_re <= imem[2*n];
		di_im <= imem[2*n+1];
		@(posedge i_clk);
	end
	di_en <= 0;
	di_re <= 'bx;
	di_im <= 'bx;
end
endtask

task SaveOutputData;
	input[80*8:1]	filename;
	integer			fp, n, m, i;
begin
	fp = $fopen(filename);
	m = 0;
	for (n = 0; n < N; n = n + 1) begin
		for (i = 0; i < NN; i = i + 1) m[NN-1-i] = n[i];
		$fdisplay(fp, "%h  %h  // %d", omem[2*m], omem[2*m+1], n[NN-1:0]);
		$display ("%h  %h  // %d", omem[2*m], omem[2*m+1], n[NN-1:0]);
	end
	$fclose(fp);
end
endtask

//----------------------------------------------------------------------
//	Module Instances
//----------------------------------------------------------------------
FFT128 U_FFT (
	.i_clk	(i_clk	),	//	i
	.i_rst_n	(i_rst_n	),	//	i
	.di_en	(di_en	),	//	i
	.di_re	(di_re	),	//	i
	.di_im	(di_im	),	//	i
	.do_en	(do_en	),	//	o
	.do_re	(do_re	),	//	o
	.do_im	(do_im	)	//	o
);

//----------------------------------------------------------------------
//	Include Stimuli
//----------------------------------------------------------------------
`include "stim.v"


initial begin 
	$fsdbDumpfile("tb_FFT.fsdb");
	$fsdbDumpvars(0, "+mda+packedOnly");
end

// integer cycle_count;

// initial begin : CYCLE_COUNT
//   cycle_count = 0;

//   @(posedge i_clk);      
//   wait (di_en);          


//   forever begin
//     @(posedge i_clk);
//     if (do_en) begin
//       $display("cycle_count = %0d", cycle_count);
//       disable CYCLE_COUNT;   
//     end
//     else
//       cycle_count = cycle_count + 1;
//   end
// end




endmodule
