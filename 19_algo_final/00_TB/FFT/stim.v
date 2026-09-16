//----------------------------------------------------------------------
//	Test Stimuli
//----------------------------------------------------------------------
initial begin : STIM
	wait (i_rst_n == 0);   //  wait reset assert  (active-low)
	wait (i_rst_n == 1);   //  wait reset release
	repeat(10) @(posedge i_clk);
	fork
		begin
			LoadInputData("input4.txt");
			GenerateInputWave;
		//	@(posedge i_clk);
			LoadInputData("input5.txt");
			GenerateInputWave;
		end
		
		begin
			wait (do_en == 1);
			repeat(N) @(posedge i_clk);
			SaveOutputData("output4.txt");
			@(negedge i_clk);
			wait (do_en == 1);
			repeat(N) @(posedge i_clk);
			SaveOutputData("output5.txt");
		end
	join

	repeat(10) @(posedge i_clk);
	$finish;
end
initial begin : TIMEOUT
	repeat(1000) #20;	//  1000 Clock Cycle Time
	$display("[FAILED] Simulation timed out.");
	$finish;
end

// integer cycle_count;

// initial begin : CYCLE_COUNT
// 	cycle_count = 0;
// 	wait(di_en == 1);
// 	while(do_en != 1) cycle_count = cycle_count + 1;
// 	$display("cycle_count: %d" , cycle_count);
// end
