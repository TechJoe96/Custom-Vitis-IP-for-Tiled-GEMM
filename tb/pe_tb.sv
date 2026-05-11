`timescale 1ns/1ps

module pe_tb;
    logic               clk;
    logic               rst;

    logic               load_weight;

    logic signed [15:0] a_in;
    logic signed [15:0] b_in;
    logic signed [31:0] c_in;

    logic signed [15:0] a_out;
    logic signed [31:0] c_out;

    pe dut (
        .clk         (clk),
        .rst         (rst),
        .load_weight (load_weight),
        .a_in        (a_in),
        .b_in        (b_in),
        .c_in        (c_in),
        .a_out       (a_out),
        .c_out       (c_out)
    );

    always #5 clk = ~clk;

    initial begin
        // For VCD files
        // $dumpfile("pe_tb.vcd");
        // $dumpvars(0, pe_tb);
        // Initialize all signals
        clk         = 0;
        rst         = 1;
        load_weight = 0;
        a_in        = 0;
        b_in        = 0;
        c_in        = 0;

// Hold reset for 3 clock cycles
        repeat (3) @(posedge clk);
        rst = 0;

// Load weight = 1
        load_weight = 1;    // flag
        b_in        = 3;
        @(posedge clk);     // wait for the next cycle
        load_weight = 0;

// Test 1: weight=3, a_in=5, c_in=10  →  expect c_out=25, a_out=5
        a_in = 5;
        c_in = 10;
        @(posedge clk); #1;
        assert (a_out === 16'sd5)  else $error("Test 1: a_out=%0d, expected 5",  a_out);
        assert (c_out === 32'sd25) else $error("Test 1: c_out=%0d, expected 25", c_out);
        $display("PASS Test 1: a_out=%0d, c_out=%0d", a_out, c_out);

// Test 2: a_in=7, c_in=25  →  expect c_out=46, a_out=7  (chains from test 1's c_out)
        a_in = 7;
        c_in = 25;
        @(posedge clk); #1;
        assert (a_out === 16'sd7)  else $error("Test 2: a_out=%0d, expected 7",  a_out);
        assert (c_out === 32'sd46) else $error("Test 2: c_out=%0d, expected 46", c_out);
        $display("PASS Test 2: a_out=%0d, c_out=%0d", a_out, c_out);

// Test 3: negative input  →  verify signed multiplication
        a_in = -2;
        c_in = 0;
        @(posedge clk); #1;
        assert (a_out === -16'sd2) else $error("Test 3: a_out=%0d, expected -2", a_out);
        assert (c_out === -32'sd6) else $error("Test 3: c_out=%0d, expected -6", c_out);
        $display("PASS Test 3: a_out=%0d, c_out=%0d", a_out, c_out);

        $display("All tests passed");
        $finish;
    end
endmodule



