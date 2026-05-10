`timescale 1ns/1ps
module systolic_array_tb;
parameter int N = 8;
// Signals connecting to the DUT
    logic clk;
    logic rst;
    logic load_weight;
    logic signed [15:0] a_in  [N];
    logic signed [15:0] b_in  [N][N];
    logic signed [31:0] c_out [N];

// Instantiate the systolic array under test
    systolic_array #(.N(N)) dut (
        .clk         (clk),
        .rst         (rst),
        .load_weight (load_weight),
        .a_in        (a_in),
        .b_in        (b_in),
        .c_out       (c_out)
    );

// 100 MHz clock
    always #5 clk = ~clk;

// Test data: flat 1D arrays, loaded from hex files
    logic signed [15:0] a_tile      [N*N];
    logic signed [15:0] b_tile      [N*N];
    logic signed [31:0] c_expected  [N*N];
    logic signed [31:0] c_collected [N*N];
    
    int errors;

initial begin
        // Initialize driven signals
        clk         = 0;
        rst         = 1;
        load_weight = 0;
        for (int k = 0; k < N; k++) a_in[k] = 0;
        for (int i = 0; i < N; i++)
            for (int j = 0; j < N; j++)
                b_in[i][j] = 0;
// Read test vectors from hex files
        $readmemh("../tb/data/a_tile.hex",     a_tile);
        $readmemh("../tb/data/b_tile.hex",     b_tile);
        $readmemh("../tb/data/c_expected.hex", c_expected);
        $display("Test vectors loaded.");
// Hold reset
        repeat (3) @(posedge clk);
        rst = 0;
        @(posedge clk);
// Weight load: all 64 B values latched in one cycle
        for (int i = 0; i < N; i++)
            for (int j = 0; j < N; j++)
                b_in[i][j] = b_tile[i*N + j];
        load_weight = 1;
        @(posedge clk);
        load_weight = 0;
// Stream A row by row for N cycles
        for (int r = 0; r < N; r++) begin
            for (int k = 0; k < N; k++) a_in[k] = a_tile[r*N + k];
            @(posedge clk);
        end
        for (int k = 0; k < N; k++) a_in[k] = 0;
// Wait for c_out to reach first valid value
        repeat (N - 1) @(posedge clk);
// Capture 8 rows of C output
        for (int r = 0; r < N; r++) begin
            #1;
            for (int j = 0; j < N; j++)
                c_collected[r*N + j] = c_out[j];
            
            $write("Row %0d:", r);
            for (int j = 0; j < N; j++) $write(" %6d", c_out[j]);
            $display("");
        if (r < N-1) @(posedge clk);
        end

// Compare against expected
        errors = 0;
        for (int idx = 0; idx < N*N; idx++) begin
            if (c_collected[idx] !== c_expected[idx]) begin
                $display("MISMATCH idx %0d (row %0d, col %0d): got %0d, expected %0d",
                         idx, idx/N, idx%N, c_collected[idx], c_expected[idx]);
                errors++;
            end
        end
        if (errors == 0)
            $display("PASS: all %0d C values match expected", N*N);
        else
            $display("FAIL: %0d mismatches out of %0d", errors, N*N);
$finish;
    end
endmodule
