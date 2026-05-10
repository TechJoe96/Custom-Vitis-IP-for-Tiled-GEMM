`timescale 1ns/1ps
module a_tile_buffer_tb;
parameter int WIDTH      = 16;
    parameter int N          = 8;
    parameter int ADDR_WIDTH = $clog2(N);
    logic clk;
    logic                       wr_en;
    logic [ADDR_WIDTH-1:0]      wr_addr;
    logic signed [WIDTH-1:0]    wr_data [N];
    logic [ADDR_WIDTH-1:0]      rd_addr;
    logic signed [WIDTH-1:0]    rd_data [N];
 
    a_tile_buffer #(.WIDTH (WIDTH), .N (N)) dut (
        .clk     (clk),
        .wr_en   (wr_en),
        .wr_addr (wr_addr),
        .wr_data (wr_data),
        .rd_addr (rd_addr),
        .rd_data (rd_data)
    );

    always #5 clk = ~clk;
    logic signed [WIDTH-1:0] expected [N][N];
    int errors;
    initial begin
        clk = 0; wr_en = 0; wr_addr = 0; rd_addr = 0;
        for (int k = 0; k < N; k++) wr_data[k] = 0;
        @(posedge clk);
        $display("Writing %0d rows (each with %0d K-positions)...", N, N);
        for (int r = 0; r < N; r++) begin
            for (int k = 0; k < N; k++) begin
                expected[r][k] = (r * 100) + k - 50;
                wr_data[k]     = expected[r][k];
            end
            wr_en   = 1;
            wr_addr = r;
            @(posedge clk);
        end
        wr_en = 0;
        for (int k = 0; k < N; k++) wr_data[k] = 0;
        @(posedge clk);
        $display("Reading %0d rows back, verifying all %0d K-positions per row...", N, N);
        errors = 0;
        for (int r = 0; r < N; r++) begin
            rd_addr = r;
            @(posedge clk); #1;
            for (int k = 0; k < N; k++) begin
                if (rd_data[k] !== expected[r][k]) begin
                    $display("MISMATCH row=%0d k=%0d: got=%0d expected=%0d",
                             r, k, rd_data[k], expected[r][k]);
                    errors++;
                end
            end
        end
        if (errors == 0)
            $display("PASS: all %0d values match (%0d rows x %0d K-positions)",
                     N*N, N, N);
        else
            $display("FAIL: %0d mismatches out of %0d", errors, N*N);
    $finish;
    end
endmodule