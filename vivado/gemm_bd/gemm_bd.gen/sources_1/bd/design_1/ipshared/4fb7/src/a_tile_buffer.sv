`timescale 1ns/1ps
module a_tile_buffer #(
    parameter int WIDTH      = 16,
    parameter int N          = 8,
    parameter int ADDR_WIDTH = $clog2(N)
) (
    input  logic                       clk,
// Write port
    input  logic                       wr_en,
    input  logic [ADDR_WIDTH-1:0]      wr_addr,
    input  logic signed [WIDTH-1:0]    wr_data [N],
// Read port
    input  logic [ADDR_WIDTH-1:0]      rd_addr,
    output logic signed [WIDTH-1:0]    rd_data [N]
);
    genvar k;
    generate
        for (k = 0; k < N; k = k + 1) begin: bank
            tile_buffer #(
                .WIDTH (WIDTH),
                .DEPTH (N)
            ) buf_inst (
                .clk     (clk),
                .wr_en   (wr_en),
                .wr_addr (wr_addr),
                .wr_data (wr_data[k]),
                .rd_addr (rd_addr),
                .rd_data (rd_data[k])
            );
        end
    endgenerate
endmodule