`timescale 1ns/1ps
module gemm_axi_tb;
    parameter int N        = 8;
    parameter int WIDTH    = 16;
    parameter int N_TESTS  = 100;
    parameter int TILE_LEN = N * N;
    parameter int TOTAL    = N_TESTS * TILE_LEN;
    localparam logic [3:0] ADDR_CTRL   = 4'h0;
    localparam logic [3:0] ADDR_STATUS = 4'h4;
// AXI signals (testbench is master, DUT is slave)
    logic        clk;
    logic        resetn;
    logic [3:0]  awaddr;
    logic        awvalid, awready;
    logic [31:0] wdata;
    logic [3:0]  wstrb;
    logic        wvalid, wready;
    logic [1:0]  bresp;
    logic        bvalid, bready;
    logic [3:0]  araddr;
    logic        arvalid, arready;
    logic [31:0] rdata;
    logic [1:0]  rresp;
    logic        rvalid, rready;
// Buffer write ports
    logic                         a_wr_en, b_wr_en;
    logic [$clog2(N)-1:0]         a_wr_addr, b_wr_addr;
    logic signed [WIDTH-1:0]      a_wr_data [N], b_wr_data [N];
// Output
    logic signed [31:0]           c_out [N];
// DUT
    gemm_top #(.N(N), .WIDTH(WIDTH)) dut (
        .s_axi_aclk    (clk),
        .s_axi_aresetn (resetn),
        .s_axi_awaddr  (awaddr),  .s_axi_awvalid (awvalid), .s_axi_awready (awready),
        .s_axi_wdata   (wdata),   .s_axi_wstrb   (wstrb),
        .s_axi_wvalid  (wvalid),  .s_axi_wready  (wready),
        .s_axi_bresp   (bresp),   .s_axi_bvalid  (bvalid),  .s_axi_bready (bready),
        .s_axi_araddr  (araddr),  .s_axi_arvalid (arvalid), .s_axi_arready (arready),
        .s_axi_rdata   (rdata),   .s_axi_rresp   (rresp),
        .s_axi_rvalid  (rvalid),  .s_axi_rready  (rready),
        .a_wr_en       (a_wr_en), .a_wr_addr     (a_wr_addr), .a_wr_data (a_wr_data),
        .b_wr_en       (b_wr_en), .b_wr_addr     (b_wr_addr), .b_wr_data (b_wr_data),
        .c_out         (c_out)
    );
    always #5 clk = ~clk;
// -------------------------- AXI master tasks --------------------------
    task automatic axi_write(input logic [3:0] addr, input logic [31:0] data);
        @(posedge clk);
        awaddr  <= addr;
        awvalid <= 1;
        wdata   <= data;
        wstrb   <= 4'hF;
        wvalid  <= 1;
        bready  <= 1;
        // wait for slave to send response (bvalid)
        while (!bvalid) @(posedge clk);
        @(posedge clk);   // bvalid && bready handshake
        awvalid <= 0;
        wvalid  <= 0;
        bready  <= 0;
    endtask
    task automatic axi_read(input logic [3:0] addr, output logic [31:0] data);
        @(posedge clk);
        araddr  <= addr;
        arvalid <= 1;
        rready  <= 1;
        while (!rvalid) @(posedge clk);
        data = rdata;
        @(posedge clk);
        arvalid <= 0;
        rready  <= 0;
    endtask

    logic signed [15:0] a_tile_all     [TOTAL];
    logic signed [15:0] b_tile_all     [TOTAL];
    logic signed [31:0] c_expected_all [TOTAL];
    int pass_count;
    int fail_count;
    int printed_failures;
    function automatic int idx(input int test, input int row, input int col);
        idx = test*TILE_LEN + row*N + col;
    endfunction

    initial begin
        // Initialize all signals
        clk = 0;
        resetn = 0;
        awaddr = 0; awvalid = 0; wdata = 0; wstrb = 0; wvalid = 0; bready = 0;
        araddr = 0; arvalid = 0; rready = 0;
        a_wr_en = 0; a_wr_addr = 0; b_wr_en = 0; b_wr_addr = 0;
        for (int k = 0; k < N; k++) begin
            a_wr_data[k] = 0;
            b_wr_data[k] = 0;
        end

        $readmemh("../tb/data/a_tile.hex",     a_tile_all);
        $readmemh("../tb/data/b_tile.hex",     b_tile_all);
        $readmemh("../tb/data/c_expected.hex", c_expected_all);
        $display("Loaded %0d test cases.", N_TESTS);
// Release reset (AXI active-low convention)
        repeat (3) @(posedge clk);
        resetn <= 1;
        repeat (2) @(posedge clk);
    
        pass_count = 0; fail_count = 0; printed_failures = 0;

        for (int T = 0; T < N_TESTS; T++) begin
            logic [31:0] status_val;
// Pre-load both buffers (still native ports)
            for (int r = 0; r < N; r++) begin
                a_wr_en   <= 1;
                a_wr_addr <= r;
                b_wr_en   <= 1;
                b_wr_addr <= r;
                for (int k = 0; k < N; k++) begin
                    a_wr_data[k] <= a_tile_all[idx(T, r, k)];
                    b_wr_data[k] <= b_tile_all[idx(T, r, k)];
                end
                @(posedge clk);
            end
            a_wr_en <= 0;
            b_wr_en <= 0;
// Kick off compute via AXI write to CTRL
            axi_write(ADDR_CTRL, 32'h1);
// Poll STATUS until done bit (bit 0) is high
            status_val = 0;
            while (status_val[0] == 1'b0) begin
                axi_read(ADDR_STATUS, status_val);
            end
// Capture & check 8 rows of C
            // Note: done was latched when the FSM finished, so several cycles
            // may have passed since row 0 was on c_out. The latch lets us
            // detect "done" reliably, but we have to be careful about timing.
            #1;
            for (int j = 0; j < N; j++) begin
                automatic logic signed [31:0] expected = c_expected_all[idx(T, 0, j)];
                if (c_out[j] !== expected) begin
                    fail_count++;
                    if (printed_failures < 10) begin
                        $display("FAIL T=%0d row=0 col=%0d got=%0d expected=%0d",
                                 T, j, c_out[j], expected);
                        printed_failures++;
                    end
                end else pass_count++;
            end
// Read remaining 7 rows
            for (int row = 1; row < N; row++) begin
                @(posedge clk); #1;
                for (int j = 0; j < N; j++) begin
                    automatic logic signed [31:0] expected = c_expected_all[idx(T, row, j)];
                    if (c_out[j] !== expected) begin
                        fail_count++;
                        if (printed_failures < 10) begin
                            $display("FAIL T=%0d row=%0d col=%0d got=%0d expected=%0d",
                                     T, row, j, c_out[j], expected);
                            printed_failures++;
                        end
                    end else pass_count++;
                end
            end
        end
        $display("==================== SUMMARY ====================");
        $display("Total checks: %0d  (%0d tests x %0d values each)",
                 pass_count + fail_count, N_TESTS, TILE_LEN);
        $display("Pass:         %0d", pass_count);
        $display("Fail:         %0d", fail_count);
        if (fail_count == 0) $display("RESULT:       ALL TESTS PASSED");
        else                  $display("RESULT:       FAILED");
        $display("=================================================");
    $finish;
    end
endmodule
