`timescale 1ns/1ps

module gemm_axi_tb;

    parameter int N          = 8;
    parameter int WIDTH      = 16;
    parameter int N_TESTS    = 100;
    parameter int TILE_LEN   = N * N;
    parameter int TOTAL      = N_TESTS * TILE_LEN;
    parameter int ADDR_WIDTH = 9;

    localparam logic [ADDR_WIDTH-1:0] ADDR_CTRL    = 9'h000;
    localparam logic [ADDR_WIDTH-1:0] ADDR_STATUS  = 9'h004;
    localparam logic [ADDR_WIDTH-1:0] ADDR_C_BASE  = 9'h010;

    function automatic logic [ADDR_WIDTH-1:0] c_addr(input int row, input int col);
        c_addr = ADDR_C_BASE + (row * N + col) * 4;
    endfunction

    logic                    clk;
    logic                    resetn;
    logic [ADDR_WIDTH-1:0]   awaddr;
    logic                    awvalid, awready;
    logic [31:0]             wdata;
    logic [3:0]              wstrb;
    logic                    wvalid, wready;
    logic [1:0]              bresp;
    logic                    bvalid, bready;
    logic [ADDR_WIDTH-1:0]   araddr;
    logic                    arvalid, arready;
    logic [31:0]             rdata;
    logic [1:0]              rresp;
    logic                    rvalid, rready;

    logic                         a_wr_en, b_wr_en;
    logic [$clog2(N)-1:0]         a_wr_addr, b_wr_addr;
    logic signed [WIDTH-1:0]      a_wr_data [N], b_wr_data [N];

    gemm_top #(.N(N), .WIDTH(WIDTH), .ADDR_WIDTH(ADDR_WIDTH)) dut (
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
        .b_wr_en       (b_wr_en), .b_wr_addr     (b_wr_addr), .b_wr_data (b_wr_data)
    );

    always #5 clk = ~clk;

    // Drop awvalid/wvalid IMMEDIATELY when we observe bvalid, otherwise the
    // slave returns to W_IDLE while these are still asserted and accepts a
    // spurious second write (-> second start_pulse). Same issue for arvalid.
    task automatic axi_write(input logic [ADDR_WIDTH-1:0] addr, input logic [31:0] data);
        @(posedge clk);
        awaddr  <= addr;  awvalid <= 1;
        wdata   <= data;  wstrb   <= 4'hF;  wvalid <= 1;
        bready  <= 1;
        while (!bvalid) @(posedge clk);
        awvalid <= 0;  wvalid <= 0;
        @(posedge clk);
        bready  <= 0;
    endtask

    task automatic axi_read(input logic [ADDR_WIDTH-1:0] addr, output logic [31:0] data);
        @(posedge clk);
        araddr  <= addr;  arvalid <= 1;  rready <= 1;
        while (!rvalid) @(posedge clk);
        data = rdata;
        arvalid <= 0;
        @(posedge clk);
        rready <= 0;
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
        clk = 0;
        resetn = 0;
        awaddr = 0; awvalid = 0; wdata = 0; wstrb = 0; wvalid = 0; bready = 0;
        araddr = 0; arvalid = 0; rready = 0;
        a_wr_en = 0; a_wr_addr = 0; b_wr_en = 0; b_wr_addr = 0;
        for (int k = 0; k < N; k++) begin
            a_wr_data[k] = 0;  b_wr_data[k] = 0;
        end

        $readmemh("../tb/data/a_tile.hex",     a_tile_all);
        $readmemh("../tb/data/b_tile.hex",     b_tile_all);
        $readmemh("../tb/data/c_expected.hex", c_expected_all);
        $display("Loaded %0d test cases.", N_TESTS);

        repeat (3) @(posedge clk);
        resetn <= 1;
        repeat (2) @(posedge clk);

        pass_count = 0; fail_count = 0; printed_failures = 0;

        for (int T = 0; T < N_TESTS; T++) begin
            logic [31:0] status_val;
            logic [31:0] c_val;

            // Reset between tests — flushes array, FSM, c_latched
            resetn <= 0;
            repeat (2) @(posedge clk);
            resetn <= 1;
            repeat (2) @(posedge clk);

            // Pre-load both buffers in parallel
            for (int r = 0; r < N; r++) begin
                a_wr_en <= 1; a_wr_addr <= r;
                b_wr_en <= 1; b_wr_addr <= r;
                for (int k = 0; k < N; k++) begin
                    a_wr_data[k] <= a_tile_all[idx(T, r, k)];
                    b_wr_data[k] <= b_tile_all[idx(T, r, k)];
                end
                @(posedge clk);
            end
            a_wr_en <= 0; b_wr_en <= 0;

            // Kick off compute
            axi_write(ADDR_CTRL, 32'h1);

            // Poll STATUS until done
            status_val = 0;
            while (status_val[0] == 1'b0) begin
                axi_read(ADDR_STATUS, status_val);
            end

            // Read C tile through AXI (64 reads per test)
            for (int row = 0; row < N; row++) begin
                for (int col = 0; col < N; col++) begin
                    automatic logic signed [31:0] expected = c_expected_all[idx(T, row, col)];
                    axi_read(c_addr(row, col), c_val);
                    if ($signed(c_val) !== expected) begin
                        fail_count++;
                        if (printed_failures < 10) begin
                            $display("FAIL T=%0d row=%0d col=%0d got=%0d expected=%0d",
                                     T, row, col, $signed(c_val), expected);
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