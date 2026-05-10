`timescale 1ns/1ps

// Step 4.5: Random-tile verification of the systolic array.
// Reads N_TESTS independent A/B/C test cases (concatenated in the hex files),
// runs each through the array, captures c_out row-by-row, and asserts a
// bit-exact match against the C++ golden values.
//
// MUST match N_TESTS in software/dump_test_vectors.py.
module systolic_array_tb;

    parameter int N        = 8;
    parameter int N_TESTS  = 100;
    parameter int TILE_LEN = N * N;                 // values per tile per signal
    parameter int TOTAL    = N_TESTS * TILE_LEN;    // values in each hex file

    // Signals connecting to the DUT (Design Under Test)
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

    // Test-vector storage. One flat array per signal, holding all N_TESTS cases.
    logic signed [15:0] a_tile_all      [TOTAL];
    logic signed [15:0] b_tile_all      [TOTAL];
    logic signed [31:0] c_expected_all  [TOTAL];

    // Pass/fail counters
    int pass_count;
    int fail_count;
    int printed_failures;

    // Helper: index into the flat per-test storage
    function automatic int idx(input int test, input int row, input int col);
        idx = test*TILE_LEN + row*N + col;
    endfunction

    initial begin
        // -----------------------------------------------------------------
        // Initial conditions
        // -----------------------------------------------------------------
        clk         = 0;
        rst         = 1;
        load_weight = 0;
        for (int k = 0; k < N; k++) a_in[k] = 0;
        for (int i = 0; i < N; i++)
            for (int j = 0; j < N; j++)
                b_in[i][j] = 0;
        pass_count       = 0;
        fail_count       = 0;
        printed_failures = 0;

        // Read all test vectors at once
        $readmemh("../tb/data/a_tile.hex",     a_tile_all);
        $readmemh("../tb/data/b_tile.hex",     b_tile_all);
        $readmemh("../tb/data/c_expected.hex", c_expected_all);
        $display("Loaded %0d test cases (%0d values per signal).", N_TESTS, TOTAL);

        // -----------------------------------------------------------------
        // Reset
        // -----------------------------------------------------------------
        repeat (3) @(posedge clk);
        rst = 0;
        @(posedge clk);

        // -----------------------------------------------------------------
        // Per-test loop
        // -----------------------------------------------------------------
        for (int T = 0; T < N_TESTS; T++) begin

            // ---- Load weights for test T ----
            // Use NBA so the next-cycle behaviour is race-free.
            for (int i = 0; i < N; i++)
                for (int j = 0; j < N; j++)
                    b_in[i][j] <= b_tile_all[idx(T, i, j)];
            load_weight <= 1;
            @(posedge clk);          // weights latch into PEs at this edge
            load_weight <= 0;

            // ---- Stream A row by row (8 edges, S_0..S_7) ----
            for (int r = 0; r < N; r++) begin
                for (int k = 0; k < N; k++) a_in[k] <= a_tile_all[idx(T, r, k)];
                @(posedge clk);
            end
            for (int k = 0; k < N; k++) a_in[k] <= 0;

            // ---- Wait 6 edges for the drain pipeline to fill ----
            // c_out is all zeros during this stretch.
            repeat (6) @(posedge clk);

            // ---- Capture and check C[T] rows 0..7 ----
            // After 6 drain-fill edges, c_out at the next edge holds C[T][0],
            // then C[T][1], ..., C[T][7] on the following 7 edges.
            for (int row = 0; row < N; row++) begin
                @(posedge clk); #1;
                for (int j = 0; j < N; j++) begin
                    automatic logic signed [31:0] expected = c_expected_all[idx(T, row, j)];
                    if (c_out[j] !== expected) begin
                        fail_count++;
                        if (printed_failures < 10) begin
                            $display("FAIL  test=%0d  row=%0d  col=%0d  got=%0d  expected=%0d",
                                     T, row, j, c_out[j], expected);
                            printed_failures++;
                            if (printed_failures == 10)
                                $display("(further failures suppressed)");
                        end
                    end else begin
                        pass_count++;
                    end
                end
            end

            // After capture, the array is clean (all c_outs and shift_regs flushed
            // to zero), so the next iteration's load_weight edge starts fresh.
        end

        // -----------------------------------------------------------------
        // Summary
        // -----------------------------------------------------------------
        $display("");
        $display("==================== SUMMARY ====================");
        $display("Total checks: %0d  (%0d tests x %0d values each)",
                 pass_count + fail_count, N_TESTS, TILE_LEN);
        $display("Pass:         %0d", pass_count);
        $display("Fail:         %0d", fail_count);
        if (fail_count == 0)
            $display("RESULT:       ALL TESTS PASSED");
        else
            $display("RESULT:       FAILED");
        $display("=================================================");

        $finish;
    end

endmodule
