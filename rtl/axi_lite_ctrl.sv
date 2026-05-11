`timescale 1ns/1ps

module axi_lite_ctrl #(
    parameter int N          = 8,
    parameter int ADDR_WIDTH = 9,
    parameter int DATA_WIDTH = 32
) (
    input  logic                       s_axi_aclk,
    input  logic                       s_axi_aresetn,
    input  logic [ADDR_WIDTH-1:0]      s_axi_awaddr,
    input  logic                       s_axi_awvalid,
    output logic                       s_axi_awready,
    input  logic [DATA_WIDTH-1:0]      s_axi_wdata,
    input  logic [DATA_WIDTH/8-1:0]    s_axi_wstrb,
    input  logic                       s_axi_wvalid,
    output logic                       s_axi_wready,
    output logic [1:0]                 s_axi_bresp,
    output logic                       s_axi_bvalid,
    input  logic                       s_axi_bready,
    input  logic [ADDR_WIDTH-1:0]      s_axi_araddr,
    input  logic                       s_axi_arvalid,
    output logic                       s_axi_arready,
    output logic [DATA_WIDTH-1:0]      s_axi_rdata,
    output logic [1:0]                 s_axi_rresp,
    output logic                       s_axi_rvalid,
    input  logic                       s_axi_rready,
    output logic                       start_pulse,
    input  logic                       done_flag,
    input  logic signed [31:0]         c_latched [N][N]
);
    logic rst;
    assign rst = ~s_axi_aresetn;
// Address decode — work in word units (drop low 2 bits)
    logic [ADDR_WIDTH-3:0] araddr_word;
    logic [5:0]            c_index;
    logic                  is_c_addr;
    assign araddr_word = s_axi_araddr[ADDR_WIDTH-1:2];
    assign c_index     = araddr_word - 7'd4;       // 0x10/4 = 4
    assign is_c_addr   = (araddr_word >= 7'd4) && (araddr_word <= 7'd67);
// ===== Write side (unchanged from Step 7.1) =====
    typedef enum logic [1:0] { W_IDLE, W_RESP } w_state_t;
    w_state_t w_state;
    always_ff @(posedge s_axi_aclk) begin
        if (rst) begin
            w_state       <= W_IDLE;
            s_axi_awready <= 1'b0;
            s_axi_wready  <= 1'b0;
            s_axi_bvalid  <= 1'b0;
            s_axi_bresp   <= 2'b00;
        end else begin
            case (w_state)
                W_IDLE: begin
                    s_axi_awready <= 1'b1;
                    s_axi_wready  <= 1'b1;
                    if (s_axi_awvalid && s_axi_wvalid) begin
                        s_axi_awready <= 1'b0;
                        s_axi_wready  <= 1'b0;
                        s_axi_bvalid  <= 1'b1;
                        s_axi_bresp   <= 2'b00;
                        w_state       <= W_RESP;
                    end
                end
                W_RESP: begin
                    if (s_axi_bready) begin
                        s_axi_bvalid <= 1'b0;
                        w_state      <= W_IDLE;
                    end
                end
            endcase
        end
    end
// CTRL strobe
    logic ctrl_start_w;
    always_ff @(posedge s_axi_aclk) begin
        if (rst) begin
            ctrl_start_w <= 1'b0;
        end else begin
            ctrl_start_w <= 1'b0;
            if (w_state == W_IDLE && s_axi_awvalid && s_axi_wvalid) begin
                if (s_axi_awaddr[ADDR_WIDTH-1:2] == 7'h00 && s_axi_wstrb[0]) begin
                    ctrl_start_w <= s_axi_wdata[0];
                end
            end
        end
    end
    assign start_pulse = ctrl_start_w;
// ===== Read side (now decodes C tile too) =====
    typedef enum logic [0:0] { R_IDLE, R_DATA } r_state_t;
    r_state_t r_state;
    always_ff @(posedge s_axi_aclk) begin
        if (rst) begin
            r_state       <= R_IDLE;
            s_axi_arready <= 1'b0;
            s_axi_rvalid  <= 1'b0;
            s_axi_rdata   <= '0;
            s_axi_rresp   <= 2'b00;
        end else begin
            case (r_state)
                R_IDLE: begin
                    s_axi_arready <= 1'b1;
                    if (s_axi_arvalid) begin
                        s_axi_arready <= 1'b0;
                        s_axi_rvalid  <= 1'b1;
                        s_axi_rresp   <= 2'b00;
                        case (araddr_word)
                            7'h00:   s_axi_rdata <= '0;  // CTRL reads as 0
                            7'h01:   s_axi_rdata <= {31'b0, done_flag};  // STATUS
                            default: begin
                                if (is_c_addr)
                                    s_axi_rdata <= c_latched[c_index[5:3]][c_index[2:0]];
                                else
                                    s_axi_rdata <= '0;
                            end
                        endcase
                    r_state <= R_DATA;
                    end
                end
                R_DATA: begin
                    if (s_axi_rready) begin
                        s_axi_rvalid <= 1'b0;
                        r_state      <= R_IDLE;
                    end
                end
            endcase
        end
    end
endmodule
