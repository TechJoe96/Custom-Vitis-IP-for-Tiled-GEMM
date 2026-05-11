-- Copyright 1986-2022 Xilinx, Inc. All Rights Reserved.
-- Copyright 2022-2023 Advanced Micro Devices, Inc. All Rights Reserved.
-- --------------------------------------------------------------------------------
-- Tool Version: Vivado v.2023.2 (lin64) Build 4029153 Fri Oct 13 20:13:54 MDT 2023
-- Date        : Mon May 11 15:03:56 2026
-- Host        : ecs02.poly.edu running 64-bit Red Hat Enterprise Linux release 8.10 (Ootpa)
-- Command     : write_vhdl -force -mode synth_stub
--               /home/yc7330/Custom-Vitis-IP-for-Tiled-GEMM/vivado/gemm_bd/gemm_bd.gen/sources_1/bd/design_1/ip/design_1_gemm_top_0_0/design_1_gemm_top_0_0_stub.vhdl
-- Design      : design_1_gemm_top_0_0
-- Purpose     : Stub declaration of top-level module interface
-- Device      : xc7z020clg400-1
-- --------------------------------------------------------------------------------
library IEEE;
use IEEE.STD_LOGIC_1164.ALL;

entity design_1_gemm_top_0_0 is
  Port ( 
    s_axi_aclk : in STD_LOGIC;
    s_axi_aresetn : in STD_LOGIC;
    s_axi_awaddr : in STD_LOGIC_VECTOR ( 8 downto 0 );
    s_axi_awvalid : in STD_LOGIC;
    s_axi_awready : out STD_LOGIC;
    s_axi_wdata : in STD_LOGIC_VECTOR ( 31 downto 0 );
    s_axi_wstrb : in STD_LOGIC_VECTOR ( 3 downto 0 );
    s_axi_wvalid : in STD_LOGIC;
    s_axi_wready : out STD_LOGIC;
    s_axi_bresp : out STD_LOGIC_VECTOR ( 1 downto 0 );
    s_axi_bvalid : out STD_LOGIC;
    s_axi_bready : in STD_LOGIC;
    s_axi_araddr : in STD_LOGIC_VECTOR ( 8 downto 0 );
    s_axi_arvalid : in STD_LOGIC;
    s_axi_arready : out STD_LOGIC;
    s_axi_rdata : out STD_LOGIC_VECTOR ( 31 downto 0 );
    s_axi_rresp : out STD_LOGIC_VECTOR ( 1 downto 0 );
    s_axi_rvalid : out STD_LOGIC;
    s_axi_rready : in STD_LOGIC;
    a_wr_en : in STD_LOGIC;
    a_wr_addr : in STD_LOGIC_VECTOR ( 2 downto 0 );
    a_wr_data : in STD_LOGIC_VECTOR ( 127 downto 0 );
    b_wr_en : in STD_LOGIC;
    b_wr_addr : in STD_LOGIC_VECTOR ( 2 downto 0 );
    b_wr_data : in STD_LOGIC_VECTOR ( 127 downto 0 )
  );

end design_1_gemm_top_0_0;

architecture stub of design_1_gemm_top_0_0 is
attribute syn_black_box : boolean;
attribute black_box_pad_pin : string;
attribute syn_black_box of stub : architecture is true;
attribute black_box_pad_pin of stub : architecture is "s_axi_aclk,s_axi_aresetn,s_axi_awaddr[8:0],s_axi_awvalid,s_axi_awready,s_axi_wdata[31:0],s_axi_wstrb[3:0],s_axi_wvalid,s_axi_wready,s_axi_bresp[1:0],s_axi_bvalid,s_axi_bready,s_axi_araddr[8:0],s_axi_arvalid,s_axi_arready,s_axi_rdata[31:0],s_axi_rresp[1:0],s_axi_rvalid,s_axi_rready,a_wr_en,a_wr_addr[2:0],a_wr_data[127:0],b_wr_en,b_wr_addr[2:0],b_wr_data[127:0]";
attribute X_CORE_INFO : string;
attribute X_CORE_INFO of stub : architecture is "gemm_top,Vivado 2023.2";
begin
end;
