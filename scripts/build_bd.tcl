# =============================================================================
# build_bd.tcl
#
# Vivado batch-mode Tcl script that creates the block design (Zynq PS + GEMM IP),
# runs synthesis + implementation, and generates the bitstream.
#
# Usage (called from Makefile):
#   vivado -mode batch -source scripts/build_bd.tcl -tclargs <project_root>
#
# Prerequisites:
#   - The IP must be packaged first (run build_ip.tcl, or `make ip`)
#   - The packaged IP repo lives at vivado/gemm_ip_packaging/ip_repo/
#
# Produces:
#   vivado/gemm_bd/                                          (Vivado project)
#   vivado/gemm_bd/gemm_bd.runs/impl_1/design_1_wrapper.bit  (bitstream)
#   vivado/gemm_bd/design_1_wrapper.xsa                      (.xsa archive)
#   vivado/gemm_bd/utilization_impl.rpt                      (post-impl utilization)
#   vivado/gemm_bd/timing_impl.rpt                           (post-impl timing)
# =============================================================================

# ---------- Argument parsing ----------
if {[llength $argv] < 1} {
    puts "ERROR: missing project_root argument"
    puts "Usage: vivado -mode batch -source build_bd.tcl -tclargs <project_root>"
    exit 1
}
set project_root [lindex $argv 0]

# ---------- Configuration ----------
set proj_name   "gemm_bd"
set part        "xc7z020clg400-1"
set bd_name     "design_1"
set ip_vlnv     "user.org:user:gemm_top:1.0"

set ip_repo     "$project_root/vivado/gemm_ip_packaging/ip_repo"
set proj_dir    "$project_root/vivado/$proj_name"

puts "============================================================"
puts " Block design + bitstream"
puts " Part        : $part"
puts " IP repo     : $ip_repo"
puts " Project dir : $proj_dir"
puts "============================================================"

# ---------- Sanity check: IP repo exists ----------
if {![file exists $ip_repo]} {
    puts "ERROR: IP repo not found at $ip_repo"
    puts "Run 'make ip' (or vivado -mode batch -source build_ip.tcl) first."
    exit 1
}

# ---------- Clean previous project ----------
if {[file exists $proj_dir]} {
    puts "Removing previous project at $proj_dir"
    file delete -force $proj_dir
}

# ---------- Create project ----------
create_project $proj_name $proj_dir -part $part -force
set_property target_language Verilog  [current_project]
set_property simulator_language Mixed [current_project]

# ---------- Register packaged IP repo ----------
set_property ip_repo_paths [list $ip_repo] [current_project]
update_ip_catalog

# ---------- Create block design ----------
create_bd_design $bd_name

# Add Zynq Processing System
puts "==> Adding Zynq PS7..."
create_bd_cell -type ip -vlnv xilinx.com:ip:processing_system7 ps7

# Apply default PS7 config (uses chip defaults; works without board file)
apply_bd_automation -rule xilinx.com:bd_rule:processing_system7 \
    -config { \
        make_external "FIXED_IO, DDR" \
        Master "Disable" \
        Slave  "Disable" \
    } \
    [get_bd_cells ps7]

# Enable M_AXI_GP0 master port on the PS (default is on for PYNQ targets)
set_property -dict [list \
    CONFIG.PCW_USE_M_AXI_GP0 {1} \
    CONFIG.PCW_FCLK_CLK0_FREQ {100000000} \
] [get_bd_cells ps7]

# Add the GEMM IP
puts "==> Adding GEMM IP ($ip_vlnv)..."
create_bd_cell -type ip -vlnv $ip_vlnv gemm_top_0

# Auto-wire PS M_AXI_GP0 -> GEMM s_axi (creates AXI Interconnect + clock/reset wizard)
puts "==> Auto-connecting AXI..."
apply_bd_automation -rule xilinx.com:bd_rule:axi4 \
    -config { \
        Master "/ps7/M_AXI_GP0" \
        Clk    "Auto" \
    } \
    [get_bd_intf_pins gemm_top_0/s_axi]

# Validate the design
puts "==> Validating block design..."
validate_bd_design

# Save the BD and generate output products
save_bd_design
set bd_file [get_files $bd_name.bd]
generate_target all $bd_file

# Make HDL wrapper and set as top
puts "==> Generating HDL wrapper..."
make_wrapper -files $bd_file -top -force
set wrapper_file "$proj_dir/$proj_name.gen/sources_1/bd/$bd_name/hdl/${bd_name}_wrapper.v"
if {![file exists $wrapper_file]} {
    # Older Vivado versions place it under .srcs
    set wrapper_file "$proj_dir/$proj_name.srcs/sources_1/bd/$bd_name/hdl/${bd_name}_wrapper.v"
}
add_files -norecurse $wrapper_file
set_property top ${bd_name}_wrapper [current_fileset]
update_compile_order -fileset sources_1

# ---------- Run synthesis + implementation + bitstream ----------
puts "==> Launching implementation (synth + place + route + bitstream)..."
launch_runs impl_1 -to_step write_bitstream -jobs 4
wait_on_run impl_1

if {[get_property PROGRESS [get_runs impl_1]] != "100%"} {
    puts "ERROR: implementation failed"
    exit 1
}

# ---------- Generate post-impl reports ----------
open_run impl_1
report_utilization    -file $proj_dir/utilization_impl.rpt
report_timing_summary -file $proj_dir/timing_impl.rpt
puts "==> Post-implementation reports written to $proj_dir"

# ---------- Export hardware (.xsa containing .bit + .hwh) ----------
puts "==> Exporting hardware platform (.xsa)..."
write_hw_platform -fixed -include_bit -force $proj_dir/${bd_name}_wrapper.xsa

# ---------- Report summary ----------
set wns [get_property STATS.WNS [get_runs impl_1]]
set tns [get_property STATS.TNS [get_runs impl_1]]

puts "============================================================"
puts " Bitstream complete"
puts "   WNS         : $wns ns"
puts "   TNS         : $tns ns"
puts "   .bit file   : $proj_dir/$proj_name.runs/impl_1/${bd_name}_wrapper.bit"
puts "   .xsa archive: $proj_dir/${bd_name}_wrapper.xsa"
puts "============================================================"

close_project
exit 0
