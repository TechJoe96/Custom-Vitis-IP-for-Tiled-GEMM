# =============================================================================
# build_ip.tcl
#
# Vivado batch-mode Tcl script that packages the GEMM IP from RTL sources.
#
# Usage (called from Makefile):
#   vivado -mode batch -source scripts/build_ip.tcl -tclargs <project_root>
#
# Produces:
#   vivado/gemm_ip_packaging/                          (Vivado project)
#   vivado/gemm_ip_packaging/ip_repo/                  (packaged IP)
#   vivado/gemm_ip_packaging/utilization_synth.rpt     (synthesis utilization)
#   vivado/gemm_ip_packaging/timing_synth.rpt          (synthesis timing)
# =============================================================================

# ---------- Argument parsing ----------
if {[llength $argv] < 1} {
    puts "ERROR: missing project_root argument"
    puts "Usage: vivado -mode batch -source build_ip.tcl -tclargs <project_root>"
    exit 1
}
set project_root [lindex $argv 0]

# ---------- Configuration ----------
set proj_name   "gemm_ip_packaging"
set part        "xc7z020clg400-1"
set top_module  "gemm_top"

set rtl_dir   "$project_root/rtl"
set proj_dir  "$project_root/vivado/$proj_name"
set ip_repo   "$proj_dir/ip_repo"

puts "============================================================"
puts " IP packaging: $top_module"
puts " Part        : $part"
puts " Project root: $project_root"
puts " Output      : $ip_repo"
puts "============================================================"

# ---------- Clean previous project ----------
if {[file exists $proj_dir]} {
    puts "Removing previous project at $proj_dir"
    file delete -force $proj_dir
}

# ---------- Create project ----------
create_project $proj_name $proj_dir -part $part -force
set_property target_language Verilog [current_project]
set_property simulator_language Mixed [current_project]

# ---------- Add RTL sources ----------
set rtl_files [glob -nocomplain $rtl_dir/*.sv]
if {[llength $rtl_files] == 0} {
    puts "ERROR: no SystemVerilog files found in $rtl_dir"
    exit 1
}
foreach f $rtl_files {
    puts "Adding RTL: [file tail $f]"
    add_files -norecurse $f
}
set_property file_type SystemVerilog [get_files *.sv]

# ---------- Set top module ----------
set_property top $top_module [current_fileset]
update_compile_order -fileset sources_1

# ---------- Run synthesis (sanity check) ----------
puts "==> Running out-of-context synthesis..."
launch_runs synth_1 -jobs 4
wait_on_run synth_1

if {[get_property PROGRESS [get_runs synth_1]] != "100%"} {
    puts "ERROR: synthesis failed"
    exit 1
}

# ---------- Generate post-synth reports ----------
open_run synth_1
report_utilization    -file $proj_dir/utilization_synth.rpt
report_timing_summary -file $proj_dir/timing_synth.rpt
puts "==> Synthesis reports written to $proj_dir"

# ---------- Package the IP ----------
puts "==> Packaging IP..."
file mkdir $ip_repo

ipx::package_project \
    -root_dir       $ip_repo \
    -vendor         user.org \
    -library        user \
    -taxonomy       /UserIP \
    -import_files \
    -set_current    true \
    -force

# Set the AXI-Lite clock association explicitly (avoids FREQ_HZ warning).
# This is the manual fix that was discovered during Phase 8 packaging.
set core [ipx::current_core]
ipx::associate_bus_interfaces -busif s_axi -clock s_axi_aclk $core

# Set the clock frequency parameter so downstream IPI tools know the rate.
set clk_iface [ipx::get_bus_interfaces s_axi_aclk -of_objects $core]
if {$clk_iface ne ""} {
    set freq_param [ipx::get_bus_parameters FREQ_HZ -of_objects $clk_iface]
    if {$freq_param eq ""} {
        ipx::add_bus_parameter FREQ_HZ $clk_iface
        set freq_param [ipx::get_bus_parameters FREQ_HZ -of_objects $clk_iface]
    }
    set_property value 100000000 $freq_param
}

# Regenerate metadata and save
ipx::create_xgui_files     $core
ipx::update_checksums      $core
ipx::save_core             $core

# Archive into a single .zip for distribution
ipx::archive_core $ip_repo/gemm_top.zip $core

puts "============================================================"
puts " IP packaging complete"
puts "   IP archive: $ip_repo/gemm_top.zip"
puts "   IP repo   : $ip_repo"
puts "============================================================"

close_project
exit 0
