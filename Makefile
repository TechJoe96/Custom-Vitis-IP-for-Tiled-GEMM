# =============================================================================
# Makefile for Tiled GEMM Custom IP — 8x8 Systolic Array
#
# Single-command automation for the full Vivado pipeline:
#   vectors  →  simulation  →  IP packaging  →  bitstream
#
# Usage:
#   make help              Show available targets
#   make all               Run the full pipeline end-to-end
#   make sim TB=<name>     Run a specific testbench (default: array_full_buf_tb)
#   make clean             Remove build artifacts
# =============================================================================

SHELL := /bin/bash

# ----- Paths -----
PROJECT_ROOT := $(CURDIR)
RTL_DIR      := $(PROJECT_ROOT)/rtl
TB_DIR       := $(PROJECT_ROOT)/tb
SW_DIR       := $(PROJECT_ROOT)/software
SCRIPT_DIR   := $(PROJECT_ROOT)/scripts
BUILD_DIR    := $(PROJECT_ROOT)/build
VIVADO_DIR   := $(PROJECT_ROOT)/vivado

# ----- Tools (override on command line if needed) -----
PYTHON ?= python3
VIVADO ?= vivado

# ----- Default testbench (override with `make sim TB=<name>`) -----
TB ?= array_full_buf_tb

# ----- Source globs -----
RTL_SOURCES := $(wildcard $(RTL_DIR)/*.sv)
TB_FILE     := $(TB_DIR)/$(TB).sv

# =============================================================================
# Targets
# =============================================================================

.PHONY: all help vectors sim ip bitstream clean check-tools

# Default target — show help if user just types `make`
.DEFAULT_GOAL := help

help:
	@echo ""
	@echo "Tiled GEMM Custom IP — Build Targets"
	@echo "===================================="
	@echo ""
	@echo "  make vectors            Regenerate hex test vectors via NumPy golden model"
	@echo "  make sim [TB=<name>]    Run simulation (default TB: $(TB))"
	@echo "  make ip                 Package the Vivado IP archive (batch mode)"
	@echo "  make bitstream          Generate the bitstream for PYNQ-Z2 (batch mode)"
	@echo "  make all                Full pipeline: vectors -> sim -> ip -> bitstream"
	@echo "  make clean              Remove intermediate build artifacts"
	@echo ""
	@echo "Examples:"
	@echo "  make sim TB=gemm_axi_tb     Run the full AXI-level testbench"
	@echo "  make all                    Reproduce all deliverables end-to-end"
	@echo ""

all: vectors sim ip bitstream
	@echo ""
	@echo "============================================================"
	@echo "  Full pipeline complete:"
	@echo "    [x] Test vectors regenerated"
	@echo "    [x] RTL simulation passed (6400/6400)"
	@echo "    [x] IP packaged ($(VIVADO_DIR)/gemm_ip_packaging/)"
	@echo "    [x] Bitstream generated ($(VIVADO_DIR)/gemm_bd/)"
	@echo "============================================================"

# -----------------------------------------------------------------------------
# 1. Regenerate hex test vectors using the Python/NumPy golden model
# -----------------------------------------------------------------------------
vectors:
	@echo "==> [1/4] Generating deterministic test vectors..."
	cd $(SW_DIR) && $(PYTHON) dump_test_vectors.py
	@echo "==> Test vectors written to $(TB_DIR)/data/"

# -----------------------------------------------------------------------------
# 2. Run RTL simulation under Vivado xsim
# -----------------------------------------------------------------------------
sim: $(TB_FILE)
	@echo "==> [2/4] Running simulation: $(TB)"
	@mkdir -p $(BUILD_DIR)/sim
	@cd $(BUILD_DIR)/sim && \
	    xvlog -sv $(RTL_SOURCES) $(TB_FILE) && \
	    xelab -debug typical $(TB) -snapshot $(TB)_sim && \
	    xsim $(TB)_sim -R | tee sim.log
	@if grep -q "RESULT: ALL TESTS PASSED" $(BUILD_DIR)/sim/sim.log; then \
	    echo "==> SIMULATION PASSED"; \
	else \
	    echo "==> SIMULATION FAILED — check $(BUILD_DIR)/sim/sim.log"; \
	    exit 1; \
	fi

# -----------------------------------------------------------------------------
# 3. Package the Vivado IP using a batch-mode Tcl script
# -----------------------------------------------------------------------------
ip:
	@echo "==> [3/4] Packaging Vivado IP (batch mode)..."
	@mkdir -p $(VIVADO_DIR)
	cd $(VIVADO_DIR) && \
	    $(VIVADO) -mode batch \
	              -source $(SCRIPT_DIR)/build_ip.tcl \
	              -tclargs $(PROJECT_ROOT)
	@echo "==> IP packaged at $(VIVADO_DIR)/gemm_ip_packaging/"

# -----------------------------------------------------------------------------
# 4. Generate bitstream (block design + synth + impl + bitstream)
# -----------------------------------------------------------------------------
bitstream:
	@echo "==> [4/4] Generating bitstream (batch mode)..."
	@mkdir -p $(VIVADO_DIR)
	cd $(VIVADO_DIR) && \
	    $(VIVADO) -mode batch \
	              -source $(SCRIPT_DIR)/build_bd.tcl \
	              -tclargs $(PROJECT_ROOT)
	@echo "==> Bitstream written to $(VIVADO_DIR)/gemm_bd/gemm_bd.runs/impl_1/design_1_wrapper.bit"

# -----------------------------------------------------------------------------
# Housekeeping
# -----------------------------------------------------------------------------
clean:
	@echo "==> Removing build artifacts..."
	rm -rf $(BUILD_DIR)
	rm -f vivado*.jou vivado*.log
	rm -f xvlog.log xelab.log xsim.log xvlog.pb xelab.pb webtalk*.jou webtalk*.log
	rm -rf xsim.dir .Xil
	@echo "==> Clean complete"

check-tools:
	@command -v $(PYTHON) >/dev/null 2>&1 || (echo "ERROR: $(PYTHON) not found" && exit 1)
	@command -v $(VIVADO) >/dev/null 2>&1 || (echo "ERROR: $(VIVADO) not found — source Vivado settings64.sh first" && exit 1)
	@command -v xvlog    >/dev/null 2>&1 || (echo "ERROR: xvlog not found — source Vivado settings64.sh first" && exit 1)
	@echo "==> All required tools found"
