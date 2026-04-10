# High-Level Project Outline: Custom Vitis IP for Tiled GEMM

## a. GitHub Repo

**Repository:**  
https://github.com/TechJoe96/Custom-Vitis-IP-for-Tiled-GEMM

**Planned outline file:**  
https://github.com/TechJoe96/Custom-Vitis-IP-for-Tiled-GEMM/blob/main/plan.md

This repository will contain the initial project outline and the remaining design information for the custom Vitis IP.

---

## b. Project Team

- YoungJo Choi
- [Teammate Name]

> Note: The project requires at least two students, so the second teammate name should be filled in before final submission.

---

## c. IP Definition

### Project Title

**Custom Vitis IP for Tiled GEMM Using a Systolic Array**

### Intended Functionality

This project designs a custom Vitis IP that accelerates **dense matrix multiplication** using tiled processing and a **systolic-array-based compute core**.

The main operation is:

```text
C = A × B


where:

* `A` is an `M × K` matrix
* `B` is a `K × N` matrix
* `C` is an `M × N` output matrix

A more general GEMM form can be written as:

```text
C = A × B + X
```

However, the first version of this project will implement the simpler form:

```text
C = A × B
```

### Mathematical Operations Performed

For each output element:

```text
C[i][j] = sum(A[i][k] * B[k][j] for k in range(K))
```

### Python-Style Pseudocode

```python
for i in range(M):
    for j in range(N):
        acc = 0
        for k in range(K):
            acc += A[i][k] * B[k][j]
        C[i][j] = acc
```

### Tiled Execution Pseudocode

```python
for i0 in range(0, M, TM):
    for j0 in range(0, N, TN):
        C_tile = zeros(TM, TN)
        for k0 in range(0, K, TK):
            A_tile = A[i0:i0+TM, k0:k0+TK]
            B_tile = B[k0:k0+TK, j0:j0+TN]
            C_tile += A_tile @ B_tile
        write_back(C_tile)
```

### Why This Is Well-Suited for Hardware Acceleration

This operation is well-suited for hardware acceleration because:

* it consists of many repeated **multiply-accumulate (MAC)** operations
* many products can be computed in parallel
* matrix tiles can be reused efficiently from on-chip buffers
* the control flow is regular and predictable
* the computation maps naturally to a pipelined **systolic array**

These properties make GEMM a strong target for FPGA acceleration.

---

## d. IP Architecture

### Overall Design Choice

The design uses **shared DDR memory at the system level** and **AXI4-Stream at the custom IP boundary**.

That means:

* input and output matrices are stored in **DDR memory**
* the processor configures the accelerator through **AXI4-Lite**
* **AXI DMA** moves matrix tiles between DDR and the custom IP
* the custom IP receives and transmits tile data through **AXI4-Stream**
* inside the IP, local buffers and FIFOs stage tile data for the systolic array

This is a modular and beginner-friendly architecture because memory movement is handled by DMA, while the custom IP focuses on buffering, control, and arithmetic.

### High-Level Dataflow

1. The processor stores matrices `A` and `B` in DDR memory.
2. The processor writes matrix dimensions and control values to the IP through AXI4-Lite.
3. AXI DMA reads matrix tiles from DDR memory.
4. The DMA sends tile data to the custom IP over AXI4-Stream.
5. The custom IP stores tiles in local buffers/FIFOs.
6. The systolic array performs tiled GEMM computation.
7. Output tiles are streamed back out of the IP.
8. AXI DMA writes the output matrix `C` back to DDR.
9. The processor reads status or completion information.

---

## Hardware Modules

### 1. AXI4-Lite Control / Status Module

This module is part of the custom IP and provides control and status registers.

Typical registers include:

* matrix dimensions: `M`, `N`, `K`
* tile sizes: `TM`, `TN`, `TK`
* start bit
* done bit
* status / error flags
* optional performance counters

It interfaces with the processor through **AXI4-Lite**.

---

### 2. AXI4-Stream Input / Output Wrapper

This module is the stream boundary of the custom IP.

Its job is to:

* receive incoming tile data from AXI DMA
* send completed result tiles back out
* handle streaming handshake signals such as `TVALID`, `TREADY`, and optional `TLAST`

This keeps the compute core independent from direct DDR protocol handling.

---

### 3. Input Tile Buffers

This module stores small blocks of matrix `A` and matrix `B` in local on-chip memory such as BRAM.

Its job is to:

* hold tiles near the compute core
* reduce repeated reads from external memory
* provide regular access patterns to the compute engine

---

### 4. Optional Transpose / Reorder Module

This module reorders one input tile if needed before entering the compute array.

Its job is to:

* rearrange matrix data into the order expected by the systolic array
* support regular flow into the processing elements
* improve reuse and feed efficiency

---

### 5. Systolic Array Compute Core

This is the main arithmetic engine of the custom IP.

Its job is to:

* receive tile data from local buffers
* perform repeated multiply-accumulate operations
* propagate operands through the array in a regular pattern
* generate partial sums and final tile results

This module adopts a **systolic array** architecture rather than a generic unstructured MAC array.

---

### 6. Accumulation / Output Buffer

This module stores partial sums and completed output tile values.

Its job is to:

* preserve accumulated values across tile iterations
* stage finished output tiles before stream-out
* separate compute timing from output timing

---

### 7. Tile Controller / Scheduler

This module is the main FSM/controller of the custom IP.

Its job is to:

* sequence load, compute, and store phases
* track tile coordinates
* coordinate buffer reuse
* manage start / done behavior
* control the systolic array execution

This module is kept separate from the arithmetic core to improve modularity and verification.

---

## Interface Decision: AXI4-Stream, Shared Memory, and FIFOs

The design intentionally uses all three, but at different levels:

* **DDR / shared memory** is used for storing large matrices because the processor naturally manages large data in memory.
* **AXI DMA** is used to move data between memory and the accelerator.
* **AXI4-Stream** is used at the IP boundary because tile data is consumed in a sequential stream.
* **Local FIFOs / buffers** are used inside the IP to connect modules and stage short bursts of data.

So the intended data path is:

```text
DDR/shared memory -> DMA -> AXI4-Stream -> local buffers/FIFOs -> systolic array
```

This is not a contradiction:

* DDR is used for storage
* AXI4-Stream is used for feeding the IP
* FIFOs are used for short internal producer-consumer communication

---

## Why the Design Is Partitioned into Modules

The design avoids one large monolithic block because:

* smaller modules are easier to understand
* each block can be tested separately
* arithmetic and control logic can be debugged independently
* modular partitioning helps parallelism and pipelining
* the design can be upgraded later without redesigning everything

Possible future upgrades include:

* larger systolic array
* double buffering
* support for `C = A × B + X`
* direct DDR access from the compute core
* additional performance counters
* support for different precisions

---

## Initial Implementation Scope

The first version of the project will support:

* one GEMM operation at a time
* tiled dense matrix multiplication
* AXI4-Lite control
* AXI4-Stream input/output
* AXI DMA-based movement between DDR and IP
* local tile buffers
* systolic-array-based compute core

The first version will **not** yet include:

* multiple queued jobs
* sparse matrices
* full GEMM bias/add variants
* advanced scheduling
* direct DDR access from the compute core

---

## Short Summary

This project designs a **custom Vitis IP for tiled GEMM acceleration using a systolic array**.

The system:

* stores matrices in **DDR shared memory**
* uses **AXI DMA** to move tiles
* connects to the custom IP through **AXI4-Stream**
* uses local buffers/FIFOs inside the IP to feed the compute array

The custom IP itself includes:

* AXI4-Lite control registers
* stream interfaces
* tile buffers
* a controller
* a systolic-array-based compute core

GEMM is a strong choice because it is a regular, highly parallel multiply-accumulate workload that maps well to FPGA acceleration.

```
```
