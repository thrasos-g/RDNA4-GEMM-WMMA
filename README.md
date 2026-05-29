# RDNA4-GEMM-WMMA

A progressive series of hand optimized GEMM (General Matrix Multiplication) kernels written in HIP C++ that target the **AMD RDNA 4** architecture (`gfx12xx`). The kernels exploit the new WMMA (Warp Matrix Multiply-Accumulate) instructions introduced in RDNA 4 via the `__builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12` compiler intrinsic, and benchmark results are validated against **rocBLAS**.

---

## Overview

Modern AMD consumer GPU architectures like RDNA 4 have hardware matrix multiply units (e.g. RX 9070 XT). Each 32 lane wavefront can execute a 16×16×16 half precision matrix multiply accumulate in a single instruction, producing float32 results. This repository explores how to expose and optimize that capability from first principles, starting from a naive global memory kernel and progressing through shared memory tiling, vectorized loads, register blocking, LDS bank conflict elimination via padding and XOR swizzling, and block level cache aware scheduling.

A second file (`mmmTensor.cpp`) applies the most optimized kernel to a **5 dimensional tensor contraction**, demonstrating how a high performance GEMM can be used as a drop in compute primitive for ML workloads.

---

## Hardware & Software Requirements

| Requirement | Details |
|---|---|
| GPU | AMD RDNA 4 (gfx12xx) — e.g. Radeon RX 9070 / 9070 XT etc. |
| ROCm | Tested with ROCm 7.1; requires `hipcc` and `hip_fp16.h` |
| rocBLAS | Included in ROCm; used for correctness validation and reference timing |
| OS | Windows (compile script is `.bat`); Linux adaptation is straightforward |
| Compiler flags | `-O3 --offload-arch=native -mcumode -lrocblas` |

> **Note:** The WMMA intrinsic `__builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12` is gfx12xx-specific and will not compile for older RDNA generations.

---

## Repository Structure

```
RDNA4-GEMM-WMMA/
├── mmm.cpp         # All GEMM kernels (naive -> fully optimized)
├── mmmTensor.cpp   # 5D tensor contraction using the optimized WMMA kernel
└── compile.bat     # Windows build + run script
```

---

## Compilation

### Windows

```bat
compile.bat
```

This script:
1. Compiles `mmm.cpp` with `-O3 --offload-arch=native -mcumode -lrocblas` -> `sgemm.exe`
2. Dumps device assembly to `kernel_rdna4.s` for inspection (targeting `gfx12xx`)
3. Runs `sgemm.exe`

### Linux (manual)

```bash
# Build
hipcc -O3 mmm.cpp --offload-arch=native -mcumode -lrocblas -o sgemm

# Optional: dump ISA
hipcc -std=c++20 -O3 --offload-arch=native -mcumode --offload-device-only -S -g mmm.cpp -o kernel_rdna4.s

# Run
./sgemm
```

For the tensor contraction benchmark:

```bash
hipcc -O3 mmmTensor.cpp --offload-arch=native -mcumode -lrocblas -o tensor && ./tensor
```

---

## WMMA Fundamentals on RDNA 4

RDNA 4 wavefronts contain 32 lanes. Each `wmma_f32_16x16x16_f16_w32` instruction computes a 16×16×16 matrix multiply accumulate over a full wavefront:

- **Input:** two 16×16 matrices in FP16 (`_Float16`)
- **Output:** one 16×16 accumulator in FP32 (`float`)
- **Data per lane:** 8 elements (`WMMA_DATA_WIDTH = 8`)
- **Vector types:** `half8` (8 × FP16), `float8` (8 × FP32), defined via GCC `ext_vector_type`

The lane mapping splits each wavefront into two **WMMA subgroups** of 16 lanes each:

```
laneWrapped = lane % 16   // position within a 16 lane subgroup
laneGroup   = lane / 16   // subgroup index (0 or 1)
```

Each lane stores 8 elements from its assigned rows/columns. On writeback, `laneGroup` selects the upper or lower half of the 16 row output tile and `ele + laneGroup * 8` gives the row offset.

---

## Kernel Progression (`mmm.cpp`)

All kernels operate on **8192 × 8192** matrices (`DIM_I = DIM_J = DIM_K = 8192`) with FP16 inputs and FP32 output. This can easily be changed by changing DIM_I, DIM_J, DIM_K at the top of the .cpp files.

### 1. `GEMM_Naive`

Baseline global memory GEMM. One thread per output element, no data reuse. Useful only as a correctness reference.

```
Grid:  (⌈N/16⌉, ⌈M/16⌉)   Block: (16, 16)
```

### 2. `GEMM_LDS`

Tiles A and B into 16×16 shared-memory blocks. Reduces global memory traffic by a factor of 16 but still uses scalar loads.

### 3. `GEMM_LDS_Vectorized`

Replaces scalar LDS loads with `uint4` casts. Each load moves 128 bits (8 × FP16) in one instruction. The tile remains 16×16.

### 4. `GEMM_LDS_Vectorized_RB`

Extends vectorized tiling to a 64×64 block tile with a 4×4 **register block** per thread, reducing `__syncthreads()` frequency and LDS bandwidth pressure.

```
Grid:  (⌈N/64⌉, ⌈M/64⌉)   Block: (16, 16)
```

### 5. `WMMA_Naive`

First WMMA kernel. One wavefront (32 threads) computes one 16×16 output tile per block. Loads A and B directly from global memory.

```
Grid:  (⌈N/16⌉, ⌈M/16⌉)   Block: (32, 1)
```

### 6. `WMMA_1D_RegBlock`

One wavefront processes a **16×32** output region by computing two 16×16 tiles. A is loaded once and reused for both B fragments (register blocking in the N dimension).

```
Grid:  (⌈N/32⌉, ⌈M/16⌉)   Block: (32, 1)
```

### 7. `WMMA_2D_RegBlock`

Four wavefronts per block arranged in a 2×2 grid, each covering a 16×16 tile for a total **32×32** block output. First step toward multi wave occupancy.

```
Grid:  (⌈N/32⌉, ⌈M/32⌉)   Block: (32, 4)
```

### 8. `WMMA_2D_RB_LDS`

Adds a 64×64 LDS staging area for a 4-wave block. Each wave computes four 16×16 tiles (a 2×2 WMMA grid). K is processed in 32 element chunks staged through shared memory.

```
Grid:  (⌈N/64⌉, ⌈M/64⌉)   Block: (32, 4)
```

### 9. `WMMA_2D_RB_VecLoad_GMEM`

Upgrades the global to LDS transfer to use `uint4` (128-bit) loads. LDS to VGPR reads for B remain scalar because B is accessed in a strided (column wise) pattern that prevents safe vectorization.

### 10. `WMMA_2D_RB_VecLoad_Full`

Adds vectorized LDS to VGPR reads for matrix A (contiguous row layout allows safe `half8` cast). B remains scalar from LDS.

### 11. `WMMA_Templated_Base<WAVE_M, WAVE_N, WAVES_I, WAVES_J>`

Fully templated kernel parameterized by:

| Parameter | Meaning |
|---|---|
| `WAVE_M` | WMMA tiles per wave in the M dimension |
| `WAVE_N` | WMMA tiles per wave in the N dimension |
| `WAVES_I` | Waves per block in the M dimension |
| `WAVES_J` | Waves per block in the N dimension |

A `static_assert` enforces a 1-to-1 mapping between threads and 128-bit load chunks, eliminating the LDS load loop entirely. A fragments are fetched inside the WMMA loop to minimize VGPR pressure.

### 12. `WMMA_LDS_Padded<WAVE_M, WAVE_N, WAVES_I, WAVES_J>`

Adds 8 element (16 byte) **padding** to each LDS row to break power of 2 strides that cause bank conflicts on RDNA 4's 32 bank LDS:

- **Before padding:** Matrix A had an 8-way LDS bank conflict per WMMA read.
- **After padding (`STRIDE_A = LDS_K + 8 = 40`):** Reduces to a 2-way conflict, yielding a significant throughput improvement.

Separate `STRIDE_A` / `STRIDE_B` variables decouple LDS addressing from global memory chunk calculations, preserving the `static_assert` 1-to-1 thread mapping.

### 13. `WMMA_LDS_Swizzled<WAVE_M, WAVE_N, WAVES_I, WAVES_J>`

Replaces padding entirely with **XOR based LDS swizzling**, possibly achieving a perfect zero way bank conflict while reclaiming all the LDS space that padding wasted.

The core idea is to scramble the column write index at load time using the row index, so threads that would normally collide on the same bank in the same cycle are redirected to different banks. On readback the inverse XOR is applied to reconstruct the original element location transparently.

**Swizzle formulas:**

| Matrix | Write swizzle | Read swizzle (inverse) |
|---|---|---|
| A | `swizzle = (row & 3) ^ ((row >> 2) & 3)` | same formula applied to `rowInLdsA` |
| B | `swizzle = (row & 15) ^ (((row >> 3) & 1) << 1)` | same formula applied to `rowInLdsB` |

Both swizzles are XOR operations so they are self inverse. Reading with the same formula undoes the scramble perfectly.

**Key changes from `WMMA_LDS_Padded`:**

- `PAD` removed; `STRIDE_A = LDS_K` and `STRIDE_B = BLOCK_N_SIZE` return to natural power-of-2 sizes.
- `As` and `Bs` are allocated at their minimal natural dimensions. No wasted shared memory.
- The `static_assert` 1-to-1 thread/chunk mapping is preserved unchanged.
- Matrix A reads remain 128bit vectorized (`half8` cast), now hitting zero bank conflicts instead of two-way.
- Matrix B reads remain scalar (strided access pattern), with the inverse swizzle applied per element.

### 14. `WMMA_Swizzled_Full<WAVE_M, WAVE_N, WAVES_I, WAVES_J>`

Combines everything from `WMMA_LDS_Swizzled` with **threadblock level output swizzling** to reduce L2 cache pressure for large matrices.

**Block swizzling** reorders block execution from the default row major schedule into vertical column strips of `SWIZZLE_SIZE = 32` blocks. Blocks in the same strip share overlapping rows of A and columns of B, so their global memory loads land in L2 cache for each other rather than going to VRAM. The remapping arithmetic:

```
block_id          = blockIdx.y * gridDim.x + blockIdx.x
group_id          = block_id / (SWIZZLE_SIZE * gridDim.y)
group_lane        = block_id % (SWIZZLE_SIZE * gridDim.y)
swizzled_block_x  = group_id * SWIZZLE_SIZE + (group_lane % SWIZZLE_SIZE)
swizzled_block_y  = group_lane / SWIZZLE_SIZE
```

An edge-case guard falls back to the original `blockIdx` coordinates when `swizzled_block_x >= gridDim.x`, making the kernel correct for any grid shape.

**Additional improvements over `WMMA_LDS_Swizzled`:**

- Row/column decomposition and swizzle indices for A and B are computed **once before the K loop**, avoiding redundant arithmetic inside the hot path.
- This is the **final and most optimized kernel** in `mmm.cpp` and serves as the direct basis for `WMMA_Swizzled_Full` in `mmmTensor.cpp`.


---

## Tensor Contraction (`mmmTensor.cpp`)

Implements a **5D tensor contraction** that maps to a GEMM via index reordering. The tensor dimensions are:

| Symbol | Size | Role |
|---|---|---|
| `mt = 128` | M-tiles | |
| `rt = 64` | R-index | |
| `nt = 64` | N-tiles | |
| `rt_l = 128` | K per N-tile | |
| `bt = 8192` | Batch/output width | |

The static weight tensor G (`rt × nt × mt × rt_l`) is packed into a 2D matrix A of shape `(mt×rt) × (nt×rt_l)` = `8192 × 8192`. The dynamic input tensor is similarly reshaped. The kernel (`WMMA_Swizzled_Full`) adds:
- **Edge-case bounds checking** so the kernel is correct when `DIM_I`, `DIM_J`, or `DIM_K` are not exact multiples of the tile size
- **Output index remapping** from standard row-major to the 5D tensor layout `(m, b, r)`

Results are validated against `rocblas_gemm_ex` and GFLOPS for both the custom kernel and rocBLAS are printed.

---

## Key Optimization Techniques Summary

| Technique | Benefit |
|---|---|
| WMMA intrinsic (`gfx12`) | Hardware matrix multiply unit; ~16× throughput vs scalar FMA |
| LDS (shared memory) tiling | Reuse global data across threads; reduce DRAM bandwidth |
| `uint4` vectorized loads (GMEM→LDS) | 128-bit loads; 8× fewer load instructions |
| `half8` vectorized reads (LDS→VGPR) | Exploits contiguous A layout; reduces instruction count |
| 2D register blocking (`WAVE_M × WAVE_N`) | Amortizes LDS and WMMA setup cost over more output elements |
| LDS padding (`PAD = 8`) | Eliminates 8-way bank conflict on A reads → 2-way |
| XOR swizzling | Zero padding overhead bank conflict elimination |
| Block swizzling (32-block groups) | Better L2 locality across wavefronts |
| Templated kernel | Compile-time tiling; enables `static_assert` loop removal |

---

## Validation

Every kernel's output is compared element wise to `rocblas_gemm_ex` (FP16 input, FP32 output, column major via transpose) with a relative tolerance of `1e-5`. The test reports CORRECT/INCORRECT and prints timing in milliseconds alongside GFLOPS for both the custom kernel and rocBLAS.

---