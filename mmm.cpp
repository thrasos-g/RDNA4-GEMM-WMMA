#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <cmath>
#include <ctime>
#include <limits>
#include <algorithm>
#include <hip/hip_ext.h>
#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include <rocblas/rocblas.h>

const int DIM_I = 8192;
const int DIM_J = 8192;
const int DIM_K = 8192;

const int Rbi = 4;
const int Rbj = 4;

#define LDS_tile 32

// Define the size of the fragment held by each lane in a wavefront.
// RDNA 4 requires 8 elements per lane for a 16x16 matrix
#define WMMA_DATA_WIDTH 8

// Define vector types expected by the RDNA 4 WMMA intrinsics
typedef _Float16 half8 __attribute__((ext_vector_type(8)));
typedef float float8 __attribute__((ext_vector_type(8)));

#define BLOCK 8 //8*8=64 => 1 wave 
#define CEIL_DIV(a,b) (((a)+(b)-1)/(b))

#define HIP_CHECK(call) do { \
  hipError_t _e = (call); \
  if (_e != hipSuccess) { \
    fprintf(stderr, "HIP error %s:%d: %s\n", __FILE__, __LINE__, hipGetErrorString(_e)); \
    exit(1); \
  } \
} while(0)



#define ROCBLAS_CHECK(call) do { \
  rocblas_status _s = (call); \
  if (_s != rocblas_status_success) { \
    fprintf(stderr, "rocBLAS error %s:%d: %d\n", __FILE__, __LINE__, _s); \
    exit(1); \
  } \
} while(0)


__global__ void GEMM_Naive(const _Float16 *A, const _Float16 *B, float *C) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < DIM_I && col < DIM_J) {
        float acc = 0.0f;
        for (int k = 0; k < DIM_K; ++k)
            acc += static_cast<float>(A[row * DIM_K + k]) * static_cast<float>(B[k * DIM_J + col]);
            
        C[row * DIM_J + col] = acc;
    }
}

__global__ void GEMM_LDS(const _Float16 *A, const _Float16 *B, float *C) {
    constexpr int TILE = 16;
    __shared__ _Float16 As[TILE][TILE];
    __shared__ _Float16 Bs[TILE][TILE];
    int row = blockIdx.y * TILE + threadIdx.y;
    int col = blockIdx.x * TILE + threadIdx.x;
    float acc = 0.0f;

    for (int tileK = 0; tileK < DIM_K; tileK += TILE) {
        int tiledColA = tileK + threadIdx.x;
        int tiledRowB = tileK + threadIdx.y;
        As[threadIdx.y][threadIdx.x] = (row < DIM_I && tiledColA < DIM_K) ? A[row * DIM_K + tiledColA] : static_cast<_Float16>(0.0f);
        Bs[threadIdx.y][threadIdx.x] = (tiledRowB < DIM_K && col < DIM_J) ? B[tiledRowB * DIM_J + col] : static_cast<_Float16>(0.0f);
        __syncthreads();

        #pragma unroll
        for (int k = 0; k < TILE; ++k)
            acc += static_cast<float>(As[threadIdx.y][k]) * static_cast<float>(Bs[k][threadIdx.x]);
        __syncthreads();
    }

    if (row < DIM_I && col < DIM_J)
        C[row * DIM_J + col] = acc;
}

__global__ void GEMM_LDS_Vectorized(const _Float16 *A, const _Float16 *B, float *C) {
    constexpr int TILE = 16;
    constexpr int ELEMENTS_PER_LOAD = 8;
    constexpr int CHUNKS_PER_ROW = TILE / ELEMENTS_PER_LOAD;
    constexpr int LOADS_PER_TILE = TILE * CHUNKS_PER_ROW;

    alignas(16) __shared__ _Float16 As[TILE * TILE];
    alignas(16) __shared__ _Float16 Bs[TILE * TILE];

    int row = blockIdx.y * TILE + threadIdx.y;
    int col = blockIdx.x * TILE + threadIdx.x;
    int tid = threadIdx.y * TILE + threadIdx.x;

    float acc = 0.0f;

    for (int tileK = 0; tileK < DIM_K; tileK += TILE) {
        for (int idx = tid; idx < LOADS_PER_TILE; idx += TILE * TILE) {
            int localRow = idx / CHUNKS_PER_ROW;
            int colChunk = idx % CHUNKS_PER_ROW;

            int globalRowA = blockIdx.y * TILE + localRow;
            int globalRowB = tileK + localRow;

            reinterpret_cast<uint4 *>(As)[localRow * CHUNKS_PER_ROW + colChunk] = reinterpret_cast<const uint4 *>(A + globalRowA * DIM_K + tileK)[colChunk];
            reinterpret_cast<uint4 *>(Bs)[localRow * CHUNKS_PER_ROW + colChunk] = reinterpret_cast<const uint4 *>(B + globalRowB * DIM_J + blockIdx.x * TILE)[colChunk];
        }

        __syncthreads();

        #pragma unroll
        for (int k = 0; k < TILE; ++k)
            acc += static_cast<float>(As[threadIdx.y * TILE + k]) * static_cast<float>(Bs[k * TILE + threadIdx.x]);

        __syncthreads();
    }

    C[row * DIM_J + col] = acc;
}

__global__ void GEMM_LDS_Vectorized_RB(const _Float16 *A, const _Float16 *B, float *C) {
    constexpr int BLOCK_M = 64;
    constexpr int BLOCK_N = 64;
    constexpr int TILE_K = 16;
    constexpr int RB_M = 4;
    constexpr int RB_N = 4;
    constexpr int THREADS_X = 16;
    constexpr int THREADS_Y = 16;
    constexpr int ELEMENTS_PER_LOAD = 8;
    constexpr int A_CHUNKS_PER_ROW = TILE_K / ELEMENTS_PER_LOAD;
    constexpr int B_CHUNKS_PER_ROW = BLOCK_N / ELEMENTS_PER_LOAD;
    constexpr int A_LOADS_PER_TILE = BLOCK_M * A_CHUNKS_PER_ROW;
    constexpr int B_LOADS_PER_TILE = TILE_K * B_CHUNKS_PER_ROW;

    alignas(16) __shared__ _Float16 As[BLOCK_M * TILE_K];
    alignas(16) __shared__ _Float16 Bs[TILE_K * BLOCK_N];

    int tid = threadIdx.y * THREADS_X + threadIdx.x;
    int blockRow = blockIdx.y * BLOCK_M;
    int blockCol = blockIdx.x * BLOCK_N;
    int localRow = threadIdx.y * RB_M;
    int localCol = threadIdx.x * RB_N;

    float acc[RB_M][RB_N] = {0.0f};

    for (int tileK = 0; tileK < DIM_K; tileK += TILE_K) {
        for (int idx = tid; idx < A_LOADS_PER_TILE; idx += THREADS_X * THREADS_Y) {
            int row = idx / A_CHUNKS_PER_ROW;
            int colChunk = idx % A_CHUNKS_PER_ROW;

            reinterpret_cast<uint4 *>(As)[row * A_CHUNKS_PER_ROW + colChunk] =
                reinterpret_cast<const uint4 *>(A + (blockRow + row) * DIM_K + tileK)[colChunk];
        }

        for (int idx = tid; idx < B_LOADS_PER_TILE; idx += THREADS_X * THREADS_Y) {
            int row = idx / B_CHUNKS_PER_ROW;
            int colChunk = idx % B_CHUNKS_PER_ROW;

            reinterpret_cast<uint4 *>(Bs)[row * B_CHUNKS_PER_ROW + colChunk] =
                reinterpret_cast<const uint4 *>(B + (tileK + row) * DIM_J + blockCol)[colChunk];
        }

        __syncthreads();

        #pragma unroll
        for (int k = 0; k < TILE_K; ++k) {
            float aReg[RB_M];
            float bReg[RB_N];

            #pragma unroll
            for (int ri = 0; ri < RB_M; ++ri)
                aReg[ri] = static_cast<float>(As[(localRow + ri) * TILE_K + k]);

            #pragma unroll
            for (int rj = 0; rj < RB_N; ++rj)
                bReg[rj] = static_cast<float>(Bs[k * BLOCK_N + localCol + rj]);

            #pragma unroll
            for (int ri = 0; ri < RB_M; ++ri) {
                #pragma unroll
                for (int rj = 0; rj < RB_N; ++rj)
                    acc[ri][rj] += aReg[ri] * bReg[rj];
            }
        }

        __syncthreads();
    }

    #pragma unroll
    for (int ri = 0; ri < RB_M; ++ri) {
        #pragma unroll
        for (int rj = 0; rj < RB_N; ++rj)
            C[(blockRow + localRow + ri) * DIM_J + blockCol + localCol + rj] = acc[ri][rj];
    }
}


// To Run:
// Update grid/block dimensions
// dim3 threadsPerBlock(32, 1, 1); 
// dim3 blocksPerGrid(CEIL_DIV(DIM_J, 16), CEIL_DIV(DIM_I, 16), 1);
// WMMA_Naive<<<blocksPerGrid, threadsPerBlock>>>(dA, dB, dC);

__global__ void WMMA_Naive(_Float16 *A, _Float16 *B, float *C) {
    int lane = threadIdx.x; // Thread ID within the wave (0-31)
    int laneWrapped = lane % 16; // Lane ID within a WMMA subgroup (0-15)
    int laneGroup = lane / 16; // Which WMMA subgroup this lane belongs to (0 or 1)

    int row_c = blockIdx.y * 16; // Start row index of the block's 16x16 output tile
    int col_c = blockIdx.x * 16; // Start column index of the block's 16x16 output tile

    float8 c_frag = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f}; // Accumulator register buffer for this thread's piece of C

    for (int k = 0; k < DIM_K; k += 16) { // Iterate across K in 16 element chunks for WMMA operations
        half8 a_frag; // Register buffer for this thread's piece of matrix A
        half8 b_frag; // Register buffer for this thread's piece of matrix B

        // Load B fragment (row major) directly from GMEM
        for(int ele = 0; ele < WMMA_DATA_WIDTH; ++ele) {
            int r_B = k + (ele + laneGroup * WMMA_DATA_WIDTH); // Calculate row offset in B
            int c_B = col_c + laneWrapped; // Calculate column offset in B
            b_frag[ele] = B[r_B * DIM_J + c_B]; 
        }

        // Load A fragment (column major) directly from GMEM
        for(int ele = 0; ele < WMMA_DATA_WIDTH; ++ele) {
            int r_A = row_c + laneWrapped; // Calculate row offset in A
            int c_A = k + (ele + laneGroup * WMMA_DATA_WIDTH); // Calculate column offset in A
            a_frag[ele] = A[r_A * DIM_K + c_A];
        }

        // Execute gfx1201 WMMA instruction: C_frag += A_frag * B_frag
        c_frag = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12(a_frag, b_frag, c_frag);
    }

    // Store C fragment (row-major) back to Global Memory
    for(int ele = 0; ele < WMMA_DATA_WIDTH; ++ele) {
        int r_C = row_c + (ele + laneGroup * WMMA_DATA_WIDTH); // Target row in C
        int c_C = col_c + laneWrapped; // Target column in C
        C[r_C * DIM_J + c_C] = c_frag[ele];
    }
}

// To Run:
// Update grid/block dimensions
// dim3 threadsPerBlock(32, 1, 1); 
// dim3 blocksPerGrid(CEIL_DIV(DIM_J, 32), CEIL_DIV(DIM_I, 16), 1);
// WMMA_1D_RegBlock<<<blocksPerGrid, threadsPerBlock>>>(dA, dB, dC);

__global__ void WMMA_1D_RegBlock(_Float16 *A, _Float16 *B, float *C) {
    int lane = threadIdx.x; // Thread ID within the wave (0-31)
    int laneWrapped = lane % 16; // Lane ID within a WMMA subgroup (0-15)
    int laneGroup = lane / 16; // Which WMMA subgroup this lane belongs to (0 or 1)

    int row_c = blockIdx.y * 16;  // Start row index (processes 16 rows)
    int col_c = blockIdx.x * 32;  // Start column index (processes 32 columns 1D register blocking)

    float8 c_frag0 = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f}; // Accumulator for first 16x16 tile
    float8 c_frag1 = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f}; // Accumulator for second 16x16 tile

    for (int k = 0; k < DIM_K; k += 16) {
        half8 a_frag; // Shared A fragment used for both matmuls
        half8 b_frag0; // B fragment for left 16 cols
        half8 b_frag1; // B fragment for right 16 cols

        // Load A (used for both matmuls)
        for(int ele = 0; ele < WMMA_DATA_WIDTH; ++ele) {
            int r_A = row_c + laneWrapped; 
            int c_A = k + (ele + laneGroup * WMMA_DATA_WIDTH);
            a_frag[ele] = A[r_A * DIM_K + c_A];
        }

        // Load B0 (for col_c)
        for(int ele = 0; ele < WMMA_DATA_WIDTH; ++ele) {
            int r_B = k + (ele + laneGroup * WMMA_DATA_WIDTH);
            int c_B0 = col_c + laneWrapped;
            b_frag0[ele] = B[r_B * DIM_J + c_B0];
        }

        // Load B1 (for col_c + 16)
        for(int ele = 0; ele < WMMA_DATA_WIDTH; ++ele) {
            int r_B = k + (ele + laneGroup * WMMA_DATA_WIDTH);
            int c_B1 = (col_c + 16) + laneWrapped;
            b_frag1[ele] = B[r_B * DIM_J + c_B1];
        }

        // Execute two RDNA4 WMMA instructions per loop iteration
        c_frag0 = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12(a_frag, b_frag0, c_frag0);
        c_frag1 = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12(a_frag, b_frag1, c_frag1);
    }

    // Store C0 fragment
    for(int ele = 0; ele < WMMA_DATA_WIDTH; ++ele) {
        int r_C = row_c + (ele + laneGroup * WMMA_DATA_WIDTH);
        int c_C0 = col_c + laneWrapped;
        C[r_C * DIM_J + c_C0] = c_frag0[ele];
    }

    // Store C1 fragment
    for(int ele = 0; ele < WMMA_DATA_WIDTH; ++ele) {
        int r_C = row_c + (ele + laneGroup * WMMA_DATA_WIDTH);
        int c_C1 = (col_c + 16) + laneWrapped;
        C[r_C * DIM_J + c_C1] = c_frag1[ele];
    }
}

// To Run: 
// dim3 threadsPerBlock(32, 4, 1); 
// dim3 blocksPerGrid(CEIL_DIV(DIM_J, 32), CEIL_DIV(DIM_I, 32), 1);
// WMMA_2D_RegBlock<<<blocksPerGrid, threadsPerBlock>>>(dA, dB, dC);

__global__ void WMMA_2D_RegBlock(_Float16 *A, _Float16 *B, float *C) {
    int lane = threadIdx.x; // Lane ID within the wave (0-31)
    int laneWrapped = lane % 16; // Lane ID within a WMMA subgroup (0-15)
    int laneGroup = lane / 16; // Which WMMA subgroup this lane belongs to (0 or 1)

    int wave_id = threadIdx.y; // Wavefront ID within the block
    int tIdx_z = wave_id % 2; // Maps wave to column partition in the 32x32 tile
    int tIdx_w = wave_id / 2; // Maps wave to row partition in the 32x32 tile

    int row_c = blockIdx.y * 32 + (tIdx_w * 16); // Starting row in global C for this wave
    int col_c = blockIdx.x * 32 + (tIdx_z * 16); // Starting column in global C for this wave

    float8 c_frag = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f}; // Wave local accumulator

    for (int k = 0; k < DIM_K; k += 16) {
        half8 a_frag; // Wave local A fragment
        half8 b_frag; // Wave local B fragment

        for(int ele = 0; ele < WMMA_DATA_WIDTH; ++ele) { // Load B fragment
            int r_B = k + (ele + laneGroup * WMMA_DATA_WIDTH);
            int c_B = col_c + laneWrapped;
            b_frag[ele] = B[r_B * DIM_J + c_B];
        }

        for(int ele = 0; ele < WMMA_DATA_WIDTH; ++ele) { // Load A fragment
            int r_A = row_c + laneWrapped;
            int c_A = k + (ele + laneGroup * WMMA_DATA_WIDTH);
            a_frag[ele] = A[r_A * DIM_K + c_A];
        }

        // Execute RDNA4 WMMA instruction
        c_frag = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12(a_frag, b_frag, c_frag);
    }

    for(int ele = 0; ele < WMMA_DATA_WIDTH; ++ele) { // Store results
        int r_C = row_c + (ele + laneGroup * WMMA_DATA_WIDTH);
        int c_C = col_c + laneWrapped;
        C[r_C * DIM_J + c_C] = c_frag[ele];
    }
}

__global__ void WMMA_2D_RB_LDS(_Float16 *A, _Float16 *B, float *C) {
    int lane = threadIdx.x;        // 0 to 31 (lanes in a wavefront)
    int wave_id = threadIdx.y;     // 0 to 3 (wavefronts in a block)
    int tid = wave_id * 32 + lane; // 0 to 127 (global thread ID in block)

    // Arrange the 4 waves into a 2x2 grid. Each wave computes a 32x32 sub-tile.
    int wave_i = wave_id / 2; 
    int wave_j = wave_id % 2; 

    // Calculate global starting coordinates for the 64x64 block
    int block_row = blockIdx.y * 64;
    int block_col = blockIdx.x * 64;

    // Calculate global starting coordinates for this specific wave's 32x32 sub-tile
    int row_c = block_row + (wave_i * 32);
    int col_c = block_col + (wave_j * 32);

    const int LDS_K = 32; // K dimension depth in Shared Memory
    const int PAD = 0; // Padding to break up power of 2 strides and prevent bank conflicts

    __shared__ _Float16 As[64 * (LDS_K + PAD)]; // Shared Memory for Matrix A
    __shared__ _Float16 Bs[LDS_K * (64 + PAD)]; // Shared Memory for Matrix B

    // Each wave computes four 16x16 tiles (a 2x2 grid of WMMA ops)
    float8 c_frag[2][2];
    #pragma unroll
    for(int i=0; i<2; ++i) {
        #pragma unroll
        for(int j=0; j<2; ++j) {
            c_frag[i][j] = (float8){0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f}; // Initialize fragments
        }
    }

    int laneWrapped = lane % 16; // Lane ID within a WMMA subgroup (0-15)
    int laneGroup = lane / 16; // Which WMMA subgroup this lane belongs to (0 or 1)

    for (int k = 0; k < DIM_K; k += LDS_K) { // Move through K dimension in LDS_K strides
        
        // Cooperative load from Global Memory to LDS
        // Load As: 64 rows, 32 cols = 2048 elements. 128 threads -> 16 elements/thread
        for (int idx = tid; idx < 64 * LDS_K; idx += 128) {
            int rA = idx / LDS_K; // Compute row in LDS
            int cA = idx % LDS_K; // Compute col in LDS
            As[rA * (LDS_K + PAD) + cA] = A[(block_row + rA) * DIM_K + (k + cA)]; // GMEM to LDS
        }
        
        // Load Bs: 32 rows, 64 cols = 2048 elements. 128 threads -> 16 elements/thread
        for (int idx = tid; idx < LDS_K * 64; idx += 128) {
            int rB = idx / 64; // Compute row in LDS
            int cB = idx % 64; // Compute col in LDS
            Bs[rB * (64 + PAD) + cB] = B[(k + rB) * DIM_J + (block_col + cB)]; // GMEM to LDS
        }
        
        __syncthreads(); // Sync memory loads

        //Compute Phase (Iterate over the LDS tile)
        for (int tk = 0; tk < LDS_K; tk += 16) {
            half8 a_frag[2]; // VGPR buffers for A fragments
            half8 b_frag[2]; // VGPR buffers for B fragments

            // Load A fragments from LDS into VGPRs (2 vertical tiles of 16x16)
            for(int i = 0; i < 2; ++i) {
                for(int ele = 0; ele < WMMA_DATA_WIDTH; ++ele) {
                    int r_A = (wave_i * 32) + (i * 16) + laneWrapped; // Calculate A row offset
                    int c_A = tk + (ele + laneGroup * WMMA_DATA_WIDTH); // Calculate A col offset
                    a_frag[i][ele] = As[r_A * (LDS_K + PAD) + c_A]; // Fetch from LDS
                }
            }

            // Load B fragments from LDS into VGPRs (2 horizontal tiles of 16x16)
            for(int j = 0; j < 2; ++j) {
                for(int ele = 0; ele < WMMA_DATA_WIDTH; ++ele) {
                    int r_B = tk + (ele + laneGroup * WMMA_DATA_WIDTH); // Calculate B row offset
                    int c_B = (wave_j * 32) + (j * 16) + laneWrapped; // Calculate B col offset
                    b_frag[j][ele] = Bs[r_B * (64 + PAD) + c_B]; // Fetch from LDS
                }
            }

            // Execute 2x2 = 4 WMMA instructions per wave
            for(int i = 0; i < 2; ++i) {
                for(int j = 0; j < 2; ++j) {
                    // C_frag += A_frag * B_frag
                    c_frag[i][j] = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12(a_frag[i], b_frag[j], c_frag[i][j]);
                }
            }
        }
        __syncthreads(); // Sync before next K iteration overwrites LDS
    }

    //Store the 32x32 computed chunk back to Global Memory
    for(int i = 0; i < 2; ++i) {
        for(int j = 0; j < 2; ++j) {
            for(int ele = 0; ele < WMMA_DATA_WIDTH; ++ele) {
                int r_C = row_c + (i * 16) + (ele + laneGroup * WMMA_DATA_WIDTH); // Global target row
                int c_C = col_c + (j * 16) + laneWrapped; // Global target column
                C[r_C * DIM_J + c_C] = c_frag[i][j][ele]; // Register to GMEM
            }
        }
    }
}

__global__ void WMMA_2D_RB_VecLoad_GMEM(_Float16 *A, _Float16 *B, float *C) {
    int lane = threadIdx.x; // Lane ID within the wave
    int wave_id = threadIdx.y; // Wave ID within the block
    int tid = wave_id * 32 + lane; // Global Thread ID in block

    int wave_i = wave_id / 2; // Maps wave to row partition
    int wave_j = wave_id % 2; // Maps wave to column partition

    int block_row = blockIdx.y * 64; // Block global row offset
    int block_col = blockIdx.x * 64; // Block global col offset

    int row_c = block_row + (wave_i * 32); // Wave global row offset
    int col_c = block_col + (wave_j * 32); // Wave global col offset

    const int LDS_K = 32; // K dimension depth in Shared Memory

    // alignas(16) guarantees that the arrays are 128-bit aligned in memory.
    // This is strictly required for safely casting to uint4 vectors.
    alignas(16) __shared__ _Float16 As[64 * LDS_K]; // Aligned shared memory for A
    alignas(16) __shared__ _Float16 Bs[LDS_K * 64]; // Aligned shared memory for B

    float8 c_frag[2][2]; // 2x2 grid of wave accumulators
    #pragma unroll
    for(int i=0; i<2; ++i) {
        #pragma unroll
        for(int j=0; j<2; ++j) {
            c_frag[i][j] = (float8){0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f}; // Initialize fragments
        }
    }

    int laneWrapped = lane % 16; // Lane ID within a WMMA subgroup (0-15)
    int laneGroup = lane / 16; // Which WMMA subgroup this lane belongs to (0 or 1)

    for (int k = 0; k < DIM_K; k += LDS_K) { // Step across K dimension
        
        //Vectorized Cooperative Load (Global to LDS)
        //We use uint4 to move 128 bits (8 x _Float16) in a single instruction.

        // A Matrix: 64 rows, 32 cols. Each row has 32/8 = 4 chunks of 128bits.
        // Total chunks = 64 * 4 = 256. With 128 threads, each thread loads 2 chunks.
        for (int idx = tid; idx < 256; idx += 128) {
            int row = idx / 4; // Row target in LDS
            int col_chunk = idx % 4; // Col chunk target in LDS
            
            // Cast pointers to uint4 to trigger 128bit wide vectorized load from GMEM
            reinterpret_cast<uint4*>(As)[row * 4 + col_chunk] = reinterpret_cast<uint4*>(A + (block_row + row) * DIM_K + k)[col_chunk];
        }
        
        // B Matrix: 32 rows, 64 cols. Each row has 64/8 = 8 chunks of 128bits.
        // Total chunks = 32 * 8 = 256. With 128 threads, each thread loads 2 chunks.
        for (int idx = tid; idx < 256; idx += 128) {
            int row = idx / 8; // Row target in LDS
            int col_chunk = idx % 8; // Col chunk target in LDS
            
            // Cast pointers to uint4 to trigger 128bit wide vectorized load from GMEM
            reinterpret_cast<uint4*>(Bs)[row * 8 + col_chunk] = reinterpret_cast<uint4*>(B + (k + row) * DIM_J + block_col)[col_chunk];
        }
        
        __syncthreads(); // Sync memory loads

        // Compute Phase
        for (int tk = 0; tk < LDS_K; tk += 16) {
            half8 a_frag[2]; // VGPR buffers for A fragments
            half8 b_frag[2]; // VGPR buffers for B fragments

            // Not vectorized LDS to VGPR (Matrix A)
            // Reverted to scalar loads to isolate the performance of global vectorization.
            for(int i = 0; i < 2; ++i) {
                #pragma unroll
                for(int ele = 0; ele < WMMA_DATA_WIDTH; ++ele) {
                    int r_A = (wave_i * 32) + (i * 16) + laneWrapped; // Row offset
                    int c_A = tk + (ele + laneGroup * WMMA_DATA_WIDTH); // Col offset
                    a_frag[i][ele] = As[r_A * LDS_K + c_A]; // Scalar load from LDS
                }
            }

            // Not vectorized LDS -> VGPR (Matrix B)
            for(int j = 0; j < 2; ++j) {
                #pragma unroll
                for(int ele = 0; ele < WMMA_DATA_WIDTH; ++ele) {
                    int r_B = tk + (ele + laneGroup * WMMA_DATA_WIDTH); // Row offset
                    int c_B = (wave_j * 32) + (j * 16) + laneWrapped; // Col offset
                    b_frag[j][ele] = Bs[r_B * 64 + c_B]; // Scalar load from LDS
                }
            }

            // Execute WMMA Instructions
            for(int i = 0; i < 2; ++i) {
                for(int j = 0; j < 2; ++j) {
                    c_frag[i][j] = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12(a_frag[i], b_frag[j], c_frag[i][j]);
                }
            }
        }
        __syncthreads(); // Sync before next K iteration
    }

    //Store the 64x64 computed chunk back to Global Memory
    for(int i = 0; i < 2; ++i) {
        for(int j = 0; j < 2; ++j) {
            for(int ele = 0; ele < WMMA_DATA_WIDTH; ++ele) {
                int r_C = row_c + (i * 16) + (ele + laneGroup * WMMA_DATA_WIDTH); // Global row target
                int c_C = col_c + (j * 16) + laneWrapped; // Global col target
                C[r_C * DIM_J + c_C] = c_frag[i][j][ele]; // Register to GMEM
            }
        }
    }
}

__global__ void WMMA_2D_RB_VecLoad_Full(_Float16 *A, _Float16 *B, float *C) {
    int lane = threadIdx.x;        // Lane ID within the wave
    int wave_id = threadIdx.y;     // Wave ID within the block
    int tid = wave_id * 32 + lane; // Global Thread ID in block

    int wave_i = wave_id / 2; // Maps wave to row partition
    int wave_j = wave_id % 2; // Maps wave to column partition

    int block_row = blockIdx.y * 64; // Block global row offset
    int block_col = blockIdx.x * 64; // Block global col offset

    int row_c = block_row + (wave_i * 32); // Wave global row offset
    int col_c = block_col + (wave_j * 32); // Wave global col offset

    const int LDS_K = 32; // K dimension depth in LDS

    // alignas(16) guarantees that the arrays are 128bit aligned in memory.
    // This is strictly required for safely casting to uint4 vectors.
    alignas(16) __shared__ _Float16 As[64 * LDS_K]; // Aligned shared memory for A
    alignas(16) __shared__ _Float16 Bs[LDS_K * 64]; // Aligned shared memory for B

    float8 c_frag[2][2]; // 2x2 grid of wave accumulators
    #pragma unroll
    for(int i=0; i<2; ++i) {
        #pragma unroll
        for(int j=0; j<2; ++j) {
            c_frag[i][j] = (float8){0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f}; // Initialize fragments
        }
    }

    int laneWrapped = lane % 16; // Lane ID within a WMMA subgroup (0-15)
    int laneGroup = lane / 16; // Which WMMA subgroup this lane belongs to (0 or 1)

    for (int k = 0; k < DIM_K; k += LDS_K) {
        
        // Vectorized Cooperative Load (Global -> LDS)
        // We use uint4 to move 128 bits (8 x _Float16) in a single instruction.

        // A Matrix: 64 rows, 32 cols. Each row has 32/8 = 4 chunks of 128bits.
        // Total chunks = 64 * 4 = 256. With 128 threads, each thread loads 2 chunks.
        for (int idx = tid; idx < 256; idx += 128) {
            int row = idx / 4; // Row target in LDS
            int col_chunk = idx % 4; // Col chunk target in LDS
            
            reinterpret_cast<uint4*>(As)[row * 4 + col_chunk] = 
                reinterpret_cast<uint4*>(A + (block_row + row) * DIM_K + k)[col_chunk]; // Vector load GMEM to LDS
        }
        
        // B Matrix: 32 rows, 64 cols. Each row has 64/8 = 8 chunks of 128bits.
        // Total chunks = 32 * 8 = 256. With 128 threads, each thread loads 2 chunks.
        for (int idx = tid; idx < 256; idx += 128) {
            int row = idx / 8; // Row target in LDS
            int col_chunk = idx % 8; // Col chunk target in LDS
            
            reinterpret_cast<uint4*>(Bs)[row * 8 + col_chunk] = 
                reinterpret_cast<uint4*>(B + (k + row) * DIM_J + block_col)[col_chunk]; // Vector load GMEM to LDS
        }
        
        __syncthreads(); 

        // Compute Phase
        for (int tk = 0; tk < LDS_K; tk += 16) {
            half8 a_frag[2]; // VGPR buffers for A fragments
            half8 b_frag[2]; // VGPR buffers for B fragments

            // Vectorized LDS to VGPR (Matrix A)
            // Because threads read A contiguously across rows, we can pull all 8 elements directly into the half8 vector in one shot.
            for(int i = 0; i < 2; ++i) {
                int r_A = (wave_i * 32) + (i * 16) + laneWrapped; // Calculate A row offset
                int c_A = tk + (laneGroup * WMMA_DATA_WIDTH); // Calculate A col offset (start of vector)
                
                // Vectorized read from LDS to VGPR using half8 cast
                a_frag[i] = *(reinterpret_cast<half8*>(&As[r_A * LDS_K + c_A]));
            }

            // Not vectorized LDS to VGPR (Matrix B)
            // Because Matrix B is accessed column-wise (strided), we cannot fetch it safely with a 128 bit vector yet
            for(int j = 0; j < 2; ++j) {
                #pragma unroll
                for(int ele = 0; ele < WMMA_DATA_WIDTH; ++ele) {
                    int r_B = tk + (ele + laneGroup * WMMA_DATA_WIDTH); // Calculate B row offset
                    int c_B = (wave_j * 32) + (j * 16) + laneWrapped; // Calculate B col offset
                    b_frag[j][ele] = Bs[r_B * 64 + c_B]; // Scalar load from LDS to VGPR
                }
            }

            // Execute WMMA Instructions
            for(int i = 0; i < 2; ++i) {
                for(int j = 0; j < 2; ++j) {
                    c_frag[i][j] = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12(a_frag[i], b_frag[j], c_frag[i][j]);
                }
            }
        }
        __syncthreads();
    }

    //Store the 64x64 computed chunk back to GMEM
    for(int i = 0; i < 2; ++i) {
        for(int j = 0; j < 2; ++j) {
            for(int ele = 0; ele < WMMA_DATA_WIDTH; ++ele) {
                int r_C = row_c + (i * 16) + (ele + laneGroup * WMMA_DATA_WIDTH); // Global row target
                int c_C = col_c + (j * 16) + laneWrapped; // Global col target
                C[r_C * DIM_J + c_C] = c_frag[i][j][ele]; // Register to GMEM
            }
        }
    }
}

/* Thread block size is defined as (32, TOTAL_WAVES, 1), where TOTAL_WAVES = WAVES_I * WAVES_J. This is (WAVES_I * WAVES_J * 32) total threads.

    Grid dimensions are scaled based on the block's output tile size
    CEIL_DIV(DIM_J, BLOCK_N) by CEIL_DIV(DIM_I, BLOCK_M), where:
    BLOCK_M = WAVES_I * WAVE_M * 16
    BLOCK_N = WAVES_J * WAVE_N * 16

    Register blocking logic: Each wave holds a 2D grid of (WAVE_MxWAVE_N) c_frag accumulators.
    By loading WAVE_M fragments of A and WAVE_N fragments of B, each wave computes (WAVE_M*WAVE_N) output fragments per WMMA iteration. 
    This reduces the frequency of LDS to VGPR memory instructions.

    Memory alignment: added alignas(16) to the LDS arrays (As and Bs) to ensure 128-bit boundaries, enabling safe vectorized access.

    Vectorized global loads: replaced scalar loops with uint4 pointer casts for global to LDS transfers.
    This moves 8 half-precision elements (128 bits) per instruction,
    reducing the total number of load instructions by a factor of 8.
*/
template <int WAVE_M, int WAVE_N, int WAVES_I, int WAVES_J>
__global__ void WMMA_Templated_Base(_Float16 *A, _Float16 *B, float *C) {
 
    int lane = threadIdx.x; // Lane ID within the wave
    int wave_id = threadIdx.y; // Wave ID within the block
    int tid = wave_id * 32 + lane; // Global Thread ID in block
    constexpr int TOTAL_WAVES = WAVES_I * WAVES_J; // Total wavefronts per block
    constexpr int TOTAL_THREADS = TOTAL_WAVES * 32; // Total threads per block
 
    int wave_i = wave_id / WAVES_J;   // Maps wave to row partition
    int wave_j = wave_id % WAVES_J;   // Maps wave to column partition
 
    constexpr int LDS_K = 32;                 // Depth of the K dimension tile in LDS 
    constexpr int TILE_M = WAVE_M * 16;       // Number of rows of A processed by each wave
    constexpr int TILE_N = WAVE_N * 16;       // Number of columns of B processed by each wave
    constexpr int BLOCK_M = WAVES_I * TILE_M; // Total rows of A processed by the entire block
    constexpr int BLOCK_N = WAVES_J * TILE_N; // Total columns of B processed by the entire block
 
    int block_row = blockIdx.y * BLOCK_M; // Block global row offset
    int block_col = blockIdx.x * BLOCK_N; // Block global col offset
 
    int row_c = block_row + wave_i * TILE_M;  // Wave global row offset
    int col_c = block_col + wave_j * TILE_N;  // Wave global col offset
 
    alignas(16) __shared__ _Float16 As[BLOCK_M * LDS_K]; // Aligned LDS memory for Matrix A
    alignas(16) __shared__ _Float16 Bs[LDS_K * BLOCK_N]; // Aligned LDS memory for Matrix B
 
    float8 c_frag[WAVE_M][WAVE_N]; // Accumulator register buffer for this wave
    #pragma unroll
    for (int i = 0; i < WAVE_M; ++i)
        #pragma unroll
        for (int j = 0; j < WAVE_N; ++j)
            c_frag[i][j] = (float8){0,0,0,0,0,0,0,0}; // Initialize to zero
 
    int laneWrapped = lane % 16;   // Lane ID within a WMMA subgroup (0-15)
    int laneGroup   = lane / 16;   // Which WMMA subgroup this lane belongs to (0 or 1)
 
    constexpr int aChunksPerRow = LDS_K / 8; // Number of 128bit chunks per row in A's LDS tile
    constexpr int totalAChunks = BLOCK_M * aChunksPerRow; // Total 128bit chunks for A
    constexpr int bChunksPerRow = BLOCK_N / 8; // Number of 128bit chunks per row in B's LDS tile
    constexpr int totalBChunks = LDS_K * bChunksPerRow; // Total 128 bit chunks for B

    // Safety checks to ensure our 1 to 1 mapping holds true
    static_assert(totalAChunks == TOTAL_THREADS, "Thread count must equal A chunks to remove the loop");
    static_assert(totalBChunks == TOTAL_THREADS, "Thread count must equal B chunks to remove the loop");
 
    for (int k = 0; k < DIM_K; k += LDS_K) { 
        
        // Matrix A Load
        int row_A = tid / aChunksPerRow; // Determine row of matrix A for this thread
        int col_chunk_A = tid % aChunksPerRow; // Determine column chunk for this thread
        reinterpret_cast<uint4*>(As)[row_A * aChunksPerRow + col_chunk_A] = 
            reinterpret_cast<uint4*>(A + (block_row + row_A) * DIM_K + k)[col_chunk_A]; // Vector load GMEM -> LDS
 
        // Matrix B Load
        int row_B = tid / bChunksPerRow; // Determine row of matrix B for this thread
        int col_chunk_B = tid % bChunksPerRow; // Determine column chunk for this thread
        reinterpret_cast<uint4*>(Bs)[row_B * bChunksPerRow + col_chunk_B] = 
            reinterpret_cast<uint4*>(B + (k + row_B) * DIM_J + block_col)[col_chunk_B]; // Vector load GMEM -> LDS
 
        __syncthreads(); // Wait for loads to finish
 
        for (int tk = 0; tk < LDS_K; tk += 16) { // Step through LDS_K
            half8 b_frag[WAVE_N]; // Local register buffer for B fragments
 
            // Pre load B fragments to registers
            #pragma unroll
            for (int j = 0; j < WAVE_N; ++j) {
                #pragma unroll
                for (int ele = 0; ele < WMMA_DATA_WIDTH; ++ele) {
                    int r_B = tk + (ele + laneGroup * WMMA_DATA_WIDTH); // LDS row index
                    int c_B = (wave_j * TILE_N) + (j * 16) + laneWrapped; // LDS col index
                    b_frag[j][ele] = Bs[r_B * BLOCK_N + c_B]; // Scalar load from LDS to VGPR
                }
            }
 
            //Compute Phase: Load A fragments on the fly inside the loop to save VGPRs
            #pragma unroll
            for (int i = 0; i < WAVE_M; ++i) {
                // Fetch a single a_frag from LDS dynamically 
                int r_A = (wave_i * TILE_M) + (i * 16) + laneWrapped; // LDS row index
                int c_A = tk + (laneGroup * WMMA_DATA_WIDTH); // LDS col index
                half8 a_frag = *(reinterpret_cast<half8*>(&As[r_A * LDS_K + c_A])); // Vectorized read LDS to VGPR
                
                #pragma unroll
                for (int j = 0; j < WAVE_N; ++j) {
                    c_frag[i][j] = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12(a_frag, b_frag[j], c_frag[i][j]); 
                }
            }
        }
 
        __syncthreads(); // Sync before overwriting LDS for next K iteration
    }
 
    #pragma unroll
    for (int i = 0; i < WAVE_M; ++i)
        #pragma unroll
        for (int j = 0; j < WAVE_N; ++j)
            #pragma unroll
            for (int ele = 0; ele < WMMA_DATA_WIDTH; ++ele) {
                int r_C = row_c + (i * 16) + (ele + laneGroup * WMMA_DATA_WIDTH); // Global row target
                int c_C = col_c + (j * 16) + laneWrapped; // Global col target
                C[r_C * DIM_J + c_C] = c_frag[i][j][ele]; // Register to GMEM store
            }
}


/* The bottleneck in the previous version was matrix A's vectorized load from global memory to LDS. This caused an 8 way bank conflict in LDS. It looked like this:
    Thread 0 read row 0 (Starts at byte 0 ➔ Bank 0)
    Thread 1 read row 1 (Starts at byte 64 ➔ Bank 16)
    Thread 2 read row 2 (Starts at byte 128 ➔ Bank 0)
    Thread 3 read row 3 (Starts at byte 192 ➔ Bank 16)

    To fix this we added padding to the LDS arrays to break the conflict. It is now a 2 way conflict instead of 8 way, which gave a significant boost in performance.
    This padding did not break the 128 bit alignement required for the vectorized loads because we carefully calculated the starting address for each thread's load to skip over the padding.
    Also changed the way the LDS size is allocated and created the STRIDE variables to calculate the correct stride, now that the padding was added, and thus the LDS arrays are not perfectly
    dimensioned to the block size anymore. The STRIDE variables are also used for the LDS to Register reads.

    Removed the chunk loop for global to LDS loads by ensuring a 1 to 1 mapping between threads and the number of 128 bit chunks we need to load for both A and B.

    Moved the A fragment load from LDS to VGPRs inside the compute loop to save registers. This means we load each A fragment from LDS multiple times (once per WMMA instruction)
    but it allows us to keep more of B in registers and reduces register pressure overall.

    See comments inside the kernel for more detailed explanations of each change.

*/

template <int WAVE_M, int WAVE_N, int WAVES_I, int WAVES_J>
__global__ void WMMA_LDS_Padded(_Float16 *A, _Float16 *B, float *C) {
 
// Thread & Wave Identification 
    int laneId = threadIdx.x; 
    int waveId = threadIdx.y; 
    int threadIdInBlock = (waveId * 32) + laneId; 
 
    int waveRow = waveId / WAVES_J;   
    int waveCol = waveId % WAVES_J;   
 
    // Matrix Dimensions & Tiling
    constexpr int LDS_K = 32; // K dimension depth of a tile in LDS      
    constexpr int WAVE_M_SIZE = WAVE_M * 16; // sub tile assigned to a single wavefront        
    constexpr int WAVE_N_SIZE = WAVE_N * 16;        
    constexpr int BLOCK_M_SIZE = WAVES_I * WAVE_M_SIZE; // workgroup (block) level tile size
    constexpr int BLOCK_N_SIZE = WAVES_J * WAVE_N_SIZE; 
 
    // Global Block Coordinates 
    int blockRowStart = blockIdx.y * BLOCK_M_SIZE;
    int blockColStart = blockIdx.x * BLOCK_N_SIZE;
 
    int globalRowC = blockRowStart + (waveRow * WAVE_M_SIZE);  
    int globalColC = blockColStart + (waveCol * WAVE_N_SIZE);  
 
    // Change 1: Added memory padding
    // Previous: Arrays were perfectly dimensioned (LDS_K and BLOCK_N).
    // New: Added an 8-element (16-byte) padding to the width of the shared memory rows.
    // Why: RDNA4 LDS has 32 memory banks (4 bytes each), wrapping every 128 bytes. 
    // The original power of 2 strides caused threads to request data from 
    // the exact same bank at the same time (8 way conflict on Matrix A, 2 way on Matrix B). 
    // The padding shifts each row's starting address, forcing threads to access different 
    // banks while maintaining the strict 16-byte alignment required for uint4/half8 casting.
    constexpr int PAD = 8;
    constexpr int STRIDE_A = LDS_K + PAD;    // 32 + 8 = 40 elements
    constexpr int STRIDE_B = BLOCK_N_SIZE + PAD;  // 128 + 8 = 136 elements
 
    // Change 2: Allocating padded LDS
    // Previous: __shared__ _Float16 As[BLOCK_M * LDS_K];
    // New: We now allocate total size based on the new STRIDE variables.
    alignas(16) __shared__ _Float16 As[BLOCK_M_SIZE * STRIDE_A];
    alignas(16) __shared__ _Float16 Bs[LDS_K * STRIDE_B];
 
    // Accumulator Initialization
    float8 c_frag[WAVE_M][WAVE_N];
    #pragma unroll
    for (int i = 0; i < WAVE_M; ++i) {
        #pragma unroll
        for (int j = 0; j < WAVE_N; ++j) {
            c_frag[i][j] = float8{0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
        }
    }
 
    // WMMA Hardware Mapping 
    int wmmaLaneWrapped = laneId % 16;   
    int wmmaLaneGroup = laneId / 16;   
 
    // Chunked Memory Load Calculations 
    constexpr int ELEMENTS_PER_CHUNK = 8; // 128 bit chunk = 8 x _Float16
    constexpr int TOTAL_WAVES = WAVES_I * WAVES_J; 
    constexpr int TOTAL_THREADS = TOTAL_WAVES * 32; 

    constexpr int aChunksPerRow = LDS_K / ELEMENTS_PER_CHUNK; 
    constexpr int totalAChunks = BLOCK_M_SIZE * aChunksPerRow;

    // Change 3: Separating global chunks from LDS stride
    // Previous: bChunksPerRow was calculated using the LDS stride.
    // New: We specifically calculate bGlobalChunksPerRow using BLOCK_N (the unpadded width).
    // Why: The number of chunks we read from global memory must perfectly match the 
    // total number of threads (TOTAL_THREADS) so we don't need a loop. If we calculated 
    // this using STRIDE_B, the compiler would think we need to read the padding from 
    // global VRAM, breaking the 1 to 1 thread mapping and failing the static_assert.
    constexpr int bGlobalChunksPerRow = BLOCK_N_SIZE / ELEMENTS_PER_CHUNK; 
    constexpr int totalBChunks = LDS_K * bGlobalChunksPerRow;

    static_assert(totalAChunks == TOTAL_THREADS, "Thread count must equal A chunks");
    static_assert(totalBChunks == TOTAL_THREADS, "Thread count must equal B chunks");
 
    // Compute Loop
    for (int k = 0; k < DIM_K; k += LDS_K) {
        
        //Matrix A Load
        int rowA = threadIdInBlock / aChunksPerRow;
        int colChunkA = threadIdInBlock % aChunksPerRow; 

        // Change 4: Writing to Padded LDS Addresses (Matrix A)
        // Previous: reinterpret_cast<uint4*>(As)[rowA * aChunksPerRow + colChunkA]
        // New: We calculate the exact starting address of the 8 element block using STRIDE_A, 
        // then cast that specific memory address to a uint4 pointer for the 128-bit write.
        // Why: Because of the padding, the LDS array is no longer a contiguous block of valid 
        // data. We have to skip over the 8 padding elements at the end of every row.
        int globalRowStartA = (blockRowStart + rowA) * DIM_K + k;
        int ldsRowStartA = rowA * STRIDE_A;

        uint4* globalPtrA = reinterpret_cast<uint4*>(&A[globalRowStartA]);
        uint4* ldsPtrA = reinterpret_cast<uint4*>(&As[ldsRowStartA]);
        
        ldsPtrA[colChunkA] = globalPtrA[colChunkA];
 
        //Matrix B Load 
        int rowB = threadIdInBlock / bGlobalChunksPerRow;
        int colChunkB = threadIdInBlock % bGlobalChunksPerRow;

        // Change 5: Writing to Padded LDS Addresses (Matrix B)
        // Same logic as Matrix A. We use STRIDE_B to calculate the destination index, 
        // ensuring the global data is written correctly while leaving the padding gaps empty.
        int globalRowStartB = (k + rowB) * DIM_J + blockColStart;
        int ldsRowStartB = rowB * STRIDE_B;

        uint4* globalPtrB = reinterpret_cast<uint4*>(&B[globalRowStartB]);
        uint4* ldsPtrB = reinterpret_cast<uint4*>(&Bs[ldsRowStartB]);

        ldsPtrB[colChunkB] = globalPtrB[colChunkB];
 
        __syncthreads();
 
        // WMMA Math 
        for (int tk = 0; tk < LDS_K; tk += 16) {
            half8 b_frag[WAVE_N];
 
            // Pre load B fragments using scalar loads
            #pragma unroll
            for (int j = 0; j < WAVE_N; ++j) {
                #pragma unroll
                for (int ele = 0; ele < WMMA_DATA_WIDTH; ++ele) {
                    int rowInLdsB = tk + (ele + wmmaLaneGroup * WMMA_DATA_WIDTH);
                    int colInLdsB = (waveCol * WAVE_N_SIZE) + (j * 16) + wmmaLaneWrapped;

                    // Change 6: Reading Matrix B with Stride
                    // Previous: Bs[rowInLdsB * BLOCK_N + colInLdsB]
                    // New: Multiplied row index (rowInLdsB) by STRIDE_B.
                    // Why: To fetch the correct value, we must account for the padded row length.
                    b_frag[j][ele] = Bs[(rowInLdsB * STRIDE_B) + colInLdsB]; 
                }
            }
 
            // Compute Phase: Fast Vectorized Load for A
            #pragma unroll
            for (int i = 0; i < WAVE_M; ++i) {
                
                int rowInLdsA = (waveRow * WAVE_M_SIZE) + (i * 16) + wmmaLaneWrapped;
                int colInLdsA = tk + (wmmaLaneGroup * WMMA_DATA_WIDTH);

                // Change 7: Vectorized read matrix A with stride
                // Previous: As[rowInLdsA * LDS_K + colInLdsA]
                // New: Multiplied row index (rowInLdsA) by STRIDE_A.
                // Why: Because STRIDE_A is padded by 8, when the 32 threads in the wave 
                // execute this 128-bit cast simultaneously, their memory requests map 
                // perfectly across the 32 physical LDS memory banks. The massive hardware 
                // stall (s_barrier_wait) is reduced.
                half8* ldsPtrA_frag = reinterpret_cast<half8*>(&As[(rowInLdsA * STRIDE_A) + colInLdsA]);
                half8 a_frag = ldsPtrA_frag[0];
                //asm volatile("" ::: "memory");
                
                #pragma unroll
                for (int j = 0; j < WAVE_N; ++j) {
                    c_frag[i][j] = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12(a_frag, b_frag[j], c_frag[i][j]);
                }
            }
        } 


 
        __syncthreads();
    }
 
    //Store Results to Global Memory 
    #pragma unroll
    for (int i = 0; i < WAVE_M; ++i) {
        #pragma unroll
        for (int j = 0; j < WAVE_N; ++j) {
            #pragma unroll
            for (int ele = 0; ele < WMMA_DATA_WIDTH; ++ele) {
                
                int targetRowC = globalRowC + (i * 16) + (ele + wmmaLaneGroup * WMMA_DATA_WIDTH);
                int targetColC = globalColC + (j * 16) + wmmaLaneWrapped;
                
                C[targetRowC * DIM_J + targetColC] = c_frag[i][j][ele];
            }
        }
    }
}

/* One bottleneck in the previous version was the remaining 2 way bank conflict during vectorized loads and the wasted shared memory from padding.
   While padding reduced the conflict from 8 way to 2 way, it didn't perfectly resolve it and consumed extra LDS space.

   To fix this, we replaced padding with XOR swizzling. This achieves a perfect 0 way bank conflict, giving the maximum possible performance boost without wasting any LDS memory.
   Swizzling works by scrambling the column index based on the row index when writing to LDS. Threads that would normally write to the same memory bank in the same cycle are shifted to different banks.
   When reading from LDS into registers, we apply the inverse XOR operation to reconstruct the original data perfectly, ensuring our 128bit vectorized loads remain perfectly aligned.

   Memory dimensions are returned to their natural power of 2 sizes, keeping the memory footprint minimal.
   
   Removed the PAD variable entirely and returned the STRIDE variables to their unpadded sizes.

   See comments inside the kernel for more detailed explanations of each change.

*/
template <int WAVE_M, int WAVE_N, int WAVES_I, int WAVES_J>
__global__ void WMMA_LDS_Swizzled(_Float16 *A, _Float16 *B, float *C) {
 
    // Thread & Wave Identification 
    int laneId = threadIdx.x; 
    int waveId = threadIdx.y; 
    int threadIdInBlock = (waveId * 32) + laneId; 
 
    int waveRow = waveId / WAVES_J;   
    int waveCol = waveId % WAVES_J;   
 
    // Matrix Dimensions & Tiling
    constexpr int LDS_K = 32;      
    constexpr int WAVE_M_SIZE = WAVE_M * 16;        
    constexpr int WAVE_N_SIZE = WAVE_N * 16;        
    constexpr int BLOCK_M_SIZE = WAVES_I * WAVE_M_SIZE; 
    constexpr int BLOCK_N_SIZE = WAVES_J * WAVE_N_SIZE; 
 
    // Global Block Coordinates 
    int blockRowStart = blockIdx.y * BLOCK_M_SIZE;
    int blockColStart = blockIdx.x * BLOCK_N_SIZE;
 
    int globalRowC = blockRowStart + (waveRow * WAVE_M_SIZE);  
    int globalColC = blockColStart + (waveCol * WAVE_N_SIZE);  
 
    // Change 1: Removed memory padding
    // Previous: constexpr int STRIDE_A = LDS_K + PAD;
    // New: Arrays are strictly dimensioned to their natural power of 2 sizes.
    // Why: Since we are using XOR swizzling to break the LDS memory bank conflicts,
    // we no longer need to waste shared memory on empty padding elements.
    constexpr int STRIDE_A = LDS_K;             // 32 elements
    constexpr int STRIDE_B = BLOCK_N_SIZE;      // 128 elements 
 
    // Change 2: Allocating unpadded LDS
    // Previous: alignas(16) __shared__ _Float16 As[BLOCK_M_SIZE * (LDS_K + PAD)];
    // New: We allocate exactly the memory needed for the tiles using the natural strides.
    alignas(16) __shared__ _Float16 As[BLOCK_M_SIZE * STRIDE_A];
    alignas(16) __shared__ _Float16 Bs[LDS_K * STRIDE_B]; 
 
    // Accumulator Initialization
    float8 c_frag[WAVE_M][WAVE_N];
    #pragma unroll
    for (int i = 0; i < WAVE_M; ++i) {
        #pragma unroll
        for (int j = 0; j < WAVE_N; ++j) {
            c_frag[i][j] = float8{0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
        }
    }
 
    // WMMA Hardware Mapping 
    int wmmaLaneWrapped = laneId % 16;   
    int wmmaLaneGroup = laneId / 16;   
 
    // Chunked Memory Load Calculations 
    constexpr int ELEMENTS_PER_CHUNK = 8; 
    constexpr int TOTAL_WAVES = WAVES_I * WAVES_J; 
    constexpr int TOTAL_THREADS = TOTAL_WAVES * 32; 

    constexpr int aChunksPerRow = LDS_K / ELEMENTS_PER_CHUNK; 
    constexpr int totalAChunks = BLOCK_M_SIZE * aChunksPerRow;

    constexpr int bGlobalChunksPerRow = BLOCK_N_SIZE / ELEMENTS_PER_CHUNK; 
    constexpr int totalBChunks = LDS_K * bGlobalChunksPerRow;

    static_assert(totalAChunks == TOTAL_THREADS, "Thread count must equal A chunks");
    static_assert(totalBChunks == TOTAL_THREADS, "Thread count must equal B chunks");
 
    // Compute Loop
    for (int k = 0; k < DIM_K; k += LDS_K) {
        
        // Matrix A Load
        int rowA = threadIdInBlock / aChunksPerRow;
        int colChunkA = threadIdInBlock % aChunksPerRow; 

        // Change 3: Swizzled Writing to LDS Addresses (A)
        // Previous: ldsPtrA[colChunkA] = globalPtrA[colChunkA]; 
        // New: We apply a 4-way XOR swizzle using the row index to scramble the column destination.
        // Why: RDNA4 LDS banks wrap every 128 bytes. By XORing the 128-bit chunk column index
        // with the lower bits of the row index, we ensure that consecutive rows do not map
        // to the exact same physical memory bank, eliminating the write conflict entirely.
        int swizzleA = (rowA & 3) ^ ((rowA >> 2) & 3);
        int swizzledColChunkA = colChunkA ^ swizzleA;

        int globalRowStartA = (blockRowStart + rowA) * DIM_K + k;
        int ldsRowStartA = rowA * STRIDE_A;

        uint4* globalPtrA = reinterpret_cast<uint4*>(&A[globalRowStartA]);
        uint4* ldsPtrA = reinterpret_cast<uint4*>(&As[ldsRowStartA]);
        
        // Write to the scrambled LDS index
        ldsPtrA[swizzledColChunkA] = globalPtrA[colChunkA];
 
        // Matrix B Load
        int rowB = threadIdInBlock / bGlobalChunksPerRow;
        int colChunkB = threadIdInBlock % bGlobalChunksPerRow;

        // Change 4: Swizzled Writing to LDS Addresses (B)
        // Same logic as Matrix A. We use the row index to scramble the column write address,
        // completely eliminating write bank conflicts for Matrix B.
        int swizzleB = (rowB & 15) ^ (((rowB >> 3) & 1) << 1);
        int swizzledColChunkB = colChunkB ^ swizzleB;

        int globalRowStartB = (k + rowB) * DIM_J + blockColStart;
        int ldsRowStartB = rowB * STRIDE_B;

        uint4* globalPtrB = reinterpret_cast<uint4*>(&B[globalRowStartB]);
        uint4* ldsPtrB = reinterpret_cast<uint4*>(&Bs[ldsRowStartB]);

        // Write contiguous 128-bit chunks to swizzled LDS locations
        ldsPtrB[swizzledColChunkB] = globalPtrB[colChunkB];
 
        __syncthreads();
 
        // WMMA Math 
        for (int tk = 0; tk < LDS_K; tk += 16) {
            half8 b_frag[WAVE_N];
 
            // Load B fragments using scalar loads with the inverse swizzle mapping
            
            for (int j = 0; j < WAVE_N; ++j) {
                for (int ele = 0; ele < WMMA_DATA_WIDTH; ++ele) {
                    int rowInLdsB = tk + (ele + wmmaLaneGroup * WMMA_DATA_WIDTH);
                    int colInLdsB = (waveCol * WAVE_N_SIZE) + (j * 16) + wmmaLaneWrapped;

                    // Change 5: Reading Matrix B with Inverse Swizzle
                    // Previous: b_frag[j][ele] = Bs[(rowInLdsB * STRIDE_B) + colInLdsB];
                    // New: We apply the inverse XOR bitwise math to the column read index.
                    // Why: Because we scrambled the data when writing it to LDS, we must unscramble it
                    // during the read to reconstruct the original, unconflicted element location.
                    int swizzleReadB = (rowInLdsB & 15) ^ (((rowInLdsB >> 3) & 1) << 1);
                    int swizzledColInLdsB = colInLdsB ^ (swizzleReadB * 8);

                    b_frag[j][ele] = Bs[(rowInLdsB * STRIDE_B) + swizzledColInLdsB]; 
                }
            }
 
            // Compute Phase: Fast Vectorized Load for A
            #pragma unroll
            for (int i = 0; i < WAVE_M; ++i) {
                
                int rowInLdsA = (waveRow * WAVE_M_SIZE) + (i * 16) + wmmaLaneWrapped;
                int colInLdsA = tk + (wmmaLaneGroup * WMMA_DATA_WIDTH);

                // Change 6: Vectorized read matrix A with Inverse Swizzle
                // Previous: reinterpret_cast<const half8*>(&As[(rowInLdsA * STRIDE_A) + colInLdsA]);
                // New: We apply the inverse swizzle mapping before casting the 128-bit vector.
                // Why: This fetches the contiguous 8-element chunk that was scattered during the write phase.
                // Because of the swizzled layout, when the 32 threads execute this read simultaneously,
                // it achieves a perfect 0-way bank conflict on the hardware, maximizing throughput.
                int swizzleReadA = (rowInLdsA & 3) ^ ((rowInLdsA >> 2) & 3);
                int swizzledColInLdsA = colInLdsA ^ (swizzleReadA * 8);

                half8* ldsPtrA_frag = reinterpret_cast<half8*>(&As[(rowInLdsA * STRIDE_A) + swizzledColInLdsA]);
                half8 a_frag = ldsPtrA_frag[0];
                
                #pragma unroll
                for (int j = 0; j < WAVE_N; ++j) {
                    c_frag[i][j] = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12(a_frag, b_frag[j], c_frag[i][j]);
                }
            }
        } 
 
        __syncthreads();
    }
 
    // Store Results to Global Memory 
    #pragma unroll
    for (int i = 0; i < WAVE_M; ++i) {
        #pragma unroll
        for (int j = 0; j < WAVE_N; ++j) {
            #pragma unroll
            for (int ele = 0; ele < WMMA_DATA_WIDTH; ++ele) {
                
                int targetRowC = globalRowC + (i * 16) + (ele + wmmaLaneGroup * WMMA_DATA_WIDTH);
                int targetColC = globalColC + (j * 16) + wmmaLaneWrapped;
                
                C[targetRowC * DIM_J + targetColC] = c_frag[i][j][ele];
            }
        }
    }
}


/* 
    Threadblock Swizzling (Global Memory L2 Optimization)
    Instead of executing blocks in standard row-major order, this logic groups 
    blocks into vertical swaths (defined by SWIZZLE_SIZE). This ensures that 
    blocks processing adjacent output tiles are scheduled concurrently.
    Because these adjacent blocks share the same rows of Matrix A and 
    columns of Matrix B, the global memory loads hit the L2 cache instead 
    of pulling from VRAM, significantly reducing global memory bandwidth pressure.

    Mainly helps with larger matrices. With larger matrices the working set exceeds L2 capacity so the L2 cache hit rate becomes more important.
*/

/**** This is the most comment dense kernel so if you want to understand most of the kernels in one read, you should read this one. ****/
template <int WAVE_M, int WAVE_N, int WAVES_I, int WAVES_J>
__global__ void WMMA_Swizzled_Full(_Float16 *A, _Float16 *B, float *C) {
    // Change 1: Global Block Swizzling Math
    // Instead of raw blockIdx coordinates, we calculate a 1D block_id and 
    // mathematically fold it into 2D tile groups of size SWIZZLE_SIZE.
    int SWIZZLE_SIZE = 32; // Number of blocks in a swizzle group (must divide gridDim.x)
    if (gridDim.x < SWIZZLE_SIZE) {
        SWIZZLE_SIZE = gridDim.x;
    }
    int block_id = blockIdx.y * gridDim.x + blockIdx.x; // Flatten 2D block index to 1D
    int blocks_per_group = SWIZZLE_SIZE * gridDim.y; // Total block in a vertical strip
    int group_id = block_id / blocks_per_group; // find the strip this block belongs to
    int group_lane = block_id % blocks_per_group; // find the block's position within the strip
    
    // Remap coordinates to process blocks in column-major strips
    int swizzled_block_x = group_id * SWIZZLE_SIZE + (group_lane % SWIZZLE_SIZE); // Find the x coordinate of the block in the strip
    int swizzled_block_y = group_lane / SWIZZLE_SIZE; // find the y coordinate of the block in the strip

    // Edge case handling for grids not perfectly divisible by SWIZZLE_SIZE
    if (swizzled_block_x >= gridDim.x) {
        swizzled_block_x = blockIdx.x;
        swizzled_block_y = blockIdx.y;
    }

    // Thread and wave identification 
    int laneId = threadIdx.x; // 0-31 within the wave
    int waveId = threadIdx.y; // identifies which wavefront (0 to WAVES_I*WAVES_J-1)
    int threadIdInBlock = (waveId * 32) + laneId; // thread id in the block
 
    int waveRow = waveId / WAVES_J; // wave row in the block
    int waveCol = waveId % WAVES_J; // wave column in the block
 
    // Matrix Dimensions and tiling
    constexpr int LDS_K = 32; // Depth of the K dimension tile in LDS  
    constexpr int WAVE_M_SIZE = WAVE_M * 16; // Number of rows of A processed by each wave
    constexpr int WAVE_N_SIZE = WAVE_N * 16; // Number of columns of B processed by each wave
    constexpr int BLOCK_M_SIZE = WAVES_I * WAVE_M_SIZE; // Total rows of A processed by the entire block
    constexpr int BLOCK_N_SIZE = WAVES_J * WAVE_N_SIZE; // Total columns of B processed by the entire block
 
    // Global block coordinates (Using Swizzled Block IDs)
    int blockRowStart = swizzled_block_y * BLOCK_M_SIZE; //Start row index of the block in global matrix C
    int blockColStart = swizzled_block_x * BLOCK_N_SIZE; // Start column index of the block in global matrix C
 
    int globalRowC = blockRowStart + (waveRow * WAVE_M_SIZE); // Find specific row indices for the current wave's output tile
    int globalColC = blockColStart + (waveCol * WAVE_N_SIZE); //Find specific column indices for the current wave's output tile.

    constexpr int STRIDE_A = LDS_K; // Stride for A in LDS            
    constexpr int STRIDE_B = BLOCK_N_SIZE; // Stride for B in LDS
 
    alignas(16) __shared__ _Float16 As[BLOCK_M_SIZE * STRIDE_A]; // Shared memory buffer for matrix A tiles (BLOCK_M_SIZE rows x STRIDE_A columns, aligned for 128-bit vectorized loads)
    alignas(16) __shared__ _Float16 Bs[LDS_K * STRIDE_B]; // Shared memory buffer for matrix B tiles (LDS_K rows x STRIDE_B columns, aligned for 128-bit vectorized loads)
 
    // Accumulator initialization
    float8 c_frag[WAVE_M][WAVE_N];
    #pragma unroll
    for (int i = 0; i < WAVE_M; ++i) {
        #pragma unroll
        for (int j = 0; j < WAVE_N; ++j) {
            c_frag[i][j] = float8{0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
        }
    }
 
    int wmmaLaneWrapped = laneId % 16; // Lane ID within a WMMA subgroup (0-15 bc WMMA intrinsics use 16 lanes per group)
    int wmmaLaneGroup = laneId / 16; // Which WMMA subgroup this lane belongs to (0 or 1 for 32 lanes total)

    // Chunked memory load calculations 
    constexpr int ELEMENTS_PER_CHUNK = 8; // Number of _Float16 elements in a 128-bit vectorized load chunk
    constexpr int TOTAL_WAVES = WAVES_I * WAVES_J; // Total wavefronts per thread block
    constexpr int TOTAL_THREADS = TOTAL_WAVES * 32; // Total threads per block (32 threads per wavefront)

    constexpr int aChunksPerRow = LDS_K / ELEMENTS_PER_CHUNK; // Number of 128-bit chunks per row in matrix A's LDS tile
    constexpr int totalAChunks = BLOCK_M_SIZE * aChunksPerRow; // Total 128-bit chunks for matrix A's entire tile

    constexpr int bGlobalChunksPerRow = BLOCK_N_SIZE / ELEMENTS_PER_CHUNK;  // Number of 128-bit chunks per row in matrix B's global tile
    constexpr int totalBChunks = LDS_K * bGlobalChunksPerRow;  // Total 128-bit chunks for matrix B's entire tile

    static_assert(totalAChunks == TOTAL_THREADS, "Thread count must equal A chunks");  // Ensures 1:1 mapping of threads to A's load chunks (no loops needed)
    static_assert(totalBChunks == TOTAL_THREADS, "Thread count must equal B chunks");  // Ensures 1:1 mapping of threads to B's load chunks (no loops needed)


    int rowA = threadIdInBlock / aChunksPerRow;  // Determine which row of matrix A this thread loads
    int colChunkA = threadIdInBlock % aChunksPerRow;  // Determine which 128-bit chunk within that row
    int swizzleA = (rowA & 3) ^ ((rowA >> 2) & 3);  // XOR swizzle pattern based on row to eliminate LDS bank conflicts
    int swizzledColChunkA = colChunkA ^ swizzleA;  // Apply swizzle to column index for writes
    int ldsRowStartA = rowA * STRIDE_A;  // LDS row offset for this thread's A chunk
    uint4* ldsPtrA = reinterpret_cast<uint4*>(&As[ldsRowStartA]);  // Pointer to 128-bit chunk in LDS (A)

    int rowB = threadIdInBlock / bGlobalChunksPerRow;  // Determine which row of matrix B this thread loads
    int colChunkB = threadIdInBlock % bGlobalChunksPerRow;  // Determine which 128-bit chunk within that row
    int swizzleB = (rowB & 15) ^ (((rowB >> 3) & 1) << 1);  // XOR swizzle pattern based on row to eliminate LDS bank conflicts
    int swizzledColChunkB = colChunkB ^ swizzleB;  // Apply swizzle to column index for writes
    int ldsRowStartB = rowB * STRIDE_B;  // LDS row offset for this thread's B chunk
    uint4* ldsPtrB = reinterpret_cast<uint4*>(&Bs[ldsRowStartB]);  // Pointer to 128-bit chunk in LDS (B)

    // Compute Loop
    for (int k = 0; k < DIM_K; k += LDS_K) {

        // Matrix A Load
        int globalRowStartA = (blockRowStart + rowA) * DIM_K + k;  // Global memory element offset to start of this thread's A chunk
        uint4* globalPtrA = reinterpret_cast<uint4*>(&A[globalRowStartA]);  // Pointer to 128-bit chunk in global memory (A)
        ldsPtrA[swizzledColChunkA] = globalPtrA[colChunkA];  // Load 128-bit chunk from global to LDS with swizzled write address
 
        // Matrix B Load
        int globalRowStartB = (k + rowB) * DIM_J + blockColStart;  // Global memory element offset to start of this thread's B chunk
        uint4* globalPtrB = reinterpret_cast<uint4*>(&B[globalRowStartB]);  // Pointer to 128-bit chunk in global memory (B)
        ldsPtrB[swizzledColChunkB] = globalPtrB[colChunkB];  // Load 128-bit chunk from global to LDS with swizzled write address
 
        __syncthreads();  // Wait for all threads to finish loading before compute

        // WMMA Math - compute 16x16 fragments across K dimension in 16-element steps
        for (int tk = 0; tk < LDS_K; tk += 16) {
            half8 b_frag[WAVE_N];  // Register buffer for B fragments for this wave
            
            for (int j = 0; j < WAVE_N; ++j) {
                for (int ele = 0; ele < WMMA_DATA_WIDTH; ++ele) {  // Load each element of the B fragment
                    int rowInLdsB = tk + (ele + wmmaLaneGroup * WMMA_DATA_WIDTH);  // Row index in LDS for B (offset by lane group within wave)
                    int colInLdsB = (waveCol * WAVE_N_SIZE) + (j * 16) + wmmaLaneWrapped;  // Column index in LDS for B (wave col + fragment j + lane position)

                    int swizzleReadB = (rowInLdsB & 15) ^ (((rowInLdsB >> 3) & 1) << 1);  // Inverse swizzle pattern for reads
                    int swizzledColInLdsB = colInLdsB ^ (swizzleReadB * 8);  // Apply inverse swizzle to undo write-time scrambling

                    b_frag[j][ele] = Bs[(rowInLdsB * STRIDE_B) + swizzledColInLdsB];  // Load B element from LDS with unswizzled address
                }
            }
 
            #pragma unroll
            for (int i = 0; i < WAVE_M; ++i) {  // For each A fragment this wave computes
                
                int rowInLdsA = (waveRow * WAVE_M_SIZE) + (i * 16) + wmmaLaneWrapped;  // Row index in LDS for A (wave row + fragment i + lane position)
                int colInLdsA = tk + (wmmaLaneGroup * WMMA_DATA_WIDTH);  // Column index in LDS for A (K offset + lane group position)

                int swizzleReadA = (rowInLdsA & 3) ^ ((rowInLdsA >> 2) & 3);  // Inverse swizzle pattern for reads
                int swizzledColInLdsA = colInLdsA ^ (swizzleReadA * 8);  // Apply inverse swizzle to undo write-time scrambling

                half8* ldsPtrA_frag = reinterpret_cast<half8*>(&As[(rowInLdsA * STRIDE_A) + swizzledColInLdsA]);  // Pointer to 128-bit A fragment in LDS
                half8 a_frag = ldsPtrA_frag[0];  // Load A fragment from LDS
                
                #pragma unroll
                for (int j = 0; j < WAVE_N; ++j) {  // Compute A[i] * B[j] using WMMA hardware
                    c_frag[i][j] = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12(a_frag, b_frag[j], c_frag[i][j]);  // Accumulate: C += A*B (fp16 input, fp32 output)
                }
            }
        } 
        __syncthreads();
    }
 
    // Store Results to Global Memory 
    #pragma unroll
    for (int i = 0; i < WAVE_M; ++i) {
        #pragma unroll
        for (int j = 0; j < WAVE_N; ++j) {
            #pragma unroll
            for (int ele = 0; ele < WMMA_DATA_WIDTH; ++ele) {    
                int targetRowC = globalRowC + (i * 16) + (ele + wmmaLaneGroup * WMMA_DATA_WIDTH);
                int targetColC = globalColC + (j * 16) + wmmaLaneWrapped; 
            
                C[targetRowC * DIM_J + targetColC] = c_frag[i][j][ele]; 
            }
        }
    }
}

float getBestTiming(hipEvent_t *start, hipEvent_t *stop, int runs) {
    float minTime = std::numeric_limits<float>::max(); //make min time very large 
    for (int i = 0; i < runs; i++){
        float elapsedTime = 0.0;
        HIP_CHECK(hipEventElapsedTime(&elapsedTime, start[i], stop[i]));
        minTime = std::min(minTime, elapsedTime); //update min time if the current run is faster
        
    }
    return minTime;
}

static bool validateResults(float *kernelC, float *rocblasC, int size) {
    // Using relative error because the accumulated absolute error between rocBLAS and our kernel can be more than 0.001 in some cases (because they accumulate results in different orders)
    // but the relative error is very small compared to how many results are calculated and accumulated in the matrix multiplication.
    // 0.001% relative error tolerance 
    float relTolerance = 0.0001; 
    
    for (int i = 0; i < size; ++i) {
        float diff = fabsf(kernelC[i] - rocblasC[i]);
        float abs_expected = fabsf(rocblasC[i]);
        
        // Prevent division by zero
        float denominator = fmaxf(abs_expected, 1e-5f); 
        float relativeError = diff / denominator;

        if (relativeError > relTolerance) {
            printf("Mismatch at %d: Kernel=%f rocBLAS=%f (Rel Error: %f)\n", i, kernelC[i], rocblasC[i], relativeError);
            return false;
        }
    }
    return true;
}

static void randomMatrix(_Float16 *m, int n) {
  for (int i = 0; i < n; ++i) {
    float val = (float)rand() / (float)RAND_MAX;
    m[i] = static_cast<_Float16>(val); // float32 to float16
  }
}

// Allocate memory
static void allocate_all(_Float16 *&hA, _Float16 *&hB, float *&hC, float *&hC_ref,
                         _Float16 *&dA, _Float16 *&dB, float *&dC) {
    HIP_CHECK(hipHostMalloc((void**)&hA, DIM_I * DIM_K * sizeof(_Float16)));
    HIP_CHECK(hipHostMalloc((void**)&hB, DIM_K * DIM_J * sizeof(_Float16)));
    HIP_CHECK(hipHostMalloc((void**)&hC, DIM_I * DIM_J * sizeof(float)));
    HIP_CHECK(hipHostMalloc((void**)&hC_ref, DIM_I * DIM_J * sizeof(float)));

    HIP_CHECK(hipMalloc((void**)&dA, DIM_I*DIM_K*sizeof(_Float16)));
    HIP_CHECK(hipMalloc((void**)&dB, DIM_K*DIM_J*sizeof(_Float16)));
    HIP_CHECK(hipMalloc((void**)&dC, DIM_I*DIM_J*sizeof(float)));
}

// Free memory
static void deallocate_all(_Float16 *hA, _Float16 *hB, float *hC, float *hC_ref,
                           _Float16 *dA, _Float16 *dB, float *dC) {
    HIP_CHECK(hipHostFree(hA));
    HIP_CHECK(hipHostFree(hB));
    HIP_CHECK(hipHostFree(hC));
    HIP_CHECK(hipHostFree(hC_ref));

    HIP_CHECK(hipFree(dA));
    HIP_CHECK(hipFree(dB));
    HIP_CHECK(hipFree(dC));
}

int main() {
    std::cout << "Launching MMM kernel\n\n";

    // Host and device pointers
    _Float16 *hA=nullptr; 
    _Float16 *hB=nullptr;
    float *hC=nullptr;
    float *hC_ref=nullptr;
    _Float16 *dA=nullptr; // Use fp16 because matrix cores expect fp16 inputs and this increases throughput. 
    _Float16 *dB=nullptr;
    float *dC=nullptr; // Final output (as well as accumulation) is in fp32

    allocate_all(hA, hB, hC, hC_ref, dA, dB, dC);
    srand(time(NULL));
    randomMatrix(hA, DIM_I*DIM_K);
    randomMatrix(hB, DIM_K*DIM_J);
    HIP_CHECK(hipMemcpy(dA, hA, DIM_I*DIM_K*sizeof(_Float16), hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(dB, hB, DIM_K*DIM_J*sizeof(_Float16), hipMemcpyHostToDevice));

    // These are the necessary parameters so that each thread loads only one chunk and not multiple. The static asserts in the kernels will block compilation if these have different values.
    constexpr int WAVE_M = 2; // We compute a 32x32 tile so with WAVE_M and WAVE_N equal to 2 each thread will load exactly one 128-bit chunk of A and one 128-bit chunk of B.
    constexpr int WAVE_N = 2;    
    constexpr int WAVES_I = 4; // With 4 waves in the I dimension and each wave computing a 32 row tile, each block computes a 128 row tile.
    constexpr int WAVES_J = 4; // With 4 waves in the J dimension and each wave computing a 32 column tile, each block computes a 128 column tile.
    constexpr int BLOCK_M = WAVES_I * WAVE_M * 16; // 1 block computes 128 rows of the output matrix
    constexpr int BLOCK_N = WAVES_J * WAVE_N * 16; // 1 block computes 128 columns of the output matrix
    constexpr int TOTAL_WAVES = WAVES_I * WAVES_J; // Total number of waves per block. We use Wave32, meaning 32 threads per wavefront. With 16 total waves (4x4), each block has 512 threads (16x32). 

    // This is for the kernels that use templates. For the other kernels use the ones described above their declaration
    dim3 threadsPerBlock(32, TOTAL_WAVES, 1); 
    dim3 blocksPerGrid(CEIL_DIV(DIM_J, BLOCK_N), CEIL_DIV(DIM_I, BLOCK_M), 1);


    //dim3 threadsPerBlock(32, 4, 1); 
    //dim3 blocksPerGrid(CEIL_DIV(DIM_J, 64), CEIL_DIV(DIM_I, 64), 1);

    // Number of times we will run.
    constexpr int nbRun = 50;
    hipEvent_t start[nbRun], stop[nbRun];
    for (int i = 0; i < nbRun; i++) {
        HIP_CHECK(hipEventCreate(start + i));
        HIP_CHECK(hipEventCreate(stop + i));
    }

    // Custom Kernel execution 
    // WARMUP RUN
    WMMA_Swizzled_Full<WAVE_M, WAVE_N, WAVES_I, WAVES_J><<<blocksPerGrid, threadsPerBlock>>>(dA, dB, dC);
    //WMMA_2D_RB_VecLoad_GMEM<<<blocksPerGrid, threadsPerBlock>>>(dA, dB, dC);

    HIP_CHECK(hipDeviceSynchronize());
    for (int i = 0; i < nbRun; i++){
        HIP_CHECK(hipEventRecord(start[i], 0));
        //WMMA_2D_RB_VecLoad_GMEM<<<blocksPerGrid, threadsPerBlock>>>(dA, dB, dC);

        WMMA_Swizzled_Full<WAVE_M, WAVE_N, WAVES_I, WAVES_J><<<blocksPerGrid, threadsPerBlock>>>(dA, dB, dC);
        HIP_CHECK(hipEventRecord(stop[i], 0));
    }
    HIP_CHECK(hipGetLastError());
    HIP_CHECK(hipDeviceSynchronize());
    // Copy custom kernel results to host
    HIP_CHECK(hipMemcpy(hC, dC, DIM_I*DIM_J*sizeof(float), hipMemcpyDeviceToHost));
    auto kernelTiming = getBestTiming(start, stop, nbRun);
    float kernelGFLOPs = (1e-6 * DIM_I * DIM_J * DIM_K * 2) / kernelTiming;

    // rocBLAS execution 
    std::cout << "Generating validation data..." << std::endl;
    rocblas_handle handle;
    ROCBLAS_CHECK(rocblas_create_handle(&handle));

    //rocBLAS parameters.
    float alpha = 1.0f;
    float beta = 0.0f;
    rocblas_int m = DIM_J;
    rocblas_int n = DIM_I;
    rocblas_int k = DIM_K;

    // rocBLAS WARMUP
    ROCBLAS_CHECK(rocblas_gemm_ex(
        handle, rocblas_operation_none, rocblas_operation_none, m, n, k, &alpha, dB, rocblas_datatype_f16_r, m, dA, rocblas_datatype_f16_r, k,
        &beta, dC, rocblas_datatype_f32_r, m, dC, rocblas_datatype_f32_r, m, rocblas_datatype_f32_r, rocblas_gemm_algo_standard, 0, 0));
    HIP_CHECK(hipDeviceSynchronize());

    //rocBLAS actual runs
    for (int i = 0; i < nbRun; i++) {
        HIP_CHECK(hipEventRecord(start[i], 0));
        ROCBLAS_CHECK(rocblas_gemm_ex(handle, rocblas_operation_none, rocblas_operation_none, m, n, k, &alpha, dB, rocblas_datatype_f16_r, m, dA, rocblas_datatype_f16_r, k,
        &beta, dC, rocblas_datatype_f32_r, m, dC, rocblas_datatype_f32_r, m, rocblas_datatype_f32_r, rocblas_gemm_algo_standard, 0, 0 ));
        HIP_CHECK(hipEventRecord(stop[i], 0));
    }

    HIP_CHECK(hipDeviceSynchronize());
    auto rocblasTiming = getBestTiming(start, stop, nbRun);
    float rocblasGFLOPs = (1e-6 * DIM_I * DIM_J * DIM_K * 2) / rocblasTiming;
    // Copy rocBLAS results to host as the reference
    HIP_CHECK(hipMemcpy(hC_ref, dC, DIM_I*DIM_J*sizeof(float), hipMemcpyDeviceToHost));
    // Validation and output 
    bool ok = validateResults(hC, hC_ref, DIM_I * DIM_J);
    printf("\nMatrix multiplication is %s.\n", ok ? "CORRECT" : "INCORRECT");
    if (ok)
        std::cout << "Custom Kernel: " << kernelTiming << " ms -> " << kernelGFLOPs << " GFLOPS" << std::endl;
    std::cout << "rocBLAS:       " << rocblasTiming << " ms -> " << rocblasGFLOPs << " GFLOPS" << std::endl;

    ROCBLAS_CHECK(rocblas_destroy_handle(handle));
    deallocate_all(hA, hB, hC, hC_ref, dA, dB, dC);
    for (int i = 0; i < nbRun; i++) {
        HIP_CHECK(hipEventDestroy(start[i]));
        HIP_CHECK(hipEventDestroy(stop[i]));
    }
    
    return ok ? 0 : 1;
}
