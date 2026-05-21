#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <cmath>
#include <limits>
#include <algorithm>
#include <hip/hip_ext.h>
#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include <rocblas/rocblas.h>

#define ROCBLAS_CHECK(call) do { \
  rocblas_status _s = (call); \
  if (_s != rocblas_status_success) { \
    fprintf(stderr, "rocBLAS error %s:%d: %d\n", __FILE__, __LINE__, _s); \
    exit(1); \
  } \
} while(0)

const int mt = 128;
const int bt = 8192;
const int rt = 64;
const int nt = 64;
const int rt_l = 128;

const int DIM_I = rt * mt;      // M dimension (8192)
const int DIM_J = bt;           // N dimension (8192)
const int DIM_K = nt * rt_l;    // K dimension (8192)

#define WMMA_DATA_WIDTH 8
#define CEIL_DIV(a,b) (((a)+(b)-1)/(b))

typedef _Float16 half8 __attribute__((ext_vector_type(8)));
typedef float float8 __attribute__((ext_vector_type(8)));

#define HIP_CHECK(call) do { \
  hipError_t _e = (call); \
  if (_e != hipSuccess) { \
    fprintf(stderr, "HIP error %s:%d: %s\n", __FILE__, __LINE__, hipGetErrorString(_e)); \
    exit(1); \
  } \
} while(0)

template <int WAVE_M, int WAVE_N, int WAVES_I, int WAVES_J>
__global__ void WMMA_Swizzled_Full(const _Float16 *A, const _Float16 *Input, float *Output)  {

    // Added explicit edge handling for partial tiles.
    // it makes the kernel usable when DIM_I, DIM_J, or DIM_K stop being exact tile multiples.

    int SWIZZLE_SIZE = 32;
    if (gridDim.x < SWIZZLE_SIZE) {
        SWIZZLE_SIZE = gridDim.x;
    }
    const int block_id = blockIdx.y * gridDim.x + blockIdx.x;
    const int blocks_per_group = SWIZZLE_SIZE * gridDim.y;
    const int group_id = block_id / blocks_per_group;
    const int group_lane = block_id % blocks_per_group;
    
    int swizzled_block_x = group_id * SWIZZLE_SIZE + (group_lane % SWIZZLE_SIZE);
    int swizzled_block_y = group_lane / SWIZZLE_SIZE;

    if (swizzled_block_x >= gridDim.x) {
        swizzled_block_x = blockIdx.x;
        swizzled_block_y = blockIdx.y;
    }

    const int laneId = threadIdx.x; 
    const int waveId = threadIdx.y; 
    const int threadIdInBlock = (waveId * 32) + laneId; 
 
    const int waveRow = waveId / WAVES_J;   
    const int waveCol = waveId % WAVES_J;   
 
    constexpr int LDS_K = 32;      
    constexpr int WAVE_M_SIZE = WAVE_M * 16;        
    constexpr int WAVE_N_SIZE = WAVE_N * 16;        
    constexpr int BLOCK_M_SIZE = WAVES_I * WAVE_M_SIZE; 
    constexpr int BLOCK_N_SIZE = WAVES_J * WAVE_N_SIZE; 
 
    const int blockRowStart = swizzled_block_y * BLOCK_M_SIZE;
    const int blockColStart = swizzled_block_x * BLOCK_N_SIZE;
 
    const int globalRowC = blockRowStart + (waveRow * WAVE_M_SIZE);  
    const int globalColC = blockColStart + (waveCol * WAVE_N_SIZE);  

    constexpr int STRIDE_A = LDS_K;             
    // B is staged as [local K][local output column]. The previous version relies
    // on this same layout, but the project copy below zero-fills out-of-range
    // chunks before writing them into the swizzled LDS tile.
    constexpr int STRIDE_B = BLOCK_N_SIZE; 
 
    alignas(16) __shared__ _Float16 As[BLOCK_M_SIZE * STRIDE_A];
    alignas(16) __shared__ _Float16 Bs[LDS_K * STRIDE_B]; 
 
    float8 c_frag[WAVE_M][WAVE_N];
    #pragma unroll
    for (int i = 0; i < WAVE_M; ++i) {
        #pragma unroll
        for (int j = 0; j < WAVE_N; ++j) {
            c_frag[i][j] = float8{0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
        }
    }
 
    const int wmmaLaneWrapped = laneId % 16;   
    const int wmmaLaneGroup = laneId / 16;   
 
    constexpr int ELEMENTS_PER_CHUNK = 8; 
    constexpr int TOTAL_WAVES = WAVES_I * WAVES_J; 
    constexpr int TOTAL_THREADS = TOTAL_WAVES * 32; 

    constexpr int aChunksPerRow = LDS_K / ELEMENTS_PER_CHUNK; 
    constexpr int totalAChunks = BLOCK_M_SIZE * aChunksPerRow;

    static_assert(totalAChunks == TOTAL_THREADS, "Thread count must equal A chunks");
 
    for (int k = 0; k < DIM_K; k += LDS_K) {
        

        // A load difference from the previous version:
        // the previous version formed a base pointer and always performed a uint4
        // load. Here we split row/column first so the fast vector path can be
        // guarded, with a scalar tail path for incomplete 8-half chunks.
        const int rowA = threadIdInBlock / aChunksPerRow;
        const int colChunkA = threadIdInBlock % aChunksPerRow; 

        // LDS swizzle is unchanged from the attached version. It scrambles the
        // 8-half chunk column to reduce LDS bank conflicts during WMMA reads.
        const int swizzleA = (rowA & 3) ^ ((rowA >> 2) & 3);
        const int swizzledColChunkA = colChunkA ^ swizzleA;

        const int globalRowA = blockRowStart + rowA;
        const int globalColA = k + (colChunkA * 8);
        const int ldsRowStartA = rowA * STRIDE_A;

        uint4 a_vec = {0, 0, 0, 0}; 
        
        if (globalRowA < DIM_I && globalColA + 7 < DIM_K) {
            const uint4* globalPtrA = reinterpret_cast<const uint4*>(&A[globalRowA * DIM_K + globalColA]);
            a_vec = globalPtrA[0];
        } else if (globalRowA < DIM_I) {
            // Unlike the previous version, partial A chunks are preserved instead
            // of reading past DIM_K. Elements outside the matrix stay zero.
            _Float16* a_vals = reinterpret_cast<_Float16*>(&a_vec);
            for (int e = 0; e < 8; ++e) {
                if (globalColA + e < DIM_K) {
                    a_vals[e] = A[globalRowA * DIM_K + globalColA + e];
                }
            }
        }
        
        uint4* ldsPtrA = reinterpret_cast<uint4*>(&As[ldsRowStartA]);
        ldsPtrA[swizzledColChunkA] = a_vec;
 

        // B/Input load difference from the previous version:
        // each thread owns one 8-half K chunk for one output column. The new versoin
        // bounds-checks the vector load and falls back to scalar fill for
        // edge chunks before applying the same LDS swizzle.
        const int b_local = threadIdInBlock / 4; 
        const int k_chunk = threadIdInBlock % 4;  

        const int global_b = blockColStart + b_local;
        const int global_k_start = k + k_chunk * 8;

        uint4 b_vec = {0, 0, 0, 0};
        
        if (global_b < DIM_J && global_k_start + 7 < DIM_K) {   
            const uint4* globalPtrB = reinterpret_cast<const uint4*>(&Input[global_b * DIM_K + global_k_start]);
            b_vec = globalPtrB[0];
        } else if (global_b < DIM_J) {
            // Scalar tail path added. It avoids out of bounds reads when DIM_K is not divisible by 8.
            _Float16* b_vals_reg = reinterpret_cast<_Float16*>(&b_vec);
            for(int e = 0; e < 8; ++e) {
                if (global_k_start + e < DIM_K) {
                    b_vals_reg[e] = Input[global_b * DIM_K + global_k_start + e];
                }
            }
        }
        
        _Float16* b_vals = reinterpret_cast<_Float16*>(&b_vec);

        #pragma unroll
        for (int ele = 0; ele < 8; ++ele) {
            int k_local = k_chunk * 8 + ele;
            const int swizzleB = (k_local & 15) ^ (((k_local >> 3) & 1) << 1);
            const int swizzled_b_local = b_local ^ (swizzleB * 8);
            Bs[k_local * STRIDE_B + swizzled_b_local] = b_vals[ele];
        }
 
        __syncthreads();
 
        for (int tk = 0; tk < LDS_K; tk += 16) {
            half8 b_frag[WAVE_N];
            
            for (int j = 0; j < WAVE_N; ++j) {
                for (int ele = 0; ele < WMMA_DATA_WIDTH; ++ele) {
                    const int rowInLdsB = tk + (ele + wmmaLaneGroup * WMMA_DATA_WIDTH);
                    const int colInLdsB = (waveCol * WAVE_N_SIZE) + (j * 16) + wmmaLaneWrapped;

                    const int swizzleReadB = (rowInLdsB & 15) ^ (((rowInLdsB >> 3) & 1) << 1);
                    const int swizzledColInLdsB = colInLdsB ^ (swizzleReadB * 8);

                    b_frag[j][ele] = Bs[(rowInLdsB * STRIDE_B) + swizzledColInLdsB]; 
                }
            }
 
            #pragma unroll
            for (int i = 0; i < WAVE_M; ++i) {
                const int rowInLdsA = (waveRow * WAVE_M_SIZE) + (i * 16) + wmmaLaneWrapped;
                const int colInLdsA = tk + (wmmaLaneGroup * WMMA_DATA_WIDTH);

                const int swizzleReadA = (rowInLdsA & 3) ^ ((rowInLdsA >> 2) & 3);
                const int swizzledColInLdsA = colInLdsA ^ (swizzleReadA * 8);

                const half8* ldsPtrA_frag = reinterpret_cast<const half8*>(&As[(rowInLdsA * STRIDE_A) + swizzledColInLdsA]);
                const half8 a_frag = ldsPtrA_frag[0];
                
                #pragma unroll
                for (int j = 0; j < WAVE_N; ++j) {
                    c_frag[i][j] = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12(a_frag, b_frag[j], c_frag[i][j]);
                }
            }
        } 
        __syncthreads();
    }
 

    #pragma unroll
    for (int i = 0; i < WAVE_M; ++i) {
        #pragma unroll
        for (int j = 0; j < WAVE_N; ++j) {
            
            // Store difference from the previous version:
            // the previous versoin vector-stored the whole float8 fragment after
            // computing m/r from r_C_base. This version stores each lane element
            // separately so boundary checks can protect partial output tiles.
            const int r_C_base = globalRowC + (i * 16) + (wmmaLaneGroup * WMMA_DATA_WIDTH);
            const int c_C = globalColC + (j * 16) + wmmaLaneWrapped;
            
            float* frag_vals = reinterpret_cast<float*>(&c_frag[i][j]);

            #pragma unroll
            for (int ele = 0; ele < WMMA_DATA_WIDTH; ++ele) {
                int r_C = r_C_base + ele;
                
                if (r_C < DIM_I && c_C < DIM_J) {
                    const int m = r_C / rt;
                    const int r = r_C % rt;
                    const int b = c_C;
                    
                    const int out_idx_base = (m * bt * rt) + (b * rt) + r;
                    Output[out_idx_base] = frag_vals[ele];
                }
            }
        }
    }
}

float getBestTiming(hipEvent_t *start, hipEvent_t *stop, int runs) {
    float minTime = std::numeric_limits<float>::max();
    for (int i = 0; i < runs; i++){
        float elapsedTime = 0.0;
        HIP_CHECK(hipEventElapsedTime(&elapsedTime, start[i], stop[i]));
        if (elapsedTime > 0.0f) {
            minTime = std::min(minTime, elapsedTime);
        }
    }
    return minTime;
}

static void randomMatrix(_Float16 *m, int n) {
  for (int i = 0; i < n; ++i) {
    float val = (float)rand() / (float)RAND_MAX;
    m[i] = static_cast<_Float16>(val); 
  }
}

static bool validateResults(float *kernelC, float *rocblasC, int mt, int bt, int rt) {
    float relTolerance = 0.00001; 
    
    for (int m = 0; m < mt; ++m) {
        for (int b = 0; b < bt; ++b) {
            for (int r = 0; r < rt; ++r) {

                int kernel_idx = m * (bt * rt) + b * rt + r;
                int rocblas_idx = (m * rt + r) * bt + b;
                
                float k_val = kernelC[kernel_idx];
                float r_val = rocblasC[rocblas_idx];

                float diff = fabsf(k_val - r_val);
                float abs_expected = fabsf(r_val);
                
                float denominator = fmaxf(abs_expected, 1e-5f); 
                float relativeError = diff / denominator;

                if (relativeError > relTolerance) {
                    printf("Mismatch at [m=%d, b=%d, r=%d]: Kernel=%f rocBLAS=%f (Rel Error: %f)\n", 
                           m, b, r, k_val, r_val, relativeError);
                    return false;
                }
            }
        }
    }
    return true;
}

int main() {
    std::cout << "Launching 5D Tensor Contraction via MMM kernel\n\n";

    _Float16* G = (_Float16*)malloc(rt * nt * mt * rt_l * sizeof(_Float16));
    _Float16* Input_tensor = (_Float16*)malloc(bt * nt * rt_l * sizeof(_Float16));
    float* Output_tensor = (float*)malloc(mt * bt * rt * sizeof(float));
    float* hC_ref = (float*)malloc(mt * bt * rt * sizeof(float));

    randomMatrix(G, rt * nt * mt * rt_l);
    randomMatrix(Input_tensor, bt * nt * rt_l);

    _Float16 *hA = (_Float16*)malloc(DIM_I * DIM_K * sizeof(_Float16));
    _Float16 *dA=nullptr, *d_Input=nullptr;
    float *d_Output=nullptr;

    HIP_CHECK(hipMalloc((void**)&dA, DIM_I * DIM_K * sizeof(_Float16)));
    HIP_CHECK(hipMalloc((void**)&d_Input, bt * nt * rt_l * sizeof(_Float16)));
    HIP_CHECK(hipMalloc((void**)&d_Output, mt * bt * rt * sizeof(float)));

    std::cout << "Packing static G tensor..." << std::endl;

    for (int r = 0; r < rt; r++) {
        for (int m = 0; m < mt; m++) {
            for (int n = 0; n < nt; n++) {
                for (int k = 0; k < rt_l; k++) {
                    int row_A = m * rt + r; 
                    int col_A = n * rt_l + k;
                    int g_idx = (r * nt * mt * rt_l) + (n * mt * rt_l) + (m * rt_l) + k;
                    hA[row_A * DIM_K + col_A] = G[g_idx]; 
                }
            }
        }
    }

    HIP_CHECK(hipMemcpy(dA, hA, DIM_I * DIM_K * sizeof(_Float16), hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(d_Input, Input_tensor, bt * nt * rt_l * sizeof(_Float16), hipMemcpyHostToDevice));

    constexpr int WAVE_M = 2; 
    constexpr int WAVE_N = 2;    
    constexpr int WAVES_I = 4;   
    constexpr int WAVES_J = 4;   

    constexpr int BLOCK_M = WAVES_I * WAVE_M * 16;
    constexpr int BLOCK_N = WAVES_J * WAVE_N * 16;
    constexpr int TOTAL_WAVES = WAVES_I * WAVES_J;

    dim3 threadsPerBlock(32, TOTAL_WAVES, 1); 
    dim3 blocksPerGrid(CEIL_DIV(DIM_J, BLOCK_N), CEIL_DIV(DIM_I, BLOCK_M), 1);

    constexpr int nbRun = 100;
    hipEvent_t start[nbRun], stop[nbRun];
    for (int i = 0; i < nbRun; i++) {
        HIP_CHECK(hipEventCreate(start + i));
        HIP_CHECK(hipEventCreate(stop + i));
    }

    WMMA_Swizzled_Full<WAVE_M, WAVE_N, WAVES_I, WAVES_J><<<blocksPerGrid, threadsPerBlock>>>(dA, d_Input, d_Output);
    HIP_CHECK(hipDeviceSynchronize());
    
    for (int i = 0; i < nbRun; i++){
        HIP_CHECK(hipEventRecord(start[i], 0));
        WMMA_Swizzled_Full<WAVE_M, WAVE_N, WAVES_I, WAVES_J><<<blocksPerGrid, threadsPerBlock>>>(dA, d_Input, d_Output);
        HIP_CHECK(hipEventRecord(stop[i], 0));
    }

    HIP_CHECK(hipGetLastError());
    HIP_CHECK(hipDeviceSynchronize());

    HIP_CHECK(hipMemcpy(Output_tensor, d_Output, mt * bt * rt * sizeof(float), hipMemcpyDeviceToHost));
    
    auto kernelTiming = getBestTiming(start, stop, nbRun);
    float kernelGFLOPs = (1e-6 * DIM_I * DIM_J * DIM_K * 2) / kernelTiming;

    std::cout << "Generating validation data with rocBLAS..." << std::endl;
    
    rocblas_handle handle;
    ROCBLAS_CHECK(rocblas_create_handle(&handle));

    float alpha = 1.0f;
    float beta = 0.0f;

    rocblas_int m_roc = DIM_J;
    rocblas_int n_roc = DIM_I;
    rocblas_int k_roc = DIM_K;

    ROCBLAS_CHECK(rocblas_gemm_ex(
        handle, 
        rocblas_operation_transpose, 
        rocblas_operation_none,
        m_roc, n_roc, k_roc, &alpha, 
        d_Input, rocblas_datatype_f16_r, k_roc, 
        dA, rocblas_datatype_f16_r, k_roc,      
        &beta, 
        d_Output, rocblas_datatype_f32_r, m_roc, 
        d_Output, rocblas_datatype_f32_r, m_roc, rocblas_datatype_f32_r,
        rocblas_gemm_algo_standard, 0, 0
    ));
    HIP_CHECK(hipDeviceSynchronize());

    for (int i = 0; i < nbRun; i++) {
        HIP_CHECK(hipEventRecord(start[i], 0));
        ROCBLAS_CHECK(rocblas_gemm_ex(
            handle, 
            rocblas_operation_transpose, 
            rocblas_operation_none,
            m_roc, n_roc, k_roc, &alpha, 
            d_Input, rocblas_datatype_f16_r, k_roc, 
            dA, rocblas_datatype_f16_r, k_roc,      
            &beta, 
            d_Output, rocblas_datatype_f32_r, m_roc, 
            d_Output, rocblas_datatype_f32_r, m_roc, rocblas_datatype_f32_r,
            rocblas_gemm_algo_standard, 0, 0
        ));
        HIP_CHECK(hipEventRecord(stop[i], 0));
    }
    
    HIP_CHECK(hipDeviceSynchronize());
    auto rocblasTiming = getBestTiming(start, stop, nbRun);
    float rocblasGFLOPs = (1e-6 * DIM_I * DIM_J * DIM_K * 2) / rocblasTiming;

    HIP_CHECK(hipMemcpy(hC_ref, d_Output, DIM_I * DIM_J * sizeof(float), hipMemcpyDeviceToHost));

    bool ok = validateResults(Output_tensor, hC_ref, mt, bt, rt);
    printf("\nMatrix multiplication is %s.\n", ok ? "CORRECT" : "INCORRECT");

    if (ok) {
        std::cout << "Custom Kernel: " << kernelTiming << " ms -> " << kernelGFLOPs << " GFLOPS" << std::endl;
    }
    std::cout << "rocBLAS:       " << rocblasTiming << " ms -> " << rocblasGFLOPs << " GFLOPS" << std::endl;

    ROCBLAS_CHECK(rocblas_destroy_handle(handle));

    ::free(hA);
    ::free(G);
    ::free(Input_tensor);
    ::free(Output_tensor);
    ::free(hC_ref);
    HIP_CHECK(hipFree(dA));
    HIP_CHECK(hipFree(d_Input));
    HIP_CHECK(hipFree(d_Output));
    
    for (int i = 0; i < nbRun; i++) {
        HIP_CHECK(hipEventDestroy(start[i]));
        HIP_CHECK(hipEventDestroy(stop[i]));
    }
    
    return ok ? 0 : 1;
}
