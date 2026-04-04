// SPDX-License-Identifier: MIT
// Fused dequantize + matrix-vector multiply for GGUF quantized weights.
// Exposes torch.ops.tinyserve_ggml.ggml_mul_mat_vec(act, weight, type, N, K).
//
// Block struct definitions derived from ggml-common.h (MIT, ggml-org/ggml).
// Dequantization logic derived from ggml dequantize.cuh / convert.cu.
// Self-contained — no ggml headers needed, avoiding build conflicts with PyTorch.

#include <torch/extension.h>
#include <c10/cuda/CUDAStream.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>

#include <cstdint>

// --------------------------------------------------------------------------
// GGML type IDs (subset we support)
// --------------------------------------------------------------------------
constexpr int GGML_TYPE_Q4_0 = 2;
constexpr int GGML_TYPE_Q8_0 = 8;
constexpr int GGML_TYPE_Q4_K = 12;
constexpr int GGML_TYPE_Q5_K = 13;
constexpr int GGML_TYPE_Q6_K = 14;

// --------------------------------------------------------------------------
// GGML block struct definitions (from ggml-common.h, MIT license)
// Using half = cuda __half for ggml_half on CUDA.
// --------------------------------------------------------------------------
#define QK_K 256
#define K_SCALE_SIZE 12
#define QK8_0 32

#define QK4_0 32

#pragma pack(push, 1)

struct block_q4_0 {
    half d;
    uint8_t qs[QK4_0 / 2];  // nibble-packed: 2 values per byte
};
static_assert(sizeof(block_q4_0) == sizeof(half) + QK4_0 / 2, "wrong q4_0 block size");

struct block_q8_0 {
    half d;
    int8_t qs[QK8_0];
};
static_assert(sizeof(block_q8_0) == sizeof(half) + QK8_0, "wrong q8_0 block size");

struct block_q4_K {
    half d;
    half dmin;
    uint8_t scales[K_SCALE_SIZE];
    uint8_t qs[QK_K / 2];
};
static_assert(sizeof(block_q4_K) == 2 * sizeof(half) + K_SCALE_SIZE + QK_K / 2,
              "wrong q4_K block size");

struct block_q5_K {
    half d;
    half dmin;
    uint8_t scales[K_SCALE_SIZE];
    uint8_t qh[QK_K / 8];
    uint8_t qs[QK_K / 2];
};
static_assert(sizeof(block_q5_K) == 2 * sizeof(half) + K_SCALE_SIZE + QK_K / 2 + QK_K / 8,
              "wrong q5_K block size");

struct block_q6_K {
    uint8_t ql[QK_K / 2];
    uint8_t qh[QK_K / 4];
    int8_t scales[QK_K / 16];
    half d;
};
static_assert(sizeof(block_q6_K) == sizeof(half) + QK_K / 16 + 3 * QK_K / 4,
              "wrong q6_K block size");

#pragma pack(pop)

// --------------------------------------------------------------------------
// Quant metadata
// --------------------------------------------------------------------------
struct QuantMeta {
    int block_elements;
    int type_size;
};

__host__ static QuantMeta get_quant_meta(int ggml_type) {
    switch (ggml_type) {
        case GGML_TYPE_Q4_0: return {32, 18};
        case GGML_TYPE_Q8_0: return {32, 34};
        case GGML_TYPE_Q4_K: return {256, 144};
        case GGML_TYPE_Q5_K: return {256, 176};
        case GGML_TYPE_Q6_K: return {256, 210};
        default: return {0, 0};
    }
}

// --------------------------------------------------------------------------
// K-quant scale decoding (from ggml)
// 12-byte packed scales for Q4_K/Q5_K: 6-bit quantized scales and mins.
// --------------------------------------------------------------------------
__device__ __forceinline__ void decode_k_scales(
    const uint8_t* sc, int ib_inner,
    float& scale_val, float& min_val, float d, float dmin
) {
    // ib_inner = sub-block index 0..7 within the super-block
    // Q4_K/Q5_K: 8 sub-blocks of 32 elements each
    // scales[0..3] contain low 4 bits of scale for sub-blocks 0-7 (two per byte)
    // scales[4..5] contain low 4 bits of min for sub-blocks 0-7
    // scales[6..7] contain bits 4-5 of scale for sub-blocks 0-7
    // scales[8..9] contain bits 4-5 of min for sub-blocks 0-7
    // But ggml actually stores them differently. Let me use the ggml convention directly.

    uint8_t sc_lo, m_lo;
    uint8_t sc_hi, m_hi;

    if (ib_inner < 4) {
        sc_lo = sc[ib_inner] & 0x3F;
        m_lo  = sc[ib_inner + 4] & 0x3F;
        sc_hi = (sc[ib_inner + 8] >> 0) & 3;
        m_hi  = (sc[ib_inner + 8] >> 2) & 3;
    } else {
        // sub-blocks 4-7
        sc_lo = (sc[ib_inner - 4] >> 6) | ((sc[ib_inner] & 0xF) << 2);
        m_lo  = (sc[ib_inner] >> 4) | ((sc[ib_inner + 4] & 0xF) << 2);
        // Wait, this is wrong. Let me re-derive from ggml source.
        // Actually the ggml K-quant scale packing is:
        // For sub-block j (0..7):
        //   if j < 4:
        //     scale = (sc[j] & 0x3F) | ((sc[j+8] & 0x03) << 4)  -- no, 6 bits
        //   Actually ggml uses get_scale_min_k4 which is:
        //   j < 4:  d_scale = sc[j] & 63,  d_min = sc[j+4] & 63
        //           d_scale |= (sc[2*(j/2)+8] >> (4*(j%2)+0)) & 3) << 4 -- no
        // This is getting complex. Let me just implement it correctly.
        sc_lo = 0; m_lo = 0; sc_hi = 0; m_hi = 0; // placeholder
    }

    int sc_full = sc_lo | (sc_hi << 6);
    int m_full  = m_lo  | (m_hi  << 6);
    scale_val = d * sc_full;
    min_val   = dmin * m_full;
}

// Actually, let me use the correct ggml K-scale decoding.
// From ggml: get_scale_min_k4(int j, const uint8_t * q, uint8_t & d, uint8_t & m)
__device__ __forceinline__ void get_scale_min_k4(
    int j, const uint8_t* q, uint8_t& d_out, uint8_t& m_out
) {
    if (j < 4) {
        d_out = q[j] & 63;
        m_out = q[j + 4] & 63;
    } else {
        d_out = (q[j + 4] & 0xF) | ((q[j - 4] >> 6) << 4);
        m_out = (q[j + 4] >>  4) | ((q[j]     >> 6) << 4);
    }
}

// --------------------------------------------------------------------------
// Fused dequant+dot kernels: one warp per output row
// Each warp loads quantized weight row, dequantizes, dots with activation.
// --------------------------------------------------------------------------

// Q8_0: 32 elements per block, 34 bytes (2 byte scale + 32 int8 quants)
// Q4_0: 32 elements per block, 18 bytes (simplest 4-bit format)
__global__ void matvec_q4_0_kernel(
    const void* __restrict__ weight,
    const float* __restrict__ act,
    float* __restrict__ out,
    int N, int K
) {
    const int row = blockIdx.x * blockDim.y + threadIdx.y;
    if (row >= N) return;

    const int n_blocks = K / QK4_0;
    const block_q4_0* w = (const block_q4_0*)weight + row * n_blocks;

    float sum = 0.0f;
    for (int b = threadIdx.x; b < n_blocks; b += blockDim.x) {
        float d = __half2float(w[b].d);
        int base = b * QK4_0;
        float local_sum = 0.0f;
        #pragma unroll
        for (int i = 0; i < QK4_0 / 2; ++i) {
            uint8_t byte = w[b].qs[i];
            float q_lo = (float)(byte & 0x0F) - 8.0f;
            float q_hi = (float)(byte >> 4) - 8.0f;
            local_sum += d * q_lo * act[base + 2 * i];
            local_sum += d * q_hi * act[base + 2 * i + 1];
        }
        sum += local_sum;
    }

    // Warp reduction
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        sum += __shfl_down_sync(0xFFFFFFFF, sum, offset);
    }

    if (threadIdx.x == 0) {
        out[row] = sum;
    }
}

__global__ void matvec_q8_0_kernel(
    const void* __restrict__ weight,
    const float* __restrict__ act,
    float* __restrict__ out,
    int N, int K
) {
    const int row = blockIdx.x * blockDim.y + threadIdx.y;
    if (row >= N) return;

    const int n_blocks = K / QK8_0;
    const block_q8_0* w = (const block_q8_0*)weight + row * n_blocks;

    float sum = 0.0f;
    for (int b = threadIdx.x; b < n_blocks; b += blockDim.x) {
        float d = __half2float(w[b].d);
        int base = b * QK8_0;
        // Each thread processes one block (32 elements)
        float local_sum = 0.0f;
        #pragma unroll
        for (int i = 0; i < QK8_0; ++i) {
            float dequant = d * (float)w[b].qs[i];
            local_sum += dequant * act[base + i];
        }
        sum += local_sum;
    }

    // Warp reduction
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        sum += __shfl_down_sync(0xFFFFFFFF, sum, offset);
    }

    if (threadIdx.x == 0) {
        out[row] = sum;
    }
}

// Q4_K: 256 elements per super-block, 144 bytes
__global__ void matvec_q4_K_kernel(
    const void* __restrict__ weight,
    const float* __restrict__ act,
    float* __restrict__ out,
    int N, int K
) {
    const int row = blockIdx.x * blockDim.y + threadIdx.y;
    if (row >= N) return;

    const int n_blocks = K / QK_K;
    const block_q4_K* w = (const block_q4_K*)weight + row * n_blocks;

    float sum = 0.0f;

    for (int b = 0; b < n_blocks; ++b) {
        float d = __half2float(w[b].d);
        float dmin = __half2float(w[b].dmin);
        const uint8_t* scales = w[b].scales;
        const uint8_t* qs = w[b].qs;

        // 8 sub-blocks of 32 elements each
        // Q4_K qs layout: 128 bytes = 4 groups of 32 bytes.
        // Group g (32 bytes) covers sub-blocks 2*g (low nibble) and 2*g+1 (high nibble).
        // Each group has 32 bytes × 1 nibble = 32 elements per sub-block.
        for (int j = threadIdx.x; j < 8; j += blockDim.x) {
            uint8_t sc, m;
            get_scale_min_k4(j, scales, sc, m);
            float scale = d * sc;
            float min = dmin * m;

            int base = b * QK_K + j * 32;
            float local_sum = 0.0f;

            int group = j / 2;
            int nibble_shift = (j % 2) * 4;
            int qs_offset = group * 32;
            #pragma unroll
            for (int i = 0; i < 32; ++i) {
                uint8_t byte = qs[qs_offset + i];
                int q = (byte >> nibble_shift) & 0xF;
                float v = scale * (float)q - min;
                local_sum += v * act[base + i];
            }
            sum += local_sum;
        }
    }

    // Warp reduction
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        sum += __shfl_down_sync(0xFFFFFFFF, sum, offset);
    }

    if (threadIdx.x == 0) {
        out[row] = sum;
    }
}

// Q5_K: 256 elements per super-block, 176 bytes
__global__ void matvec_q5_K_kernel(
    const void* __restrict__ weight,
    const float* __restrict__ act,
    float* __restrict__ out,
    int N, int K
) {
    const int row = blockIdx.x * blockDim.y + threadIdx.y;
    if (row >= N) return;

    const int n_blocks = K / QK_K;
    const block_q5_K* w = (const block_q5_K*)weight + row * n_blocks;

    float sum = 0.0f;

    for (int b = 0; b < n_blocks; ++b) {
        float d = __half2float(w[b].d);
        float dmin = __half2float(w[b].dmin);
        const uint8_t* scales = w[b].scales;
        const uint8_t* qs = w[b].qs;
        const uint8_t* qh = w[b].qh;

        // Q5_K layout:
        // qs: 128 bytes = 4 groups of 32 bytes (same as Q4_K)
        //   group g covers sub-blocks 2*g (low nibble) and 2*g+1 (high nibble)
        // qh: 32 bytes = 256 bits (1 high bit per element)
        //   sub-block j uses bit j of qh[0..31]
        for (int j = threadIdx.x; j < 8; j += blockDim.x) {
            uint8_t sc, m;
            get_scale_min_k4(j, scales, sc, m);
            float scale = d * sc;
            float min = dmin * m;

            int base = b * QK_K + j * 32;
            float local_sum = 0.0f;

            int group = j / 2;
            int nibble_shift = (j % 2) * 4;
            int qs_offset = group * 32;

            #pragma unroll
            for (int i = 0; i < 32; ++i) {
                uint8_t byte = qs[qs_offset + i];
                int low4 = (byte >> nibble_shift) & 0xF;
                int hbit = (qh[i] >> j) & 1;
                int q = low4 | (hbit << 4);
                float v = scale * (float)q - min;
                local_sum += v * act[base + i];
            }
            sum += local_sum;
        }
    }

    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        sum += __shfl_down_sync(0xFFFFFFFF, sum, offset);
    }

    if (threadIdx.x == 0) {
        out[row] = sum;
    }
}

// Q6_K: 256 elements per super-block, 210 bytes
__global__ void matvec_q6_K_kernel(
    const void* __restrict__ weight,
    const float* __restrict__ act,
    float* __restrict__ out,
    int N, int K
) {
    const int row = blockIdx.x * blockDim.y + threadIdx.y;
    if (row >= N) return;

    const int n_blocks = K / QK_K;
    const block_q6_K* w = (const block_q6_K*)weight + row * n_blocks;

    float sum = 0.0f;

    for (int b = 0; b < n_blocks; ++b) {
        float d = __half2float(w[b].d);
        const uint8_t* ql = w[b].ql;
        const uint8_t* qh = w[b].qh;
        const int8_t*  sc = w[b].scales;

        // Q6_K layout: 16 sub-blocks of 16 elements each (256 total).
        //
        // ql (128 bytes): 2 groups of 64 bytes. Each group → shift [0,4] →
        //   4 sub-groups of 32 low-nibble values. Total: 8 groups of 32.
        //   group 0: low nibbles of ql[0:32]
        //   group 1: low nibbles of ql[32:64]
        //   group 2: high nibbles of ql[0:32]
        //   group 3: high nibbles of ql[32:64]
        //   group 4: low nibbles of ql[64:96]
        //   group 5: low nibbles of ql[96:128]
        //   group 6: high nibbles of ql[64:96]
        //   group 7: high nibbles of ql[96:128]
        //
        // qh (64 bytes): 2 groups of 32 bytes. Each group → shift [0,2,4,6] →
        //   4 sub-groups of 32 two-bit values.
        //   group 0: bits[0:2] of qh[0:32]
        //   group 1: bits[2:4] of qh[0:32]
        //   group 2: bits[4:6] of qh[0:32]
        //   group 3: bits[6:8] of qh[0:32]
        //   group 4: bits[0:2] of qh[32:64]
        //   group 5: bits[2:4] of qh[32:64]
        //   group 6: bits[4:6] of qh[32:64]
        //   group 7: bits[6:8] of qh[32:64]
        //
        // q = ql | (qh << 4), shape (8, 32) → reshape (16, 16).
        // Sub-block j: q-group = j/2, half = j%2.

        for (int j = threadIdx.x; j < 16; j += blockDim.x) {
            float scale = d * sc[j];
            int base_act = b * QK_K + j * 16;
            float local_sum = 0.0f;

            int qg = j / 2;   // q-group 0..7
            int qhalf = j % 2;

            // ql mapping for q-group qg
            int ql_64_grp = qg / 4;       // 0 or 1
            int ql_local = qg % 4;
            int ql_nibble_shift = (ql_local / 2) * 4;  // 0 for groups 0,1; 4 for groups 2,3
            int ql_byte_off = ql_64_grp * 64 + (ql_local % 2) * 32;

            // qh mapping for q-group qg
            int qh_32_grp = qg / 4;       // 0 or 1
            int qh_shift = (qg % 4) * 2;
            int qh_byte_off = qh_32_grp * 32;

            #pragma unroll
            for (int i = 0; i < 16; ++i) {
                int ci = qhalf * 16 + i;
                int low4 = (ql[ql_byte_off + ci] >> ql_nibble_shift) & 0xF;
                int high2 = (qh[qh_byte_off + ci] >> qh_shift) & 0x3;
                int q_val = (low4 | (high2 << 4)) - 32;
                local_sum += scale * (float)q_val * act[base_act + i];
            }
            sum += local_sum;
        }
    }

    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        sum += __shfl_down_sync(0xFFFFFFFF, sum, offset);
    }

    if (threadIdx.x == 0) {
        out[row] = sum;
    }
}

// --------------------------------------------------------------------------
// Host dispatch
// --------------------------------------------------------------------------

#define CUDA_CHECK(call) do { \
    cudaError_t err = (call); \
    TORCH_CHECK(err == cudaSuccess, "CUDA error: ", cudaGetErrorString(err)); \
} while (0)

torch::Tensor ggml_mul_mat_vec(
    torch::Tensor activation,
    torch::Tensor weight_data,
    int64_t ggml_type,
    int64_t out_features,
    int64_t in_features
) {
    TORCH_CHECK(activation.is_cuda(), "activation must be on CUDA");
    TORCH_CHECK(weight_data.is_cuda(), "weight_data must be on CUDA");
    TORCH_CHECK(weight_data.dtype() == torch::kUInt8, "weight_data must be uint8");

    auto meta = get_quant_meta(ggml_type);
    TORCH_CHECK(meta.block_elements > 0,
                "Unsupported ggml_type: ", ggml_type,
                ". Supported: Q8_0(8), Q4_K(12), Q5_K(13), Q6_K(14)");
    TORCH_CHECK(in_features % meta.block_elements == 0,
                "in_features (", in_features, ") must be divisible by block_elements (",
                meta.block_elements, ")");

    int64_t n_blocks_per_row = in_features / meta.block_elements;
    int64_t expected_bytes = out_features * n_blocks_per_row * meta.type_size;
    TORCH_CHECK(weight_data.numel() >= expected_bytes,
                "weight_data too small: got ", weight_data.numel(),
                " bytes, expected ", expected_bytes);

    auto act_f32 = activation.to(torch::kFloat32).contiguous().view({-1});
    TORCH_CHECK(act_f32.numel() >= in_features,
                "activation too small: got ", act_f32.numel(),
                " elements, expected ", in_features);

    auto output = torch::empty({out_features},
        torch::TensorOptions().dtype(torch::kFloat32).device(activation.device()));

    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();
    int N = static_cast<int>(out_features);
    int K = static_cast<int>(in_features);

    const int warp_size = 32;

    switch (ggml_type) {
        case GGML_TYPE_Q4_0: {
            constexpr int rows_per_block = 4;
            dim3 grid((N + rows_per_block - 1) / rows_per_block);
            dim3 block(warp_size, rows_per_block);
            matvec_q4_0_kernel<<<grid, block, 0, stream>>>(
                weight_data.data_ptr<uint8_t>(),
                act_f32.data_ptr<float>(),
                output.data_ptr<float>(),
                N, K
            );
            break;
        }
        case GGML_TYPE_Q8_0: {
            // One warp per row, multiple rows per block
            constexpr int rows_per_block = 4;
            dim3 grid((N + rows_per_block - 1) / rows_per_block);
            dim3 block(warp_size, rows_per_block);
            matvec_q8_0_kernel<<<grid, block, 0, stream>>>(
                weight_data.data_ptr<uint8_t>(),
                act_f32.data_ptr<float>(),
                output.data_ptr<float>(),
                N, K
            );
            break;
        }
        case GGML_TYPE_Q4_K: {
            // 8 sub-blocks per super-block, warp handles them
            constexpr int rows_per_block = 4;
            dim3 grid((N + rows_per_block - 1) / rows_per_block);
            dim3 block(warp_size, rows_per_block);
            matvec_q4_K_kernel<<<grid, block, 0, stream>>>(
                weight_data.data_ptr<uint8_t>(),
                act_f32.data_ptr<float>(),
                output.data_ptr<float>(),
                N, K
            );
            break;
        }
        case GGML_TYPE_Q5_K: {
            constexpr int rows_per_block = 4;
            dim3 grid((N + rows_per_block - 1) / rows_per_block);
            dim3 block(warp_size, rows_per_block);
            matvec_q5_K_kernel<<<grid, block, 0, stream>>>(
                weight_data.data_ptr<uint8_t>(),
                act_f32.data_ptr<float>(),
                output.data_ptr<float>(),
                N, K
            );
            break;
        }
        case GGML_TYPE_Q6_K: {
            constexpr int rows_per_block = 4;
            dim3 grid((N + rows_per_block - 1) / rows_per_block);
            dim3 block(warp_size, rows_per_block);
            matvec_q6_K_kernel<<<grid, block, 0, stream>>>(
                weight_data.data_ptr<uint8_t>(),
                act_f32.data_ptr<float>(),
                output.data_ptr<float>(),
                N, K
            );
            break;
        }
        default:
            TORCH_CHECK(false, "Unsupported ggml_type: ", ggml_type);
    }
    CUDA_CHECK(cudaGetLastError());

    return output.view({1, out_features}).to(activation.dtype());
}

TORCH_LIBRARY(tinyserve_ggml, m) {
    m.def("ggml_mul_mat_vec(Tensor activation, Tensor weight_data, int ggml_type, int out_features, int in_features) -> Tensor");
    m.impl("ggml_mul_mat_vec", &ggml_mul_mat_vec);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("ggml_mul_mat_vec", &ggml_mul_mat_vec,
          "Fused dequant + matrix-vector multiply for GGUF quantized weights");
}
