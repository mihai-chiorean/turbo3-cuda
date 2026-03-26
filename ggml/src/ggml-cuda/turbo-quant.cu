/*
 * turbo-quant.cu — TurboQuant turbo3 CUDA kernels for NVIDIA GPUs
 *
 * Part of the turbo3 CUDA port for llama.cpp.
 * Implements dequantize (contiguous + non-contiguous), FWHT rotation kernel,
 * and SET_ROWS quantize kernel (normalize → WHT rotate → 3-bit pack) for
 * GGML_TYPE_TURBO3_0 KV cache compression.
 *
 * Block format: 14 bytes per 32 values = 3.5 bits/value = 4.6× vs fp16.
 * WHT rotation is at graph level via GGML_OP_TURBO_WHT (128-element groups).
 *
 * Based on TheTom's Metal implementation (llama-cpp-turboquant)
 * and the TurboQuant paper (arXiv:2504.19874, ICLR 2026).
 *
 * Author: Erol Germain (@erolgermain)
 * Date:   March 2026
 * License: MIT (matching upstream llama.cpp)
 */

#include "common.cuh"
#include "ggml-common.h"
#include <cuda_fp16.h>

static __constant__ float TURBO3_CENTROIDS_C[8] = {
    -0.190685f, -0.117832f, -0.065717f, -0.021460f,
     0.021460f,  0.065717f,  0.117832f,  0.190685f
};

template<typename dst_t>
static __global__ void dequantize_block_turbo3_0_kernel(
    const void * __restrict__ vx,
    dst_t      * __restrict__ y,
    const int64_t k
) {
    const int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= k) return;

    const int64_t blk_idx = i / QK_TURBO3;
    const int     elem    = (int)(i % QK_TURBO3);
    const block_turbo3_0 * x = (const block_turbo3_0 *)vx + blk_idx;

    const float norm = __half2float(x->norm);
    uint8_t low2 = (x->qs[elem >> 2] >> ((elem & 3) << 1)) & 0x3;
    uint8_t hi1  = (x->signs[elem >> 3] >> (elem & 7)) & 0x1;
    uint8_t idx  = low2 | (hi1 << 2);

    y[i] = (dst_t)(TURBO3_CENTROIDS_C[idx] * norm);
}

void dequantize_row_turbo3_0_fp16_cuda(
    const void * vx, half * y, int64_t k, cudaStream_t stream
) {
    const int threads = 256;
    const int blocks  = (k + threads - 1) / threads;
    dequantize_block_turbo3_0_kernel<half><<<blocks, threads, 0, stream>>>(vx, y, k);
}

void dequantize_row_turbo3_0_fp32_cuda(
    const void * vx, float * y, int64_t k, cudaStream_t stream
) {
    const int threads = 256;
    const int blocks  = (k + threads - 1) / threads;
    dequantize_block_turbo3_0_kernel<float><<<blocks, threads, 0, stream>>>(vx, y, k);
}

void dequantize_row_turbo3_0_bf16_cuda(
    const void * vx, nv_bfloat16 * y, int64_t k, cudaStream_t stream
) {
    const int threads = 256;
    const int blocks  = (k + threads - 1) / threads;
    dequantize_block_turbo3_0_kernel<nv_bfloat16><<<blocks, threads, 0, stream>>>(vx, y, k);
}

// Non-contiguous dequant (for nc dispatch tables)
template<typename dst_t>
static __global__ void dequantize_block_turbo3_0_nc_kernel(
    const void * __restrict__ vx,
    dst_t      * __restrict__ y,
    const int64_t ne00, const int64_t ne01, const int64_t ne02, const int64_t ne03,
    const int64_t s01,  const int64_t s02,  const int64_t s03
) {
    const int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= ne00 * ne01 * ne02 * ne03) return;

    const int64_t i03 = i / (ne00 * ne01 * ne02);
    const int64_t i02 = (i / (ne00 * ne01)) % ne02;
    const int64_t i01 = (i / ne00) % ne01;
    const int64_t i00 = i % ne00;

    const int64_t src_offset = i01*s01 + i02*s02 + i03*s03;
    const char * src = (const char *)vx + src_offset;

    const int64_t blk_idx = i00 / QK_TURBO3;
    const int     elem    = (int)(i00 % QK_TURBO3);
    const block_turbo3_0 * x = (const block_turbo3_0 *)src + blk_idx;

    const float norm = __half2float(x->norm);
    uint8_t low2 = (x->qs[elem >> 2] >> ((elem & 3) << 1)) & 0x3;
    uint8_t hi1  = (x->signs[elem >> 3] >> (elem & 7)) & 0x1;
    uint8_t idx  = low2 | (hi1 << 2);

    y[i] = (dst_t)(TURBO3_CENTROIDS_C[idx] * norm);
}

void dequantize_row_turbo3_0_fp16_nc_cuda(
    const void * vx, half * y,
    int64_t ne00, int64_t ne01, int64_t ne02, int64_t ne03,
    int64_t s01, int64_t s02, int64_t s03, cudaStream_t stream
) {
    const int64_t total = ne00 * ne01 * ne02 * ne03;
    const int threads = 256;
    const int blocks  = (total + threads - 1) / threads;
    dequantize_block_turbo3_0_nc_kernel<half><<<blocks, threads, 0, stream>>>(
        vx, y, ne00, ne01, ne02, ne03, s01, s02, s03);
}

void dequantize_row_turbo3_0_fp32_nc_cuda(
    const void * vx, float * y,
    int64_t ne00, int64_t ne01, int64_t ne02, int64_t ne03,
    int64_t s01, int64_t s02, int64_t s03, cudaStream_t stream
) {
    const int64_t total = ne00 * ne01 * ne02 * ne03;
    const int threads = 256;
    const int blocks  = (total + threads - 1) / threads;
    dequantize_block_turbo3_0_nc_kernel<float><<<blocks, threads, 0, stream>>>(
        vx, y, ne00, ne01, ne02, ne03, s01, s02, s03);
}

void dequantize_row_turbo3_0_bf16_nc_cuda(
    const void * vx, nv_bfloat16 * y,
    int64_t ne00, int64_t ne01, int64_t ne02, int64_t ne03,
    int64_t s01, int64_t s02, int64_t s03, cudaStream_t stream
) {
    const int64_t total = ne00 * ne01 * ne02 * ne03;
    const int threads = 256;
    const int blocks  = (total + threads - 1) / threads;
    dequantize_block_turbo3_0_nc_kernel<nv_bfloat16><<<blocks, threads, 0, stream>>>(
        vx, y, ne00, ne01, ne02, ne03, s01, s02, s03);
}

// ═══════════════════════════════════════════════════════════════════════
//  GGML_OP_TURBO_WHT — Fast Walsh-Hadamard Transform with sign rotation
//
//  Ported from Metal kernel_turbo_wht (ggml-metal.metal line 3018)
//  and CPU ggml_compute_forward_turbo_wht_f32 (ops.cpp line 10594).
//
//  Each CUDA block (128 threads) processes one 128-element group.
//  direction=0: forward (signs1 -> FWHT -> signs2)
//  direction=1: inverse (signs2 -> FWHT -> signs1)
// ═══════════════════════════════════════════════════════════════════════

static __constant__ float d_turbo_wht_signs1[128] = {
    -1, 1, 1,-1,-1, 1,-1, 1,-1,-1, 1, 1, 1, 1, 1, 1,
     1,-1, 1,-1, 1,-1,-1, 1, 1, 1,-1, 1, 1,-1,-1,-1,
    -1, 1, 1,-1, 1, 1,-1, 1,-1, 1, 1,-1,-1, 1,-1, 1,
     1, 1, 1,-1,-1,-1,-1,-1, 1,-1, 1, 1, 1, 1,-1, 1,
    -1,-1, 1,-1,-1,-1, 1,-1,-1,-1, 1,-1,-1,-1, 1, 1,
     1,-1,-1, 1, 1, 1,-1,-1, 1, 1,-1, 1, 1,-1, 1,-1,
    -1, 1, 1,-1, 1,-1, 1,-1, 1, 1, 1, 1,-1, 1,-1, 1,
     1,-1, 1, 1,-1,-1,-1,-1,-1, 1, 1,-1, 1, 1,-1, 1
};

static __constant__ float d_turbo_wht_signs2[128] = {
     1, 1, 1, 1,-1, 1, 1,-1, 1,-1,-1,-1, 1,-1,-1,-1,
     1, 1,-1,-1, 1,-1, 1,-1, 1,-1,-1, 1,-1, 1, 1, 1,
     1, 1,-1,-1,-1, 1,-1,-1,-1,-1,-1,-1, 1, 1, 1,-1,
     1,-1, 1, 1, 1,-1,-1, 1,-1,-1,-1,-1,-1,-1, 1, 1,
     1,-1, 1,-1,-1,-1,-1, 1,-1, 1,-1, 1,-1,-1, 1, 1,
    -1, 1,-1, 1, 1,-1, 1,-1,-1,-1,-1, 1,-1,-1, 1,-1,
     1,-1, 1, 1, 1,-1,-1, 1,-1, 1,-1, 1, 1,-1,-1, 1,
    -1, 1,-1, 1, 1,-1, 1,-1, 1,-1,-1,-1,-1,-1, 1,-1
};

static __global__ void kernel_turbo_wht(
    const float * __restrict__ src,
    float       * __restrict__ dst,
    const int64_t n_elements,
    const int     direction
) {
    const int64_t group_idx = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    const int64_t n_groups = n_elements / 128;
    if (group_idx >= n_groups) return;

    const float * in  = src + group_idx * 128;
    float       * out = dst + group_idx * 128;

    float x[128];

    const float * s_first  = (direction == 0) ? d_turbo_wht_signs1 : d_turbo_wht_signs2;
    const float * s_second = (direction == 0) ? d_turbo_wht_signs2 : d_turbo_wht_signs1;

    for (int i = 0; i < 128; i++) {
        x[i] = in[i] * s_first[i];
    }

    for (int h = 1; h < 128; h *= 2) {
        for (int i = 0; i < 128; i += h * 2) {
            for (int j = i; j < i + h; j++) {
                float a = x[j], b = x[j + h];
                x[j]     = a + b;
                x[j + h] = a - b;
            }
        }
    }

    const float inv_sqrt_128 = 0.08838834764831845f;
    for (int i = 0; i < 128; i++) {
        out[i] = x[i] * inv_sqrt_128 * s_second[i];
    }
}

// ═══════════════════════════════════════════════════════════════════════
//  SET_ROWS for turbo3 — custom kernel that groups 128 elements,
//  normalizes, rotates via WHT, then quantizes into 4 turbo3 blocks.
//
//  Each thread processes one 128-element group.
// ═══════════════════════════════════════════════════════════════════════

static __constant__ float TURBO3_MIDPOINTS_QC[7] = {
    -0.154259f, -0.091775f, -0.043589f, 0.0f,
     0.043589f,  0.091775f,  0.154259f
};

static __global__ void kernel_set_rows_turbo3(
    const float * __restrict__ src0,
    const int64_t * __restrict__ src1,
    block_turbo3_0 * __restrict__ dst,
    const int64_t ne00,
    const int64_t ne01,
    const int64_t nb01,
    const int64_t nb1,
    const int n_groups_per_row
) {
    const int64_t row = blockIdx.x;
    if (row >= ne01) return;

    const int grp_idx = blockIdx.y * blockDim.x + threadIdx.x;
    if (grp_idx >= n_groups_per_row) return;

    const float * src_row = (const float *)((const char *)src0 + row * nb01);
    const int64_t dst_row_idx = src1[row];
    block_turbo3_0 * dst_row = (block_turbo3_0 *)((char *)dst + dst_row_idx * nb1);

    const float * grp_src = src_row + grp_idx * 128;

    // Step 1: Compute 128-element group norm
    float norm_sq = 0.0f;
    for (int j = 0; j < 128; j++) norm_sq += grp_src[j] * grp_src[j];
    float grp_norm = sqrtf(norm_sq);
    float inv_norm = (grp_norm > 1e-10f) ? (1.0f / grp_norm) : 0.0f;

    // Step 2: Normalize and rotate
    float x[128];
    for (int i = 0; i < 128; i++) x[i] = grp_src[i] * inv_norm;

    // Apply signs1
    for (int i = 0; i < 128; i++) x[i] *= d_turbo_wht_signs1[i];

    // FWHT butterfly (7 stages for 128 elements)
    for (int h = 1; h < 128; h *= 2) {
        for (int i = 0; i < 128; i += h * 2) {
            for (int j = i; j < i + h; j++) {
                float a = x[j], b = x[j + h];
                x[j]     = a + b;
                x[j + h] = a - b;
            }
        }
    }

    // Normalize and apply signs2
    const float inv_sqrt_128 = 0.08838834764831845f;
    for (int i = 0; i < 128; i++) x[i] = x[i] * inv_sqrt_128 * d_turbo_wht_signs2[i];

    // Step 3: Quantize into 4 blocks of 32
    for (int b = 0; b < 4; b++) {
        block_turbo3_0 * blk = &dst_row[grp_idx * 4 + b];
        const int off = b * 32;

        blk->norm = __float2half(grp_norm);

        // Clear packed bytes
        for (int j = 0; j < 8; j++) blk->qs[j] = 0;
        for (int j = 0; j < 4; j++) blk->signs[j] = 0;

        for (int j = 0; j < 32; j++) {
            float rv = x[off + j];

            // Nearest centroid via midpoints
            uint8_t idx = 0;
            idx += (rv >= TURBO3_MIDPOINTS_QC[0]);
            idx += (rv >= TURBO3_MIDPOINTS_QC[1]);
            idx += (rv >= TURBO3_MIDPOINTS_QC[2]);
            idx += (rv >= TURBO3_MIDPOINTS_QC[3]);
            idx += (rv >= TURBO3_MIDPOINTS_QC[4]);
            idx += (rv >= TURBO3_MIDPOINTS_QC[5]);
            idx += (rv >= TURBO3_MIDPOINTS_QC[6]);

            // Pack lower 2 bits
            blk->qs[j >> 2] |= ((idx & 0x3) << ((j & 3) << 1));

            // Pack upper 1 bit
            if (idx & 0x4) {
                blk->signs[j >> 3] |= (1u << (j & 7));
            }
        }
    }
}

void ggml_cuda_op_set_rows_turbo3(
    ggml_backend_cuda_context & ctx,
    ggml_tensor * dst
) {
    const ggml_tensor * src0 = dst->src[0];
    const ggml_tensor * src1 = dst->src[1];

    const float * src0_d = (const float *)src0->data;
    const int64_t * src1_d = (const int64_t *)src1->data;
    block_turbo3_0 * dst_d = (block_turbo3_0 *)dst->data;

    const int64_t ne00 = src0->ne[0];
    const int64_t ne01 = src0->ne[1];
    const int64_t nb01 = src0->nb[1];
    const int64_t nb1  = dst->nb[1];

    GGML_ASSERT(ne00 % 128 == 0);
    const int n_groups_per_row = ne00 / 128;

    const int threads = 32;
    const int grp_blocks = (n_groups_per_row + threads - 1) / threads;

    dim3 grid(ne01, grp_blocks);
    dim3 block(threads);

    kernel_set_rows_turbo3<<<grid, block, 0, ctx.stream()>>>(
        src0_d, src1_d, dst_d, ne00, ne01, nb01, nb1, n_groups_per_row);
}

void ggml_cuda_op_turbo_wht(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * src0 = dst->src[0];

    const float * src_data = (const float *)src0->data;
    float * dst_data = (float *)dst->data;

    int32_t direction;
    memcpy(&direction, dst->op_params, sizeof(int32_t));

    const int64_t n_elements = ggml_nelements(src0);
    GGML_ASSERT(n_elements % 128 == 0);

    const int64_t n_groups = n_elements / 128;
    const int threads = 256;
    const int blocks = (n_groups + threads - 1) / threads;

    kernel_turbo_wht<<<blocks, threads, 0, ctx.stream()>>>(
        src_data, dst_data, n_elements, direction);
}
