/*
 * tbq-quant.cu -- TBQ4_0 CUDA kernels
 */

#include "common.cuh"
#include "ggml-common.h"
#include <cuda_fp16.h>

static __device__ float d_tbq_rotation[128 * 128];
#include "tbq-rotation-128.h"

static bool tbq_rotation_loaded = false;
void tbq_ensure_rotation_loaded(cudaStream_t stream) {
    if (!tbq_rotation_loaded) {
        CUDA_CHECK(cudaMemcpyToSymbolAsync(d_tbq_rotation, TBQ_ROTATION_128x128,
            128 * 128 * sizeof(float), 0, cudaMemcpyHostToDevice, stream));
        CUDA_CHECK(cudaStreamSynchronize(stream));
        tbq_rotation_loaded = true;
    }
}

static __constant__ float TBQ4_CENTROIDS[16] = {
    -2.7326f, -2.0690f, -1.6180f, -1.2562f,
    -0.9424f, -0.6568f, -0.3881f, -0.1284f,
     0.1284f,  0.3881f,  0.6568f,  0.9424f,
     1.2562f,  1.6180f,  2.0690f,  2.7326f,
};

static __constant__ float TBQ4_MIDPOINTS[15] = {
    -2.4008f, -1.8435f, -1.4371f, -1.0993f,
    -0.7996f, -0.5225f, -0.2583f,  0.0000f,
     0.2583f,  0.5225f,  0.7996f,  1.0993f,
     1.4371f,  1.8435f,  2.4008f,
};

static __device__ __forceinline__ uint8_t tbq4_quantize_gpu(float val) {
    uint8_t idx = 0;
    idx += (val >= TBQ4_MIDPOINTS[ 0]); idx += (val >= TBQ4_MIDPOINTS[ 1]);
    idx += (val >= TBQ4_MIDPOINTS[ 2]); idx += (val >= TBQ4_MIDPOINTS[ 3]);
    idx += (val >= TBQ4_MIDPOINTS[ 4]); idx += (val >= TBQ4_MIDPOINTS[ 5]);
    idx += (val >= TBQ4_MIDPOINTS[ 6]); idx += (val >= TBQ4_MIDPOINTS[ 7]);
    idx += (val >= TBQ4_MIDPOINTS[ 8]); idx += (val >= TBQ4_MIDPOINTS[ 9]);
    idx += (val >= TBQ4_MIDPOINTS[10]); idx += (val >= TBQ4_MIDPOINTS[11]);
    idx += (val >= TBQ4_MIDPOINTS[12]); idx += (val >= TBQ4_MIDPOINTS[13]);
    idx += (val >= TBQ4_MIDPOINTS[14]);
    return idx;
}

// Dequant kernels
template<typename dst_t>
static __global__ void dequantize_block_tbq4_0_kernel(
    const void * __restrict__ vx, dst_t * __restrict__ y, const int64_t k) {
    const int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= k) return;
    const block_tbq4_0 * x = (const block_tbq4_0 *)vx + i / QK_TBQ4;
    const int elem = (int)(i % QK_TBQ4);
    const float norm = __half2float(x->d);
    uint8_t idx = (elem % 2 == 0) ? (x->qs[elem/2] & 0x0F) : ((x->qs[elem/2] >> 4) & 0x0F);
    y[i] = (dst_t)(TBQ4_CENTROIDS[idx] * 0.0625f * norm);
}

void dequantize_row_tbq4_0_fp16_cuda(const void * vx, half * y, int64_t k, cudaStream_t stream) {
    const int t=256, b=(k+t-1)/t;
    dequantize_block_tbq4_0_kernel<half><<<b,t,0,stream>>>(vx,y,k);
}
void dequantize_row_tbq4_0_fp32_cuda(const void * vx, float * y, int64_t k, cudaStream_t stream) {
    const int t=256, b=(k+t-1)/t;
    dequantize_block_tbq4_0_kernel<float><<<b,t,0,stream>>>(vx,y,k);
}
void dequantize_row_tbq4_0_bf16_cuda(const void * vx, nv_bfloat16 * y, int64_t k, cudaStream_t stream) {
    const int t=256, b=(k+t-1)/t;
    dequantize_block_tbq4_0_kernel<nv_bfloat16><<<b,t,0,stream>>>(vx,y,k);
}

// NC dequant
template<typename dst_t>
static __global__ void dequantize_block_tbq4_0_nc_kernel(
    const void * __restrict__ vx, dst_t * __restrict__ y,
    const int64_t ne00, const int64_t ne01, const int64_t ne02, const int64_t ne03,
    const int64_t s01, const int64_t s02, const int64_t s03) {
    const int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    const int64_t total = ne00*ne01*ne02*ne03;
    if (i >= total) return;
    const int64_t i00 = i % ne00;
    const int64_t i01 = (i / ne00) % ne01;
    const int64_t i02 = (i / (ne00*ne01)) % ne02;
    const int64_t i03 = i / (ne00*ne01*ne02);
    const int64_t src_idx = i00 + i01*s01 + i02*s02 + i03*s03;
    const block_tbq4_0 * x = (const block_tbq4_0 *)vx + src_idx / QK_TBQ4;
    const int elem = (int)(src_idx % QK_TBQ4);
    const float norm = __half2float(x->d);
    uint8_t idx = (elem%2==0) ? (x->qs[elem/2]&0x0F) : ((x->qs[elem/2]>>4)&0x0F);
    y[i] = (dst_t)(TBQ4_CENTROIDS[idx] * 0.0625f * norm);
}

void dequantize_row_tbq4_0_fp16_nc_cuda(const void * vx, half * y,
    int64_t ne00, int64_t ne01, int64_t ne02, int64_t ne03,
    int64_t s01, int64_t s02, int64_t s03, cudaStream_t stream) {
    const int64_t n=ne00*ne01*ne02*ne03; const int t=256,b=(n+t-1)/t;
    dequantize_block_tbq4_0_nc_kernel<half><<<b,t,0,stream>>>(vx,y,ne00,ne01,ne02,ne03,s01,s02,s03);
}
void dequantize_row_tbq4_0_fp32_nc_cuda(const void * vx, float * y,
    int64_t ne00, int64_t ne01, int64_t ne02, int64_t ne03,
    int64_t s01, int64_t s02, int64_t s03, cudaStream_t stream) {
    const int64_t n=ne00*ne01*ne02*ne03; const int t=256,b=(n+t-1)/t;
    dequantize_block_tbq4_0_nc_kernel<float><<<b,t,0,stream>>>(vx,y,ne00,ne01,ne02,ne03,s01,s02,s03);
}
void dequantize_row_tbq4_0_bf16_nc_cuda(const void * vx, nv_bfloat16 * y,
    int64_t ne00, int64_t ne01, int64_t ne02, int64_t ne03,
    int64_t s01, int64_t s02, int64_t s03, cudaStream_t stream) {
    const int64_t n=ne00*ne01*ne02*ne03; const int t=256,b=(n+t-1)/t;
    dequantize_block_tbq4_0_nc_kernel<nv_bfloat16><<<b,t,0,stream>>>(vx,y,ne00,ne01,ne02,ne03,s01,s02,s03);
}

// SET_ROWS kernel
__launch_bounds__(32, 4)
static __global__ void kernel_set_rows_tbq4(
    const float * __restrict__ src0,
    const int64_t * __restrict__ src1,
    block_tbq4_0 * __restrict__ dst,
    const int64_t ne00, const int64_t ne01,
    const int64_t nb01, const int64_t nb1,
    const int n_blocks_per_row) {
    const int64_t row = blockIdx.x;
    if (row >= ne01) return;
    const int blk_idx = blockIdx.y;
    if (blk_idx >= n_blocks_per_row) return;

    const float * src_row = (const float *)((const char *)src0 + row * nb01);
    const int64_t dst_row_idx = src1[row];
    block_tbq4_0 * dst_blk = (block_tbq4_0 *)((char *)dst + dst_row_idx * nb1) + blk_idx;

    const int lane = threadIdx.x;
    const float * grp_src = src_row + blk_idx * QK_TBQ4;

    __shared__ float s_unit[128];

    // Load + compute norm
    float local_norm_sq = 0.0f;
    for (int i = lane; i < 128; i += 32) {
        float v = grp_src[i];
        s_unit[i] = v;
        local_norm_sq += v * v;
    }
    for (int offset = 16; offset > 0; offset >>= 1)
        local_norm_sq += __shfl_xor_sync(0xFFFFFFFF, local_norm_sq, offset);

    float block_norm = sqrtf(local_norm_sq);
    if (block_norm < 1e-10f) block_norm = 1e-10f;
    float inv_norm = 1.0f / block_norm;

    for (int i = lane; i < 128; i += 32)
        s_unit[i] *= inv_norm;
    __syncwarp();

    const float scale_up = 11.3137085f; // sqrt(128)

    // Each lane writes 2 bytes (covering all 64 bytes = 128 elements)
    for (int b = 0; b < 2; b++) {
        int byte_idx = lane + b * 32;
        int elem0 = byte_idx * 2;
        int elem1 = elem0 + 1;

        float sum0 = 0.0f, sum1 = 0.0f;
        for (int j = 0; j < 128; j++) {
            float u = s_unit[j];
            sum0 += d_tbq_rotation[elem0 * 128 + j] * u;
            sum1 += d_tbq_rotation[elem1 * 128 + j] * u;
        }
        uint8_t idx0 = tbq4_quantize_gpu(sum0 * scale_up);
        uint8_t idx1 = tbq4_quantize_gpu(sum1 * scale_up);

        dst_blk->qs[byte_idx] = idx0 | (idx1 << 4);
    }

    if (lane == 0) {
        dst_blk->d = __float2half(block_norm);
    }
}

void ggml_cuda_op_set_rows_tbq4(
    ggml_backend_cuda_context & ctx,
    ggml_tensor * dst) {
    const ggml_tensor * src0 = dst->src[0];
    const ggml_tensor * src1 = dst->src[1];

    const float * src0_d = (const float *)src0->data;
    const int64_t * src1_d = (const int64_t *)src1->data;
    block_tbq4_0 * dst_d = (block_tbq4_0 *)dst->data;

    const int64_t ne00 = src0->ne[0];
    const int64_t ne01 = src0->ne[1];
    const int64_t nb01 = src0->nb[1];
    const int64_t nb1  = dst->nb[1];

    GGML_ASSERT(ne00 % QK_TBQ4 == 0);
    const int n_blocks_per_row = ne00 / QK_TBQ4;

    tbq_ensure_rotation_loaded(ctx.stream());

    dim3 grid(ne01, n_blocks_per_row);
    dim3 block(32);

    kernel_set_rows_tbq4<<<grid, block, 0, ctx.stream()>>>(
        src0_d, src1_d, dst_d, ne00, ne01, nb01, nb1, n_blocks_per_row);
}
