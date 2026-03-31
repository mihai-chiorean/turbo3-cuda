/*
 * tbq-quant.cu -- TBQ4_0 CUDA kernels
 */

#include "common.cuh"
#include "ggml-common.h"
#include <cuda_fp16.h>
#include <mutex>

static __device__ float d_tbq_rotation[128 * 128];
#include "tbq-rotation-128.h"

static std::once_flag tbq_rotation_once;
void tbq_ensure_rotation_loaded(cudaStream_t stream) {
    std::call_once(tbq_rotation_once, [stream]() {
        CUDA_CHECK(cudaMemcpyToSymbolAsync(d_tbq_rotation, TBQ_ROTATION_128x128,
            128 * 128 * sizeof(float), 0, cudaMemcpyHostToDevice, stream));
        CUDA_CHECK(cudaStreamSynchronize(stream));
    });
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

// Dequant kernels -- block-level with inverse rotation
// Each block of 128 threads handles one TBQ4 block (128 elements).
// 1. Codebook lookup + scale by 1/sqrt(128) * norm  (rotated-domain values)
// 2. Inverse rotation: x_j = sum_i R[i*128+j] * y_i  (R^T * y)
template<typename dst_t>
static __global__ void dequantize_block_tbq4_0_kernel(
    const void * __restrict__ vx, dst_t * __restrict__ y, const int64_t k) {

    const int64_t n_blocks = k / QK_TBQ4;
    const int64_t blk_id = blockIdx.x;
    if (blk_id >= n_blocks) return;

    const block_tbq4_0 * x = (const block_tbq4_0 *)vx + blk_id;
    const int lane = threadIdx.x;  // 0..127
    if (lane >= QK_TBQ4) return;

    const float norm = __half2float(x->d);
    const float scale = 0.08838834764831845f;  // 1/sqrt(128)

    // Step 1: Codebook lookup in rotated domain
    __shared__ float rotated[128];
    {
        uint8_t idx = (lane % 2 == 0) ? (x->qs[lane/2] & 0x0F) : ((x->qs[lane/2] >> 4) & 0x0F);
        rotated[lane] = TBQ4_CENTROIDS[idx] * scale;
    }
    __syncthreads();

    // Step 2: Inverse rotation: x_j = sum_i R[i][j] * rotated[i] = sum_i R[i*128+j] * rotated[i]
    float val = 0.0f;
    for (int i = 0; i < 128; i++) {
        val += d_tbq_rotation[i * 128 + lane] * rotated[i];
    }
    val *= norm;

    y[blk_id * QK_TBQ4 + lane] = (dst_t)val;
}

void dequantize_row_tbq4_0_fp16_cuda(const void * vx, half * y, int64_t k, cudaStream_t stream) {
    tbq_ensure_rotation_loaded(stream);
    const int n_blocks = k / QK_TBQ4;
    dequantize_block_tbq4_0_kernel<half><<<n_blocks, 128, 0, stream>>>(vx, y, k);
}
void dequantize_row_tbq4_0_fp32_cuda(const void * vx, float * y, int64_t k, cudaStream_t stream) {
    tbq_ensure_rotation_loaded(stream);
    const int n_blocks = k / QK_TBQ4;
    dequantize_block_tbq4_0_kernel<float><<<n_blocks, 128, 0, stream>>>(vx, y, k);
}
void dequantize_row_tbq4_0_bf16_cuda(const void * vx, nv_bfloat16 * y, int64_t k, cudaStream_t stream) {
    tbq_ensure_rotation_loaded(stream);
    const int n_blocks = k / QK_TBQ4;
    dequantize_block_tbq4_0_kernel<nv_bfloat16><<<n_blocks, 128, 0, stream>>>(vx, y, k);
}

// NC dequant -- block-level with inverse rotation
// Each CUDA block = one TBQ4 block (128 threads). Grid iterates over rows/batches.
template<typename dst_t>
static __global__ void dequantize_block_tbq4_0_nc_kernel(
    const void * __restrict__ vx, dst_t * __restrict__ y,
    const int64_t ne00, const int64_t ne01, const int64_t ne02, const int64_t ne03,
    const int64_t s01, const int64_t s02, const int64_t s03) {

    // blockIdx.x = TBQ4 block index across the flattened tensor
    // threadIdx.x = element within the TBQ4 block (0..127)
    const int64_t total_blocks = (ne00 * ne01 * ne02 * ne03) / QK_TBQ4;
    const int64_t blk_flat = blockIdx.x;
    if (blk_flat >= total_blocks) return;

    const int lane = threadIdx.x;
    if (lane >= QK_TBQ4) return;

    // Map flat block index back to output position
    const int64_t blocks_per_row = ne00 / QK_TBQ4;
    const int64_t out_elem = blk_flat * QK_TBQ4 + lane;
    const int64_t i00 = out_elem % ne00;
    const int64_t i01 = (out_elem / ne00) % ne01;
    const int64_t i02 = (out_elem / (ne00*ne01)) % ne02;
    const int64_t i03 = out_elem / (ne00*ne01*ne02);

    // Source block uses strided addressing
    const int64_t src_base = i01*s01 + i02*s02 + i03*s03;
    const int64_t src_blk_idx = src_base / QK_TBQ4 + (i00 / QK_TBQ4);
    // All 128 threads in this CUDA block access the SAME TBQ4 block
    const int64_t blk_start_i00 = (blk_flat % blocks_per_row) * QK_TBQ4;
    const int64_t src_idx_base = blk_start_i00 + i01*s01 + i02*s02 + i03*s03;
    const block_tbq4_0 * x = (const block_tbq4_0 *)vx + src_idx_base / QK_TBQ4;

    const float norm = __half2float(x->d);
    const float scale = 0.08838834764831845f;

    __shared__ float rotated[128];
    {
        uint8_t idx = (lane%2==0) ? (x->qs[lane/2]&0x0F) : ((x->qs[lane/2]>>4)&0x0F);
        rotated[lane] = TBQ4_CENTROIDS[idx] * scale;
    }
    __syncthreads();

    float val = 0.0f;
    for (int i = 0; i < 128; i++) {
        val += d_tbq_rotation[i * 128 + lane] * rotated[i];
    }
    val *= norm;

    y[out_elem] = (dst_t)val;
}

void dequantize_row_tbq4_0_fp16_nc_cuda(const void * vx, half * y,
    int64_t ne00, int64_t ne01, int64_t ne02, int64_t ne03,
    int64_t s01, int64_t s02, int64_t s03, cudaStream_t stream) {
    tbq_ensure_rotation_loaded(stream);
    const int64_t n_blocks = (ne00*ne01*ne02*ne03) / QK_TBQ4;
    dequantize_block_tbq4_0_nc_kernel<half><<<(int)n_blocks, 128, 0, stream>>>(vx,y,ne00,ne01,ne02,ne03,s01,s02,s03);
}
void dequantize_row_tbq4_0_fp32_nc_cuda(const void * vx, float * y,
    int64_t ne00, int64_t ne01, int64_t ne02, int64_t ne03,
    int64_t s01, int64_t s02, int64_t s03, cudaStream_t stream) {
    tbq_ensure_rotation_loaded(stream);
    const int64_t n_blocks = (ne00*ne01*ne02*ne03) / QK_TBQ4;
    dequantize_block_tbq4_0_nc_kernel<float><<<(int)n_blocks, 128, 0, stream>>>(vx,y,ne00,ne01,ne02,ne03,s01,s02,s03);
}
void dequantize_row_tbq4_0_bf16_nc_cuda(const void * vx, nv_bfloat16 * y,
    int64_t ne00, int64_t ne01, int64_t ne02, int64_t ne03,
    int64_t s01, int64_t s02, int64_t s03, cudaStream_t stream) {
    tbq_ensure_rotation_loaded(stream);
    const int64_t n_blocks = (ne00*ne01*ne02*ne03) / QK_TBQ4;
    dequantize_block_tbq4_0_nc_kernel<nv_bfloat16><<<(int)n_blocks, 128, 0, stream>>>(vx,y,ne00,ne01,ne02,ne03,s01,s02,s03);
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

// ============================================================================
// TBQ3_0: 3-bit TurboBlockQuant CUDA kernels
// 8 Lloyd-Max centroids for N(0,1), 128-element blocks, 50 bytes/block = 3.125 bpw
// ============================================================================

static __constant__ float TBQ3_CENTROIDS[8] = {
    -1.5104f, -0.9816f, -0.6568f, -0.3177f,
     0.3177f,  0.6568f,  0.9816f,  1.5104f,
};

static __constant__ float TBQ3_MIDPOINTS[7] = {
    -1.2460f, -0.8192f, -0.4872f, 0.0000f,
     0.4872f,  0.8192f,  1.2460f,
};

static __device__ __forceinline__ uint8_t tbq3_quantize_gpu(float val) {
    uint8_t idx = 0;
    idx += (val >= TBQ3_MIDPOINTS[0]); idx += (val >= TBQ3_MIDPOINTS[1]);
    idx += (val >= TBQ3_MIDPOINTS[2]); idx += (val >= TBQ3_MIDPOINTS[3]);
    idx += (val >= TBQ3_MIDPOINTS[4]); idx += (val >= TBQ3_MIDPOINTS[5]);
    idx += (val >= TBQ3_MIDPOINTS[6]);
    return idx;
}

// Helper: extract 3-bit index from packed array
static __device__ __forceinline__ uint8_t tbq3_unpack(const uint8_t * qs, int j) {
    int bit_offset = j * 3;
    int byte_idx = bit_offset / 8;
    int bit_pos = bit_offset % 8;
    uint16_t raw = (uint16_t)qs[byte_idx];
    if (byte_idx + 1 < 48) raw |= (uint16_t)qs[byte_idx + 1] << 8;
    return (uint8_t)((raw >> bit_pos) & 0x7);
}

// Standalone dequant -- block-level with inverse rotation (same as TBQ4_0 approach)
template<typename dst_t>
static __global__ void dequantize_block_tbq3_0_kernel(
    const void * __restrict__ vx, dst_t * __restrict__ y, const int64_t k) {

    const int64_t n_blocks = k / QK_TBQ3;
    const int64_t blk_id = blockIdx.x;
    if (blk_id >= n_blocks) return;

    const block_tbq3_0 * x = (const block_tbq3_0 *)vx + blk_id;
    const int lane = threadIdx.x;
    if (lane >= QK_TBQ3) return;

    const float norm = __half2float(x->d);
    const float scale = 0.08838834764831845f;  // 1/sqrt(128)

    __shared__ float rotated[128];
    {
        uint8_t idx = tbq3_unpack(x->qs, lane);
        rotated[lane] = TBQ3_CENTROIDS[idx] * scale;
    }
    __syncthreads();

    // Inverse rotation: x_j = sum_i R[i*128+j] * rotated[i]
    float val = 0.0f;
    for (int i = 0; i < 128; i++) {
        val += d_tbq_rotation[i * 128 + lane] * rotated[i];
    }
    val *= norm;

    y[blk_id * QK_TBQ3 + lane] = (dst_t)val;
}

void dequantize_row_tbq3_0_fp16_cuda(const void * vx, half * y, int64_t k, cudaStream_t stream) {
    tbq_ensure_rotation_loaded(stream);
    const int n_blocks = k / QK_TBQ3;
    dequantize_block_tbq3_0_kernel<half><<<n_blocks, 128, 0, stream>>>(vx, y, k);
}
void dequantize_row_tbq3_0_fp32_cuda(const void * vx, float * y, int64_t k, cudaStream_t stream) {
    tbq_ensure_rotation_loaded(stream);
    const int n_blocks = k / QK_TBQ3;
    dequantize_block_tbq3_0_kernel<float><<<n_blocks, 128, 0, stream>>>(vx, y, k);
}
void dequantize_row_tbq3_0_bf16_cuda(const void * vx, nv_bfloat16 * y, int64_t k, cudaStream_t stream) {
    tbq_ensure_rotation_loaded(stream);
    const int n_blocks = k / QK_TBQ3;
    dequantize_block_tbq3_0_kernel<nv_bfloat16><<<n_blocks, 128, 0, stream>>>(vx, y, k);
}

// NC dequant for TBQ3_0
template<typename dst_t>
static __global__ void dequantize_block_tbq3_0_nc_kernel(
    const void * __restrict__ vx, dst_t * __restrict__ y,
    const int64_t ne00, const int64_t ne01, const int64_t ne02, const int64_t ne03,
    const int64_t s01, const int64_t s02, const int64_t s03) {

    const int64_t total_blocks = (ne00 * ne01 * ne02 * ne03) / QK_TBQ3;
    const int64_t blk_flat = blockIdx.x;
    if (blk_flat >= total_blocks) return;

    const int lane = threadIdx.x;
    if (lane >= QK_TBQ3) return;

    const int64_t blocks_per_row = ne00 / QK_TBQ3;
    const int64_t out_elem = blk_flat * QK_TBQ3 + lane;
    const int64_t i01 = (out_elem / ne00) % ne01;
    const int64_t i02 = (out_elem / (ne00*ne01)) % ne02;
    const int64_t i03 = out_elem / (ne00*ne01*ne02);

    const int64_t blk_start_i00 = (blk_flat % blocks_per_row) * QK_TBQ3;
    const int64_t src_idx_base = blk_start_i00 + i01*s01 + i02*s02 + i03*s03;
    const block_tbq3_0 * x = (const block_tbq3_0 *)vx + src_idx_base / QK_TBQ3;

    const float norm = __half2float(x->d);
    const float scale = 0.08838834764831845f;

    __shared__ float rotated[128];
    {
        uint8_t idx = tbq3_unpack(x->qs, lane);
        rotated[lane] = TBQ3_CENTROIDS[idx] * scale;
    }
    __syncthreads();

    float val = 0.0f;
    for (int i = 0; i < 128; i++) {
        val += d_tbq_rotation[i * 128 + lane] * rotated[i];
    }
    val *= norm;

    y[out_elem] = (dst_t)val;
}

void dequantize_row_tbq3_0_fp16_nc_cuda(const void * vx, half * y,
    int64_t ne00, int64_t ne01, int64_t ne02, int64_t ne03,
    int64_t s01, int64_t s02, int64_t s03, cudaStream_t stream) {
    tbq_ensure_rotation_loaded(stream);
    const int64_t n_blocks = (ne00*ne01*ne02*ne03) / QK_TBQ3;
    dequantize_block_tbq3_0_nc_kernel<half><<<(int)n_blocks, 128, 0, stream>>>(vx,y,ne00,ne01,ne02,ne03,s01,s02,s03);
}
void dequantize_row_tbq3_0_fp32_nc_cuda(const void * vx, float * y,
    int64_t ne00, int64_t ne01, int64_t ne02, int64_t ne03,
    int64_t s01, int64_t s02, int64_t s03, cudaStream_t stream) {
    tbq_ensure_rotation_loaded(stream);
    const int64_t n_blocks = (ne00*ne01*ne02*ne03) / QK_TBQ3;
    dequantize_block_tbq3_0_nc_kernel<float><<<(int)n_blocks, 128, 0, stream>>>(vx,y,ne00,ne01,ne02,ne03,s01,s02,s03);
}
void dequantize_row_tbq3_0_bf16_nc_cuda(const void * vx, nv_bfloat16 * y,
    int64_t ne00, int64_t ne01, int64_t ne02, int64_t ne03,
    int64_t s01, int64_t s02, int64_t s03, cudaStream_t stream) {
    tbq_ensure_rotation_loaded(stream);
    const int64_t n_blocks = (ne00*ne01*ne02*ne03) / QK_TBQ3;
    dequantize_block_tbq3_0_nc_kernel<nv_bfloat16><<<(int)n_blocks, 128, 0, stream>>>(vx,y,ne00,ne01,ne02,ne03,s01,s02,s03);
}

// SET_ROWS kernel for TBQ3_0
// Use shared memory for indices, then single-lane serial packing for correctness.
__launch_bounds__(32, 4)
static __global__ void kernel_set_rows_tbq3(
    const float * __restrict__ src0,
    const int64_t * __restrict__ src1,
    block_tbq3_0 * __restrict__ dst,
    const int64_t ne00, const int64_t ne01,
    const int64_t nb01, const int64_t nb1,
    const int n_blocks_per_row) {
    const int64_t row = blockIdx.x;
    if (row >= ne01) return;
    const int blk_idx = blockIdx.y;
    if (blk_idx >= n_blocks_per_row) return;

    const float * src_row = (const float *)((const char *)src0 + row * nb01);
    const int64_t dst_row_idx = src1[row];
    block_tbq3_0 * dst_blk = (block_tbq3_0 *)((char *)dst + dst_row_idx * nb1) + blk_idx;

    const int lane = threadIdx.x;
    const float * grp_src = src_row + blk_idx * QK_TBQ3;

    __shared__ float s_unit[128];
    __shared__ uint8_t s_indices[128];
    __shared__ uint8_t s_packed[48];

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

    // Phase 1: Rotate and quantize all 128 elements into shared memory indices
    for (int pass = 0; pass < 4; pass++) {
        int elem = lane + pass * 32;
        if (elem >= 128) break;

        float sum = 0.0f;
        for (int j = 0; j < 128; j++) {
            sum += d_tbq_rotation[elem * 128 + j] * s_unit[j];
        }
        s_indices[elem] = tbq3_quantize_gpu(sum * scale_up);
    }
    __syncwarp();

    // Phase 2: Pack 3-bit indices into qs[] bytes using lane 0 (simple + correct)
    if (lane < 16) {
        // Each of 16 lanes packs 3 bytes (= 8 indices worth of 3-bit data)
        // Lane 0: bytes 0..2 (indices 0..7), Lane 1: bytes 3..5 (indices 8..15), etc.
        // Total: 16 lanes * 3 bytes = 48 bytes, 16 * 8 = 128 indices
        int base_idx = lane * 8;
        int base_byte = lane * 3;

        // 8 indices * 3 bits = 24 bits = 3 bytes exactly
        uint32_t bits = 0;
        for (int k = 0; k < 8; k++) {
            bits |= ((uint32_t)s_indices[base_idx + k]) << (k * 3);
        }
        s_packed[base_byte + 0] = (uint8_t)(bits & 0xFF);
        s_packed[base_byte + 1] = (uint8_t)((bits >> 8) & 0xFF);
        s_packed[base_byte + 2] = (uint8_t)((bits >> 16) & 0xFF);
    }
    __syncwarp();

    // Copy packed bytes to output
    for (int b = lane; b < 48; b += 32) {
        dst_blk->qs[b] = s_packed[b];
    }
    __syncwarp();

    if (lane == 0) {
        dst_blk->d = __float2half(block_norm);
    }
}

void ggml_cuda_op_set_rows_tbq3(
    ggml_backend_cuda_context & ctx,
    ggml_tensor * dst) {
    const ggml_tensor * src0 = dst->src[0];
    const ggml_tensor * src1 = dst->src[1];

    const float * src0_d = (const float *)src0->data;
    const int64_t * src1_d = (const int64_t *)src1->data;
    block_tbq3_0 * dst_d = (block_tbq3_0 *)dst->data;

    const int64_t ne00 = src0->ne[0];
    const int64_t ne01 = src0->ne[1];
    const int64_t nb01 = src0->nb[1];
    const int64_t nb1  = dst->nb[1];

    GGML_ASSERT(ne00 % QK_TBQ3 == 0);
    const int n_blocks_per_row = ne00 / QK_TBQ3;

    tbq_ensure_rotation_loaded(ctx.stream());

    dim3 grid(ne01, n_blocks_per_row);
    dim3 block(32);

    kernel_set_rows_tbq3<<<grid, block, 0, ctx.stream()>>>(
        src0_d, src1_d, dst_d, ne00, ne01, nb01, nb1, n_blocks_per_row);
}
