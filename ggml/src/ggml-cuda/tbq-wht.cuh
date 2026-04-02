#pragma once
/*
 * tbq-wht.cuh -- 128-point Fast Walsh-Hadamard Transform helpers for TBQ
 *
 * Replaces the dense 128x128 rotation matrix (64KB) with FWHT (512 bytes).
 * Sign flip arrays match TheTom/Madreag upstream turbo-quant for compatibility.
 *
 * Forward:  y = signs2 * FWHT(signs1 * x) / sqrt(128)
 * Inverse:  x = signs1 * FWHT(signs2 * y) / sqrt(128)
 * (FWHT is self-adjoint; applying forward twice = identity up to scaling)
 */

#ifdef __CUDACC__

// Sign flip arrays -- must match turbo-quant.cu d_turbo_wht_signs1/2
static __constant__ float d_tbq_wht_signs1[128] = {
    -1, 1, 1,-1,-1, 1,-1, 1,-1,-1, 1, 1, 1, 1, 1, 1,
     1,-1, 1,-1, 1,-1,-1, 1, 1, 1,-1, 1, 1,-1,-1,-1,
    -1, 1, 1,-1, 1, 1,-1, 1,-1, 1, 1,-1,-1, 1,-1, 1,
     1, 1, 1,-1,-1,-1,-1,-1, 1,-1, 1, 1, 1, 1,-1, 1,
    -1,-1, 1,-1,-1,-1, 1,-1,-1,-1, 1,-1,-1,-1, 1, 1,
     1,-1,-1, 1, 1, 1,-1,-1, 1, 1,-1, 1, 1,-1, 1,-1,
    -1, 1, 1,-1, 1,-1, 1,-1, 1, 1, 1, 1,-1, 1,-1, 1,
     1,-1, 1, 1,-1,-1,-1,-1,-1, 1, 1,-1, 1, 1,-1, 1
};

static __constant__ float d_tbq_wht_signs2[128] = {
     1, 1, 1, 1,-1, 1, 1,-1, 1,-1,-1,-1, 1,-1,-1,-1,
     1, 1,-1,-1, 1,-1, 1,-1, 1,-1,-1, 1,-1, 1, 1, 1,
     1, 1,-1,-1,-1, 1,-1,-1,-1,-1,-1,-1, 1, 1, 1,-1,
     1,-1, 1, 1, 1,-1,-1, 1,-1,-1,-1,-1,-1,-1, 1, 1,
     1,-1, 1,-1,-1,-1,-1, 1,-1, 1,-1, 1,-1,-1, 1, 1,
    -1, 1,-1, 1, 1,-1, 1,-1,-1,-1,-1, 1,-1,-1, 1,-1,
     1,-1, 1, 1, 1,-1,-1, 1,-1, 1,-1, 1, 1,-1,-1, 1,
    -1, 1,-1, 1, 1,-1, 1,-1, 1,-1,-1,-1,-1,-1, 1,-1
};

// Serial FWHT in shared memory -- called by a single thread.
// Requires smem[0..127] to be populated. Result is in-place.
// direction=0: forward (signs1 -> butterfly -> signs2)
// direction=1: inverse (signs2 -> butterfly -> signs1)
static __device__ __forceinline__ void tbq_fwht_128_serial(float * smem, int direction) {
    const float * s_pre  = (direction == 0) ? d_tbq_wht_signs1 : d_tbq_wht_signs2;
    const float * s_post = (direction == 0) ? d_tbq_wht_signs2 : d_tbq_wht_signs1;

    // Pre-sign flip
    for (int i = 0; i < 128; i++) smem[i] *= s_pre[i];

    // 7-stage butterfly
    for (int h = 1; h < 128; h *= 2) {
        for (int i = 0; i < 128; i += h * 2) {
            for (int j = i; j < i + h; j++) {
                float a = smem[j], b = smem[j + h];
                smem[j]     = a + b;
                smem[j + h] = a - b;
            }
        }
    }

    // Normalize + post-sign flip: multiply by 1/sqrt(128)
    const float inv_sqrt_128 = 0.08838834764831845f;
    for (int i = 0; i < 128; i++) smem[i] *= inv_sqrt_128 * s_post[i];
}

// Cooperative FWHT in shared memory -- 128 threads, each owns smem[tid].
// direction=0: forward (signs1 -> butterfly -> signs2)
// direction=1: inverse (signs2 -> butterfly -> signs1)
// Requires 128 threads and __syncthreads() between stages.
static __device__ __forceinline__ void tbq_fwht_128_coop(float * smem, int tid, int direction) {
    const float * s_pre  = (direction == 0) ? d_tbq_wht_signs1 : d_tbq_wht_signs2;
    const float * s_post = (direction == 0) ? d_tbq_wht_signs2 : d_tbq_wht_signs1;

    // Pre-sign flip
    smem[tid] *= s_pre[tid];
    __syncthreads();

    // 7-stage butterfly
    #pragma unroll
    for (int step = 1; step < 128; step <<= 1) {
        int pair = tid ^ step;
        float u = smem[tid];
        float v = smem[pair];
        __syncthreads();
        bool is_low = (tid & step) == 0;
        smem[tid] = is_low ? (u + v) : (v - u);
        __syncthreads();
    }

    // Normalize + post-sign flip
    smem[tid] *= 0.08838834764831845f * s_post[tid];
}

// Serial FWHT in registers -- for single-thread-per-group kernels (SET_ROWS).
// x[0..127] is modified in-place.
static __device__ __forceinline__ void tbq_fwht_128_reg(float * x, int direction) {
    const float * s_pre  = (direction == 0) ? d_tbq_wht_signs1 : d_tbq_wht_signs2;
    const float * s_post = (direction == 0) ? d_tbq_wht_signs2 : d_tbq_wht_signs1;

    for (int i = 0; i < 128; i++) x[i] *= s_pre[i];

    for (int h = 1; h < 128; h *= 2) {
        for (int i = 0; i < 128; i += h * 2) {
            for (int j = i; j < i + h; j++) {
                float a = x[j], b = x[j + h];
                x[j] = a + b; x[j + h] = a - b;
            }
        }
    }

    const float inv_sqrt_128 = 0.08838834764831845f;
    for (int i = 0; i < 128; i++) x[i] *= inv_sqrt_128 * s_post[i];
}

#endif // __CUDACC__
