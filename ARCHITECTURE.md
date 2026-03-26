# Architecture — turbo3 CUDA Implementation

Technical deep-dive into the turbo3 block format, FWHT rotation, CUDA kernel design, and Flash Attention integration.

## 1. turbo3 Block Format

Defined in `ggml/src/ggml-common.h`:

```c
#define QK_TURBO3 32
#define QK_TURBO3_GROUP 128

typedef struct {
    ggml_half  norm;                  //  2 bytes: L2 norm (fp16)
    uint8_t    qs[QK_TURBO3 / 4];    //  8 bytes: lower 2-bit indices (4 per byte)
    uint8_t    signs[QK_TURBO3 / 8]; //  4 bytes: upper 1-bit sign extension (8 per byte)
} block_turbo3_0;                     // 14 bytes total
```

### Memory layout (14 bytes per 32 values)

```
Offset  Size  Field     Contents
─────────────────────────────────────────
0       2     norm      fp16 L2 norm of the 128-element group
2       8     qs[8]     2-bit indices: 4 values packed per byte
10      4     signs[4]  1-bit sign extensions: 8 values packed per byte
─────────────────────────────────────────
Total: 14 bytes for 32 values = 3.5 bits/value
```

### Compression ratio
- fp16 baseline: 32 values × 2 bytes = 64 bytes
- turbo3: 14 bytes
- **Ratio: 4.57× compression**

### 3-bit index encoding

Each value gets a 3-bit index (0–7) into the Lloyd-Max centroid table:
- **Bits [1:0]** → stored in `qs` (2 bits, 4 values per byte)
- **Bit [2]** → stored in `signs` (1 bit, 8 values per byte)

Reconstruction: `idx = (qs_bits) | (sign_bit << 2)`

### Centroid table (Lloyd-Max optimal for unit Gaussian)

```
Index  Centroid      Meaning
──────────────────────────────
  0    -0.190685     Large negative
  1    -0.117832     Medium negative
  2    -0.065717     Small negative
  3    -0.021460     Near-zero negative
  4    +0.021460     Near-zero positive
  5    +0.065717     Small positive
  6    +0.117832     Medium positive
  7    +0.190685     Large positive
```

Note the symmetry: `centroid[i] = -centroid[7-i]`.

### Dequantization formula

```
value = centroid[idx] × norm
```

Where `norm` is the L2 norm of the original 128-element group (stored per-block as fp16, shared across 4 blocks in the same group).

### Hybrid architecture implications

turbo3 compression only applies to layers with a KV cache. For **Qwen3.5-27B** (the tested model), only **16 of 64 layers** are standard GatedAttention layers with KV cache. The other 48 layers are GatedDeltaNet (SSM-like, linear attention) which use a fixed-size recurrent state regardless of context length.

This means the per-token KV cost for Qwen3.5-27B is:
```
KV_per_token = 2 × 16 layers × 4 KV heads × 256 head_dim × (14/32) = 14,336 bytes ≈ 14 KB
```

For standard transformer models where all layers have KV cache (e.g. Llama-3.1-70B with 80 layers), the per-token cost is proportionally larger. The turbo3 compression ratio (4.6×) applies equally regardless of architecture.

## 2. FWHT Rotation

### Why rotation is needed

KV cache vectors have non-uniform coordinate distributions. TurboQuant applies a **random orthogonal rotation** to make coordinates approximately i.i.d. Gaussian, which is the optimal input distribution for scalar Lloyd-Max quantization.

The rotation uses the **Fast Walsh-Hadamard Transform** (FWHT) combined with random sign flips, which gives an O(n log n) random orthogonal transform instead of O(n²) for a general rotation matrix.

### Algorithm

For a 128-element group `x`:

1. **Apply sign array 1:** `x[i] *= signs1[i]` for i in 0..127
2. **FWHT butterfly:** 7 stages (log₂(128) = 7), each stage pairs elements with XOR-partner spacing
3. **Normalize:** `x[i] *= 1/sqrt(128)`
4. **Apply sign array 2:** `x[i] *= signs2[i]`

The inverse (used during dequantization at graph level) swaps signs1 ↔ signs2 (direction parameter).

### FWHT butterfly structure

```
Stage h=1:  pairs (0,1), (2,3), (4,5), ...
Stage h=2:  pairs (0,2), (1,3), (4,6), (5,7), ...
Stage h=4:  pairs (0,4), (1,5), (2,6), (3,7), ...
...
Stage h=64: pairs (0,64), (1,65), ..., (63,127)
```

Each butterfly operation:
```
a = x[j]
b = x[j + h]
x[j]     = a + b
x[j + h] = a - b
```

### Sign arrays

The 128-element sign arrays (`±1`) are fixed random sequences, identical across CPU/Metal/CUDA. They're stored in `__constant__` memory on CUDA for fast broadcast access.

### CUDA implementation

`kernel_turbo_wht` in `turbo-quant.cu`: each CUDA thread processes one complete 128-element group sequentially (128 registers per thread). This is serial per-group but parallelized across groups.

The SET_ROWS kernel also embeds an inline copy of the forward FWHT (normalize → rotate → quantize in one pass), avoiding a separate kernel launch.

## 3. Quantize Pipeline (SET_ROWS)

When the KV cache writes a new row, the turbo3 quantize path is:

```
Input: float[ne00] (one row of K or V, head_dim typically 128)
                    ↓
        ┌─────────────────────────────┐
        │  For each 128-element group │
        ├─────────────────────────────┤
        │  1. Compute L2 norm         │
        │  2. Normalize: x /= norm    │
        │  3. Apply signs1             │
        │  4. FWHT butterfly (7 stages)│
        │  5. Normalize × 1/√128      │
        │  6. Apply signs2             │
        │  7. For each block of 32:   │
        │     - Nearest centroid lookup│
        │     - Pack 2-bit + 1-bit    │
        │     - Store fp16 norm       │
        └─────────────────────────────┘
                    ↓
Output: block_turbo3_0[ne00/32] (quantized row)
```

### Nearest centroid lookup

Uses 7 midpoint comparisons (no branches, just additions of comparison results):
```cuda
uint8_t idx = 0;
idx += (val >= -0.154259f);
idx += (val >= -0.091775f);
idx += (val >= -0.043589f);
idx += (val >=  0.0f);
idx += (val >=  0.043589f);
idx += (val >=  0.091775f);
idx += (val >=  0.154259f);
```

This is branchless and maps directly to predicate instructions on NVIDIA GPUs.

### CUDA kernel: `kernel_set_rows_turbo3`

- **Grid:** `(ne01, ceil(n_groups/32))` — one row per x-block, groups parallelized in y
- **Block:** 32 threads — each thread handles one 128-element group
- **Registers:** 128 floats for the WHT working array
- **No shared memory** — each thread works independently

## 4. Dequantize Pipeline

Simpler than quantize — just index lookup and scale:

```
Input: block_turbo3_0 (14 bytes)
            ↓
    For each of 32 elements:
        1. Extract 2-bit from qs: low2 = (qs[j/4] >> ((j%4)*2)) & 0x3
        2. Extract 1-bit from signs: hi1 = (signs[j/8] >> (j%8)) & 0x1
        3. Reconstruct index: idx = low2 | (hi1 << 2)
        4. Lookup centroid: val = centroids[idx]
        5. Scale: output = val × norm
            ↓
Output: float or half (dequantized value)
```

Three CUDA kernel variants:
- **Contiguous:** `dequantize_block_turbo3_0_kernel` — flat 1D, used for MUL_MAT dequant
- **Non-contiguous:** `dequantize_block_turbo3_0_nc_kernel` — 4D strided, used for tensor operations
- **Inline (FA):** In `fattn-common.cuh`, dequant is inlined into the Flash Attention kernel

## 5. Flash Attention Integration

turbo3 integrates into llama.cpp's Flash Attention via two template functions in `fattn-common.cuh`.

### K dot product: `vec_dot_fattn_vec_KQ_turbo3_0`

Computes `dot(K_row, Q_row)` where K is in turbo3 format and Q is in float.

**Key design decision:** Mirrors the **f16 path** (not the quantized q4/q8 path):
- Uses `float2` pairs like the f16 version (not q8_1 quantized Q)
- Same `cpy_ne` / `cpy_nb` constants
- Same loop structure and Q_v indexing: `Q_v[k_KQ_0/nthreads + k_KQ_1]`
- `Q_q8_1 = false` for turbo3 (Q stays in float, not requantized to q8_1)

This works because turbo3's 3.5-bit values are dequantized to float inline, then dotted with the float Q vector — same numeric path as f16, just with a different source decode.

### V dequantize: `dequantize_V_turbo3_0`

Reads `ne` contiguous elements from turbo3 V cache starting at flat offset `i0`, outputs to `half` or `float`.

Used in the Flash Attention V accumulation loop where each thread loads a small tile of V values per iteration.

### Thread configuration (fattn-vec.cuh)

turbo3 uses the **same thread/block config as f16**:
```cuda
nthreads_KQ = (type_K == GGML_TYPE_F16 || type_K == GGML_TYPE_TURBO3_0) ? 128/cpy_nb : nthreads_KQ_q;
nthreads_V  = (type_V == GGML_TYPE_F16 || type_V == GGML_TYPE_TURBO3_0) ? 128/cpy_nb : nthreads_V_q;
V_rows_per_thread = (...F16 || ...TURBO3_0) ? 2*cpy_ne : 4;
```

This gives turbo3 the same parallelism as f16, which is appropriate because the dequant overhead per element is small (a few bit extractions + one FMA).

### Template instances

Two files provide all head-dim × V-type combinations:
- `fattn-vec-instance-turbo3_0-f16.cu` — turbo3 K, fp16 V (mixed mode)
- `fattn-vec-instance-turbo3_0-turbo3_0.cu` — turbo3 K, turbo3 V (full compression)

Each instantiates D=64, D=128, D=256.

## 6. CUDA Kernel Details

### Dequantize kernel
- **Threads:** 256 per block
- **Grid:** `ceil(total_elements / 256)`
- **Memory:** Global reads only (turbo3 blocks are small enough for L2 cache)
- **Compute:** 2 byte extractions + 1 table lookup + 1 FMA per element

### FWHT kernel (`kernel_turbo_wht`)
- **One thread per 128-element group** (serial per-group, parallel across groups)
- **Registers:** 128 floats = 512 bytes per thread
- **No shared memory** — no inter-thread communication needed
- **7 butterfly stages** with in-register swaps
- **Occupancy:** Limited by register pressure (128 floats/thread)

### SET_ROWS quantize kernel
- **Grid:** `(rows, ceil(groups_per_row/32))`
- **Block:** 32 threads
- **Per thread:** Processes one 128-element group (same register pressure as FWHT)
- **Includes inline FWHT** to avoid extra kernel launch
- **Writes 4 turbo3 blocks** per thread (4 × 32 = 128 elements)

## 7. Dispatch Wiring

### Operations and their dispatch files

| Operation | File | Function |
|-----------|------|----------|
| Dequantize (contiguous) | `convert.cu` | `ggml_get_to_fp{16,32}_cuda` dispatch tables |
| Dequantize (non-contiguous) | `convert.cu` | `ggml_get_to_fp{16,32,bf16}_nc_cuda` dispatch tables |
| Get rows | `getrows.cu` | `ggml_cuda_get_rows_switch_src0_type` switch |
| Set rows (quantize) | `set-rows.cu` | `ggml_cuda_op_set_rows` → `ggml_cuda_op_set_rows_turbo3` |
| Copy (same-type) | `cpy.cu` | `ggml_cuda_cpy` raw memcpy path |
| MUL_MAT | `ggml-cuda.cu` | Excluded from MMVQ/MMQ, falls through to cuBLAS-after-dequant |
| Flash Attention K·Q | `fattn-common.cuh` | `vec_dot_fattn_vec_KQ_turbo3_0` |
| Flash Attention V | `fattn-common.cuh` | `dequantize_V_turbo3_0` |
| FA dispatch | `fattn.cu` | `FATTN_VEC_CASES_ALL_D` entries |
| TURBO_WHT | `turbo-quant.cu` | `ggml_cuda_op_turbo_wht` |
| Op support | `ggml-cuda.cu` | `ggml_backend_cuda_device_supports_op` |
| Build system | `CMakeLists.txt` | Glob `fattn-vec*turbo3*.cu` templates |

### MUL_MAT routing

turbo3 is a quantized type (`ggml_is_quantized` returns true), but it's explicitly excluded from the MMVQ and MMQ paths:

```cuda
const bool is_turbo = (src0->type == GGML_TYPE_TURBO3_0 || src0->type == GGML_TYPE_TURBO4_0);
bool use_mul_mat_vec_q = ggml_is_quantized(src0->type) && !is_turbo && ...;
bool use_mul_mat_q     = ggml_is_quantized(src0->type) && !is_turbo && ...;
```

This routes turbo3 MUL_MAT through the dequantize-then-cuBLAS path, which is correct because turbo3 doesn't have optimized dot product kernels for weight multiplication (it's designed for KV cache, not weight storage).

## 8. Differences from Metal

The CUDA port is a faithful translation of TheTom's Metal kernels with these adaptations:

| Aspect | Metal | CUDA |
|--------|-------|------|
| Sign arrays | `constant` buffer | `__constant__` memory |
| FWHT | Single thread per group (Metal threads) | Single CUDA thread per group |
| SET_ROWS | Separate WHT op + quantize | Combined in `kernel_set_rows_turbo3` (also separate `kernel_turbo_wht` for graph-level) |
| FA K dot | Metal SIMD group functions | CUDA warp-level with `#pragma unroll` |
| FA V dequant | Metal SIMD loads | CUDA `#pragma unroll` loops |
| Thread config | Metal threadgroup sizes | Mirrors f16 path thread counts |
| Block quantize | `cpy_blck_f32_turbo3_0` in Metal | `quantize_f32_turbo3_0_block` in `cpy-utils.cuh` + `kernel_set_rows_turbo3` in `turbo-quant.cu` |
| Non-contiguous dequant | Metal kernel with strides | Separate `_nc_kernel` variant |

The centroid tables, midpoint tables, sign arrays, FWHT butterfly structure, and bit-packing logic are **identical** between Metal and CUDA.
