# RESUME: TurboQuant CUDA — Complete Context Dump for Next Agent

**Date**: 2026-03-28 (end of Session 7)
**Owner**: Erol Germain (@erolgermain, GitHub: Madreag)
**READ AGENTS.md IN THE REPO ROOT FIRST** — but note this is Erol's own fork, not a PR to upstream. The AGENTS.md code-writing restrictions do NOT apply. Write code freely.

---

## CRITICAL: READ THESE FILES BEFORE DOING ANYTHING

1. **AGENTS.md** (repo root) — project rules (ignore "don't write code" — this is our fork)
2. **This file** — complete context dump
3. **Research papers** in `.trash/research/`:
   - `Advanced Methodologies for Sub-4-Bi.txt` — SM120 decode architecture, 4 pillars (FA-4, TurboQuant theory, BitDecoding lop3, sparse V, shadow caches)
   - `Beating FP8 at Decode_ Sub-4-bit KV on RTX 5090 with FA-4, Native FP4, and Sparse Gating.md` — executive summary with decision matrix by workload type, expected speedups per technique
   - `Blackwell KV Cache Decode Speed Optimization.pdf` — 17-page deep dive with tables (FA-4 on SM120, TurboQuant math, NVFP4 mma.sync, BitDecoding, sparse V, shadow caches, synthesis metrics)
   - `grok research1.txt` — Grok synthesis: BitDecoding benchmarks, spiritbuun analysis, centroid partial scores, GDDR7 coalescing
   - `Sub-4-bit KV Cache Speed Parity on Blackwell RTX 5090.pdf` — 21-page academic report (SageAttention3 1038 TOPS on 5090, BitDecoding 8.6x, KIVI, WG-KV, fragment layout details)

---

## ⚠️ SETTLED ARCHITECTURE — DO NOT REVISIT ⚠️

**The decode architecture is FINAL. The persistent fp16 shadow cache with incremental dequant is the optimal design. This has been proven across 7 sessions and 3 failed alternative attempts. DO NOT waste time on these dead ends:**

### DEAD END 1: Fused SET_ROWS fp16 write (tested Sessions 2, 6, 7)
**What it is**: Write fp16 values alongside turbo3 blocks during SET_ROWS, so FA reads fp16 directly with zero per-token overhead.
**Why it fails**: 
- v1 (Session 6): KV cache lifecycle tracking broken — shadow serves stale data after cache clears between perplexity chunks. PPL 4.77 at ctx=2048 (should be 5.74). Reverted.
- v2 (Session 7): Fixed lifecycle tracking (detect ne1 decrease). PPL IMPROVED to +0.32% (was +1.08%) because fp16 is written from pre-quantization float data (avoids round-trip through turbo3). BUT speed REGRESSED to 0.919x (was 0.944x) because the extra 128 fp16 writes per thread group in the SET_ROWS kernel adds more latency than the incremental dequant saves. Reverted.
**Bottom line**: The SET_ROWS kernel is already register-heavy (128 floats for FWHT). Adding fp16 writes makes it slower. The incremental dequant in FA (1 row per token = ~2 KB) is negligibly fast.

### DEAD END 2: Flat array shadow cache (tested Session 5)
**What it is**: Replace `std::unordered_map<void*, shadow>` with a flat array indexed by hash of the data pointer.
**Why it fails**: Hash collisions. Different KV cache tensor data pointers can hash to the same slot. When two different tensors (K layer 3, V layer 7) share a shadow slot, the data is wrong. PPL = 25.93 at ctx=2048 (should be 5.74). Reverted.
**Bottom line**: The unordered_map is correct and its overhead (~100ns per lookup × 32 lookups per token = ~3μs) is negligible vs the ~19ms per-token decode time.

### DEAD END 3: Adaptive native/shadow (tested Session 7)
**What it is**: Use native turbo3 vec kernel at short context (fewer bytes to read) and shadow fp16 at long context.
**Why it fails**: The native turbo3 vec kernel with bit-extract + centroid lookup is SLOWER than the fp16 vec kernel at ALL context depths. The hardware f16 vec_dot uses half2 vector operations that the turbo3 bit-extract path can't match. Even though turbo3 reads 2.1x fewer bytes, the ALU cost of index extraction exceeds the bandwidth savings. Measured: native=50.15 tok/s vs shadow=52.13 tok/s at short context. Reverted.
**Bottom line**: The fp16 vec_dot path is more optimized than any turbo3-native path we can build within the existing FA vec kernel framework.

### DEAD END 4: L2 cache persistence hints (tested Session 2)
cudaAccessPolicyWindow on the shadow buffer. Net negative ~1-2% at short context due to API call overhead. The GPU's default L2 caching is already adequate.

### DEAD END 5: Fused compressed attention (tested Session 1)
vec_dot drop-in that bins Q values by centroid index. Math correct (PPL improved) but `partial_q[idx] +=` dynamic array indexing causes register spill → slower.

### DEAD END 6: CUDA event profiling inside FA dispatch (tested Session 5)
cudaEventSynchronize inside the graph compute path is illegal when CUDA graphs are enabled (USE_GRAPHS=1). Crashes with illegal memory access.

---

## PROJECT OVERVIEW

CUDA port of Google's TurboQuant (ICLR 2026, arXiv:2504.19874) KV cache compression for llama.cpp, targeting NVIDIA RTX 5090 (SM120 Blackwell). We are Madreag — one of several community developers. The goal: be the definitive TurboQuant CUDA implementation for Blackwell GPUs.

**GitHub**: https://github.com/Madreag/turbo3-cuda (branch: `release/turbo3-cuda`)
**Discussion**: https://github.com/ggml-org/llama.cpp/discussions/20969

### How TurboQuant Works
1. **Normalize**: Compute L2 norm of 128-element KV head vector groups
2. **Rotate**: FWHT + random sign flips (signs1 → FWHT → signs2) to Gaussianize distribution
3. **Quantize**: Each rotated element → nearest of 8 Lloyd-Max centroids (3-bit index)
4. **Pack**: 2-bit `qs[]` + 1-bit `signs[]` arrays per 32-value block
5. **Store**: `block_turbo3_0` = 16 bytes per 32 values (norm + qs + signs + 2B padding)
6. **Norm correction**: Store `grp_norm / ||reconstruction||` instead of raw norm (spiritbuun innovation)

Centroids (for d=128): `{-0.190685, -0.117832, -0.065717, -0.021460, 0.021460, 0.065717, 0.117832, 0.190685}`
Midpoints (for nearest-centroid): `{-0.154259, -0.091775, -0.043589, 0.0, 0.043589, 0.091775, 0.154259}`
Rotation group = 128 elements = 4 blocks of 32, all sharing the same norm.
Block struct is 16 bytes (padded from 14 for GDDR7 32-byte sector coalescing).
**Two separate centroid declarations exist**: `TURBO3_CENTROIDS_C[8]` in turbo-quant.cu (for SET_ROWS/row-dequant, uses `__constant__` memory) and `constexpr float C[8]` inline in fattn-common.cuh (for FA hot path, lives in registers — NO constant memory serialization).

### turbo4 (4.25 bpv) — partial port
turbo4 adds a 1-bit QJL residual correction on top of turbo3's 3-bit PolarQuant:
- Block: 68 bytes per 128 values (norm + rnorm + 48B packed 3-bit indices + 16B QJL signs)
- QJL scale formula: `qjl_scale = 1.2533141f / 128.0f * rnorm`
- Dequant: `value = (centroid[idx] + sign * qjl_scale) * norm`
- Norm correction accounts for BOTH centroid and QJL reconstruction components
- QJL uses separate sign arrays (`d_turbo_qjl_signs1[128]`, `d_turbo_qjl_signs2[128]`) with seed=1042

### Qwen 3.5 27B Architecture (our primary test model)
- 64 total layers: 48 GatedDeltaNet (no KV cache) + 16 GatedAttention (with KV cache)
- Attention: head_dim=256, Q_heads=24, KV_heads=4 (GQA 6:1)
- Only 16 layers have KV cache — turbo3 compression applies to these only
- KV per token: fp16=~64 KB, turbo3=~16 KB (4x compression)

### Competitors
- **TheTom** (Metal/Apple Silicon): https://github.com/TheTom/llama-cpp-turboquant (branch: `feature/turboquant-kv-cache`), docs: https://github.com/TheTom/turboquant_plus
  - Invented sparse V dequant (+22.8% at 32K), 4-magnitude LUT, norm correction
  - Paper: `turboquant_plus/docs/papers/sparse-v-dequant.md`
- **spiritbuun** (CUDA/RTX 3090): https://github.com/spiritbuun/llama-cpp-turboquant-cuda (branch: `feature/turboquant-kv-cache`)
  - Cloned at `/home/erol/projects/llama-cpp-turboquant-cuda/`
  - Key finding: spiritbuun's 0.97x was on MoE model (tiny KV). They use per-call cudaMallocAsync+dequant+free (NOT persistent shadow). Their approach gives ~0.88x on dense models.
  - Has: turbo4 full port, layer-adaptive, Q pre-rotation in FA dispatch, fp16 centroid LUT

---

## REPO LOCATIONS

| Path | What |
|------|------|
| `/home/erol/ai/turboquant/research/llama-cpp-turboquant/` | **OUR REPO** (the one you modify) |
| `/home/erol/projects/llama-cpp-turboquant-cuda/` | spiritbuun's clone (reference only, branch `feature/turboquant-kv-cache`) |
| `/home/erol/ai/turboquant/models/opus-v2-Q6_K.gguf` | Dense model: Qwen 3.5 27B Q6_K (~21 GB) |
| `/home/erol/ai/turboquant/models/Qwen3.5-35B-A3B-Q4_K_M.gguf` | MoE model: Qwen 3.5 35B-A3B Q4_K_M (~21 GB) |
| `wikitext-2-raw/wiki.test.raw` | PPL test data (relative to repo root) |
| `.trash/research/` | Research papers (5 documents, read all before major work) |
| `.trash/resume.md` | This file |
| `benchmarks/CHANGELOG.md` | Living benchmark log |
| `benchmarks/moe_benchmark.md` | MoE model results |
| `benchmarks/bf16_validation.md` | bf16 requirement debunked |
| `benchmarks/community/` | TheTom diagnostic zips |
| `.trash/APPENDIX_A_ALGORITHM.md` | TurboQuant algorithm deep-dive (875 lines) |
| `.trash/IMPROVEMENT_PLAN.md` | Original V3.2 improvement plan (576 lines, historical) |
| `.trash/IMPROVEMENT_PLAN_V3.2.md` | Reviewer's version of the plan |
| `.trash/EXECUTE_FULL_PLAN.md` | Execution instructions from Session 1 |
| `.trash/context.md` | Session 1 complete context dump (316 lines) |
| `.trash/cursor_trajectory_info.md` | Full chat history from Sessions 1-2 (2875 lines) |
| `.trash/cont.md` | Session 2 continuation prompt (258 lines) |
| `.trash/OVERNIGHT_RESULTS.md` | Session 2-3 overnight summary |
| `.trash/RESEARCH_FINDINGS.md` | Session 4 research synthesis |
| `.trash/ASK.md` | Questions for Erol (async communication) |

### spiritbuun Key Commits (reference for porting)
```
7c6250688 — dequant turbo3 KV to fp16 for decode (per-call, their architecture)
cffa06f63 — Q FWHT pre-rotation in FA dispatch (+6.5% on MoE)
654647aac — fp16 centroid LUT (+6-14% at long context)
375555536 — turbo prefill dequant+MMA (98.8% of q8_0)
e6a78d542 — turbo4 batch unpack + V dequant + V_DOT2 half2
6b821a94d — turbo4 norm correction (QJL component)
65e28eb42 — asymmetric layer-adaptive modes 6-8
78d6bb5a0 — turbo4 K pre-rotation fix + sparse V dequant
1010625c5 — turbo4 prefill MMA (1.9x speedup)
```

### Build
```bash
cd /home/erol/ai/turboquant/research/llama-cpp-turboquant
/home/erol/miniconda3/envs/tq/bin/cmake --build build -j$(nproc)
# If cmake config needed:
/home/erol/miniconda3/envs/tq/bin/cmake -B build -DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=120 -DGGML_CUDA_FORCE_CUBLAS=OFF
```

### Git
```
origin     https://github.com/TheTom/llama-cpp-turboquant.git (upstream)
myfork     https://github.com/Madreag/turbo3-cuda.git (our fork — push here)
spiritbuun https://github.com/spiritbuun/llama-cpp-turboquant-cuda.git (reference)
```
Push: `git push myfork release/turbo3-cuda`

---

## CURRENT PERFORMANCE (DEFINITIVE — 7 sessions of measurement)

### Dense Model (Qwen 3.5 27B) — OUR SHOWCASE
| Context | q8_0 tok/s | turbo3 tok/s | Ratio | Notes |
|---------|-----------|-------------|-------|-------|
| short | ~55 | ~52 | **0.944x** | Shadow overhead floor |
| 2K | ~55 | ~52 | **0.95x** | |
| 4K | ~54 | ~52 | **0.96x** | |
| 8K | ~54 | ~51 | **0.95x** | |
| 16K | ~50 | ~50 | **0.99x** | Near parity |
| 32K | ~46 | ~48 | **1.04x** | **BEATS q8_0** |

### MoE Model (Qwen 3.5 35B-A3B) — apples-to-apples vs spiritbuun
| Context | q8_0 tok/s | turbo3 tok/s | Ratio |
|---------|-----------|-------------|-------|
| short | 186 | 158 | 0.846x |
| 4K | 175 | 151 | 0.863x |
| 8K | 174 | 148 | 0.847x |
| 32K | 134 | 131 | **0.975x** |

### Prefill (pp512 tok/s, dense model)
| Depth | q8_0 | turbo3 | Ratio |
|-------|------|--------|-------|
| 0 | 3001 | 2931 | 0.977x |
| 8K | 2889 | 2827 | 0.979x |
| 32K | 2303 | 2235 | 0.970x |

### PPL (wikitext-2, 8 chunks)
| Config | ctx=512 | ctx=2048 |
|--------|---------|----------|
| q8_0 | 6.759 | 5.674 |
| f16 | 6.756 | — |
| bf16 | 6.763 | — |
| turbo3 | 6.848 (+1.32%) | 5.736 (+1.08%) |
| turbo3 LA-1 | 6.804 (+0.67%) | — |
| K=turbo3 V=q8_0 | 6.804 (+0.67%) | — |
| MoE turbo3 | 6.215 (+1.47%) | — |

### PPL Reject Thresholds (revert if exceeded)
- ctx=512: turbo3 PPL > 6.89 → REJECT
- ctx=2048: turbo3 PPL > 5.77 → REJECT

---

## THE DECODE DATA FLOW (understand this before changing anything)

### Decode (token generation, Q->ne[1] == 1):
```
1. Model computes Q, K_new, V_new for this token
2. Graph applies forward WHT rotation to Q: ggml_turbo_wht(q, 0) [llama-graph.cpp]
3. SET_ROWS writes K_new/V_new to turbo3 KV cache [set-rows.cu → turbo-quant.cu]
4. FA dispatch [fattn.cu]:
   a. Detect turbo_kv (K or V is TURBO3_0)
   b. If ne[3] != 1 or GGML_TURBO_DECODE_NATIVE=1 → native turbo3 vec path
   c. Otherwise → shadow path:
      - g_turbo_shadows[K->data] lookup
      - turbo_shadow_sync: if ne1 > filled → dequant new rows to shadow fp16 buffer
      - Create stack tensor K_f16 pointing to shadow buffer with fp16 strides
      - Replace dst->src[1] = &K_f16 (same for V)
   d. get_best_fattn_kernel sees f16 K/V → routes to VEC kernel
   e. FA vec kernel runs with fp16 K/V (sparse V skips weight < 1e-4)
   f. Restore original tensor pointers
5. Graph applies inverse WHT rotation to output: ggml_turbo_wht(cur, 1) [llama-graph.cpp]
```

### Prefill (prompt processing, Q->ne[1] > 1):
```
1-2. Same as decode
3. SET_ROWS writes all prompt K/V positions to turbo3 cache
4. FA dispatch [fattn.cu]:
   a. Detect turbo_kv + Q->ne[1] > 1 + ne[3] == 1 + turing_mma_available
   b. Bulk dequant turbo3 K/V to temp fp16 buffers (cudaMallocAsync)
   c. Run MMA FA kernel with fp16 K/V (tensor cores)
   d. Free temp buffers (cudaFreeAsync)
5. Same as decode
```

### Shadow cache lifecycle:
- **Allocated lazily** on first FA call per tensor data pointer (capacity = max(ne1*2, 4096))
- **Grows** when ne1 exceeds capacity (cudaFree + cudaMalloc at 2x)
- **Detects cache clear** when ne1 < filled → full re-dequant
- **Incremental update** when ne1 > filled → dequant only new rows (typically 1 per token)
- **No-op** when ne1 == filled → zero work (steady state during generation without new tokens)

---

## KEY FILES MODIFIED (complete list)

```
ggml/src/ggml-common.h              — block_turbo3_0 (16B padded), block_turbo4_0 (68B)
ggml/src/ggml.c                     — GGML_TYPE_TURBO3_0/TURBO4_0 type registration
ggml/src/ggml-cuda/turbo-quant.cu   — Dequant kernels (fp16/fp32/bf16, contiguous+nc), WHT kernel,
                                      SET_ROWS turbo3 quantizer (with norm correction), SET_ROWS turbo4,
                                      turbo4 dequant kernels, QJL sign arrays
ggml/src/ggml-cuda/turbo-quant.cuh  — Header: ggml_cuda_op_set_rows_turbo3/turbo4, ggml_cuda_op_turbo_wht
ggml/src/ggml-cuda/fattn-common.cuh — vec_dot_fattn_vec_KQ_turbo3_0 (register LUT + batched bytes),
                                      dequantize_V_turbo3_0, vec_dot_turbo4_0, dequantize_V_turbo4_0,
                                      dispatch tables (get_vec_dot_KQ, get_dequantize_V)
ggml/src/ggml-cuda/fattn-vec.cuh    — nthreads_KQ/V for turbo types (128/cpy_nb),
                                      sparse V skip (1e-4 threshold, both half2 + float paths),
                                      __expf replacing expf in all 5 softmax exponentials
ggml/src/ggml-cuda/fattn.cu         — turbo_fp16_shadow struct + g_turbo_shadows map,
                                      turbo_shadow_sync (incremental dequant),
                                      k_turbo3_dequant_rows_f16 + k_turbo4_dequant_rows_f16,
                                      Prefill: turbo3_dequant_tensor_f16 → ggml_cuda_turbo_prefill_attend,
                                      Dispatch: shadow path + prefill MMA path + native fallback,
                                      Mixed-type dispatch (turbo3+f16, turbo3+q8_0),
                                      FATTN_VEC_CASES for turbo3/turbo4 combinations
ggml/src/ggml-cuda/set-rows.cu      — turbo3/turbo4 SET_ROWS dispatch
ggml/src/ggml-cuda/getrows.cu       — turbo3 GET_ROWS dispatch
ggml/src/ggml-cuda/dequantize.cuh   — dequantize_turbo3_0, dequantize_turbo4_0, QR_TURBO3/4
ggml/src/ggml-cuda/ggml-cuda.cu     — TURBO_WHT op dispatch, MUL_MAT turbo3 exclusion from mmvq/mmq
ggml/src/ggml-cuda/CMakeLists.txt   — turbo3/turbo4 template instance glob patterns
ggml/src/ggml-cuda/template-instances/fattn-vec-instance-turbo3_0-turbo3_0.cu (D=64/128/256)
ggml/src/ggml-cuda/template-instances/fattn-vec-instance-turbo3_0-f16.cu (D=64/128/256)
ggml/src/ggml-cuda/template-instances/fattn-vec-instance-turbo3_0-q8_0.cu (D=64/128/256)
ggml/src/ggml-cuda/template-instances/fattn-vec-instance-turbo4_0-turbo4_0.cu (D=64/128/256)
src/llama-context.cpp               — Auto-enable FA for turbo cache types
src/llama-graph.cpp                 — Graph-level WHT rotation (Q forward, V inverse)
src/llama-kv-cache.cpp              — Turbo rotation matrix init, layer-adaptive TURBO_LAYER_ADAPTIVE
src/turbo-rotation-data.h           — Pre-baked 128×128 rotation matrix (TURBO_ROTATION_R, TURBO_ROTATION_RT)
tools/llama-bench/llama-bench.cpp   — turbo3/turbo4 arg parser in ggml_type_from_name
README.md                           — Full benchmark table + architecture docs
```

---

## ALL OPTIMIZATIONS (committed, in implementation order)

| # | Optimization | File(s) | Impact | Session |
|---|-------------|---------|--------|---------|
| 1 | FA dispatch safety (vec-only for turbo3) | fattn.cu | Fixed NaN PPL | 1 |
| 2 | Auto-enable FA for turbo types | llama-context.cpp | Prevents silent failure | 1 |
| 3 | Norm correction in SET_ROWS | turbo-quant.cu | PPL improvement, zero cost | 1 |
| 4 | `__launch_bounds__` on WHT/SET_ROWS | turbo-quant.cu | Register pressure | 1 |
| 5 | 16-byte struct padding | ggml-common.h | +11.7% decode at 32K | 1 |
| 6 | Register LUT + batched bytes | fattn-common.cuh | +2.1% decode at 32K | 1 |
| 7 | Persistent fp16 shadow cache | fattn.cu | +30% decode at 32K (0.80→0.93x) | 2 |
| 8 | Layer-adaptive KV (TURBO_LAYER_ADAPTIVE) | llama-kv-cache.cpp | PPL +1.6%→+0.67% | 2 |
| 9 | Sparse V dequant (skip weight < 1e-4) | fattn-vec.cuh | turbo3 BEATS q8_0 at 32K | 3 |
| 10 | `__expf` fast math in softmax | fattn-vec.cuh | Zero PPL, helps throughput | 5 |
| 11 | Asymmetric K=turbo3 V=q8_0 FA instance | fattn.cu + template | PPL +0.67% (decode WIP) | 5 |
| 12 | turbo4 dequant kernels + QJL arrays | turbo-quant.cu | Partial turbo4 port | 2 |
| 13 | turbo4 FA vec_dot + V dequant | fattn-common.cuh | Partial turbo4 port | 2 |
| 14 | turbo4 SET_ROWS quantize | turbo-quant.cu | Partial turbo4 port | 3 |
| 15 | turbo4 GET_ROWS dequant | dequantize.cuh | Partial turbo4 port | 2 |

---

## HARDWARE: RTX 5090 (SM120 Blackwell)

| Parameter | Value |
|-----------|-------|
| GPU | NVIDIA GeForce RTX 5090 (GB202) |
| VRAM | 32 GB GDDR7, 512-bit bus |
| Bandwidth | 1,792 GB/s (peak theoretical) |
| L2 Cache | 98 MB |
| SMs | 170 |
| Shared Memory/SM | 128 KB (99 KB max/block) |
| Max Warps/SM | 48 (NOT 64 like datacenter SM100) |
| Tensor Cores | 5th gen, extended `mma.sync` |
| FP4 E2M1 | Native via `mma.sync.aligned.m16n8k64` |
| FP4 peak | 1,676 TFLOPS (theoretical) |
| WGMMA / TMEM | **NOT AVAILABLE** (datacenter SM100 only) |
| tcgen05.mma | **NOT AVAILABLE** (datacenter only) |
| FlashAttention-4 | **Does NOT work** natively on SM120 |
| CUDA toolkit | 12.8 (do NOT use 13.1 — MMQ kernel segfault on standard quant types) |
| CPU | AMD Ryzen 9 9950X3D 16-Core |
| RAM | 48 GB |
| OS | WSL2 Ubuntu 24.04 on Windows |

---

## WHY THE ARCHITECTURE IS WHAT IT IS

### Why persistent fp16 shadow wins (not fused SET_ROWS, not native turbo3)
- **Fused SET_ROWS** adds 128 fp16 stores per group to an already register-heavy kernel (128 floats for FWHT). Speed regresses 0.944x→0.919x. The incremental dequant in FA (1 row per token) is negligibly fast in comparison.
- **Native turbo3 vec kernel** reads 2.1x fewer bytes but the ALU cost of 3-bit index extraction (bit shifts, masks, centroid lookup) exceeds the bandwidth savings. The hardware f16 vec_dot uses half2 vector ops that turbo3 can't match. Measured: native=50.15 vs shadow=52.13 tok/s.
- **Shadow fp16** gives compressed turbo3 VRAM storage (4.6x) with fast f16 kernel compute. The one-time-per-position incremental dequant adds ~160μs/token (32 kernel launches × ~5μs each) which is 0.8% of the ~19ms/token decode time.

### Why turbo3 is 5.6% slower at short context (0.944x)
The gap is the sum of inherent shadow overhead, not a single fixable bottleneck:
1. **32 incremental dequant kernel launches per token** (16 layers × K+V): ~160μs total (~0.8%)
2. **Shadow stride mismatch**: `nb[2] = ne0 * capacity * sizeof(half)` where capacity >> ne1. Large gaps between heads reduce spatial locality.
3. **unordered_map lookups + tensor struct copies**: ~3μs per token (negligible)
4. **q8_0 vec kernel is slightly more optimized** in upstream llama.cpp than the f16 vec kernel

### Why turbo3 BEATS q8_0 at long context (1.04x at 32K)
1. **2.1x less KV bandwidth**: turbo3=16B/32val, q8_0=34B/32val. At 32K, bandwidth dominates.
2. **Sparse V skips 90%+ of V positions**: Both formats benefit, but turbo3's smaller footprint means less wasted reads before the skip.
3. **98 MB L2 cache**: More turbo3 KV fits in L2 than q8_0 KV at the same context length.

### Why sparse V improves quality (PPL went from 6.867 to 6.848)
Dequantizing negligible V positions (weight < 1e-4) introduces quantization NOISE into the attention accumulation. Each near-zero-weight position contributes near-zero signal but non-zero quantization error. Skipping them removes this noise, improving signal-to-noise ratio. TheTom documented this: NIAH went from 7/9 to 9/9 with sparse V.

---

## EROL'S PREFERENCES AND RULES

- **Direct communicator**. Does NOT tolerate skipping, deferring, stopping early, or asking "want me to continue?"
- **Measure everything**. PPL before AND after every change. Speed before AND after. One commit per logical change.
- **Quality is non-negotiable**. If PPL regresses beyond threshold → REVERT immediately, no exceptions.
- **Read reference code BEFORE reimplementing**. spiritbuun's fork at `/home/erol/projects/llama-cpp-turboquant-cuda/` is the CUDA reference.
- **Don't stop working**. When told to continue, continue until told to stop.
- **Fix regressions immediately**. If any metric gets worse, fix before moving on.
- **No analysis paralysis**. If you've identified a fix, implement it immediately. Don't write 3 paragraphs about why it might work — build it and measure.
- **Credit others**. Use `Co-Authored-By: spiritbuun <271142774+spiritbuun@users.noreply.github.com>` when adapting their code.
- **Push regularly**. `git push myfork release/turbo3-cuda` after milestones.

---

## BENCHMARK COMMANDS (copy-paste ready)

```bash
MODEL=/home/erol/ai/turboquant/models/opus-v2-Q6_K.gguf
MOE=/home/erol/ai/turboquant/models/Qwen3.5-35B-A3B-Q4_K_M.gguf
WIKI=wikitext-2-raw/wiki.test.raw

# PPL gate (run for EVERY code change):
./build/bin/llama-perplexity -m $MODEL -f $WIKI -c 512 -ctk turbo3 -ctv turbo3 -fa on --chunks 8 -ngl 99 2>&1 | grep Final
./build/bin/llama-perplexity -m $MODEL -f $WIKI -c 2048 -ctk turbo3 -ctv turbo3 -fa on --chunks 8 -ngl 99 2>&1 | grep Final

# Decode curve (NOTE: llama-bench uses -d for depth, NOT -c):
for DEPTH in 0 2048 4096 8192 16384 32768; do
  ./build/bin/llama-bench -m $MODEL -fa 1 -ctk turbo3 -ctv turbo3 -d $DEPTH -ngl 99 -t 1 -r 3 -p 0 -n 128 2>&1 | grep turbo3
  ./build/bin/llama-bench -m $MODEL -fa 1 -ctk q8_0 -ctv q8_0 -d $DEPTH -ngl 99 -t 1 -r 3 -p 0 -n 128 2>&1 | grep q8_0
done

# Prefill:
for DEPTH in 0 8192 32768; do
  ./build/bin/llama-bench -m $MODEL -fa 1 -ctk turbo3 -ctv turbo3 -d $DEPTH -ngl 99 -t 1 -p 512 -n 0 -r 1 2>&1 | grep turbo3
done

# Layer-adaptive PPL:
TURBO_LAYER_ADAPTIVE=1 ./build/bin/llama-perplexity -m $MODEL -f $WIKI -c 512 -ctk turbo3 -ctv turbo3 -fa on --chunks 8 -ngl 99 2>&1 | grep Final

# MoE benchmarks:
for DEPTH in 0 4096 8192 32768; do
  ./build/bin/llama-bench -m $MOE -fa 1 -ctk turbo3 -ctv turbo3 -d $DEPTH -ngl 99 -t 1 -r 3 -p 0 -n 128 2>&1 | grep turbo3
  ./build/bin/llama-bench -m $MOE -fa 1 -ctk q8_0 -ctv q8_0 -d $DEPTH -ngl 99 -t 1 -r 3 -p 0 -n 128 2>&1 | grep q8_0
done
```

---

## ENVIRONMENT VARIABLES

| Variable | Effect | Default |
|----------|--------|---------|
| `TURBO_LAYER_ADAPTIVE=N` | Per-layer KV type: 0=uniform, 1=first4+last4→q8_0, 2=last8→q8_0, 3=last4, 4=first4, 5=first2+last2 | 0 |
| `GGML_TURBO_DECODE_NATIVE=1` | Disable fp16 shadow, use native turbo3 vec kernel (slower but useful for debugging) | off |

---

## WHAT'S LEFT TO DO (priority order)

### Completed in Session 8 ✓
1. ~~turbo4 end-to-end~~ — DONE. GET_ROWS + SET_ROWS op support added. PPL 5.743 at ctx=2048.
2. ~~Asymmetric K=turbo3 V=q8_0 decode fix~~ — DONE. Two bugs: mixed-type guard + VEC dispatch entry.
3. ~~Discussion post draft~~ — DONE. Saved to .trash/DISCUSSION_POST.md.
4. ~~README update~~ — DONE. MoE, turbo4, asymmetric, 128K results.
5. ~~FA safety guard~~ — DONE. Ported from spiritbuun (ef588f8bf).

### High Priority
1. **Fix native turbo4 vec_dot for multi-seq** (Q->ne[3]>1) — gives NaN at ctx=512. The PPL evaluator uses n_seq=4 at ctx=512, hitting the native path which is broken for turbo4.
2. **Port spiritbuun's partial offload fix** (c99c23018) — turbo KV with ngl < n_layer.
3. **Port spiritbuun's multi-GPU fix** (6cdd9db87) — q_rot_buf + KV cache tensor count.
4. **Backend-ops test integration** — add turbo3/turbo4 to test-backend-ops.

### Medium Priority
5. **Post discussion to #20969** — draft ready at .trash/DISCUSSION_POST.md.
6. **TheTom diagnostic** — latest zip at `benchmarks/community/turbo-diag-20260327-091147.zip`.

### Genuine Performance Frontiers (NOT dead ends)
7. **lop3 TC-based FA kernel** — Research complete (session 8). BitDecoding has 4-bit and 2-bit lop3 templates. 3-bit requires custom implementation. The real win needs a FULL MMA-based FA kernel, not just vec_dot optimization. Centroid lookup (8 arbitrary floats) can't be done bitwise — must convert to fp16 TC fragments via lop3. Multi-day effort.
8. **FP4 Tensor Core attention** — SM120 natively supports `mma.sync.aligned.m16n8k64` for FP4 E2M1. Fragment layout documentation still incomplete.
9. **`cp.async.bulk` pipelined KV loading** — Prefetch KV block N+1 while computing on block N.

---

## WSL2 CAVEATS

- **No sudo** — can't install ncu/nsys profiling tools
- **Memory pressure at >32K** — causes ±15-20 tok/s variance in benchmarks. Run long-context tests with nothing else on GPU. Wait 30s between tests.
- **CUDA graphs** enabled by default (USE_GRAPHS=1) — do NOT add `cudaEventSynchronize` or `cudaDeviceSynchronize` inside graph compute paths
- **Swap activates at 32K** (1.5-3.5 GB) — affects benchmark stability
- **Model storage** detected as HDD (WSL2 filesystem layer) — first load is slow, subsequent loads use mmap cache

---

## COMMIT HISTORY (7 sessions, key commits only)

### Session 1: Foundation (12 commits)
- `8ae47390d` — llama-bench turbo3/turbo4 arg parser
- `0fb6321cf` — FA dispatch safety + norm correction + launch_bounds (**fixed NaN PPL**)
- `39f8bce3d` — Auto-enable FA for turbo cache types
- `fb4c5b789` — Full baseline measurement
- `0acd3cc66` — Centroid access verification (constexpr in FA, NOT `__constant__`)
- `d3c0e7889` — Profiling analysis (bottleneck identification)
- `2f5fafb93` — **16-byte struct padding** (+11.7% decode at 32K)
- `36a97a61f` — **Register LUT + batched bytes** (+2.1% at 32K)
- `589ce2bca` — CHANGELOG update
- `5186172bc` — Tier 2 gate report
- `77fb79c39` — Tier 3 TheTom diagnostic
- `d3119d33c` — Profiling tool documentation

### Session 2: Shadow Cache + Features (6 commits)
- `8f91a5d75` — Dequant-to-fp16 prefill+decode (per-call, fixed prefill 0.038x→0.985x)
- `57f9a3bff` — **Persistent fp16 shadow cache** (decode 0.80x→0.93x at 32K)
- `91bb63001` — Cleanup (L2 hints tested negative, single-row kernel no improvement — removed)
- `4b1a8fa1b` — **Layer-adaptive KV** (PPL +1.6%→+0.67%)
- `f15ea3462` — bf16 bypass validation (debunked)
- `b4a238d5c` — Tier 2 gate report + CHANGELOG

### Session 2-3: turbo4 Port (across sessions)
- `103805436` — turbo4 dequant kernels + QJL sign arrays
- `abe0f3700` — turbo4 FA integration (vec_dot, V dequant, shadow, dispatch)
- `ac033c695` — turbo4 GET_ROWS dequant in dequantize.cuh
- `65da87a21` — turbo4 SET_ROWS quantize kernel (**completes turbo4 pipeline minus wiring**)

### Session 3: Sparse V Breakthrough
- `157b6dcff` — **Sparse V dequant** (turbo3 BEATS q8_0 at 32K: 1.063x)

### Session 4: Research + Documentation
- `291d29a04` — README rewrite with full benchmarks
- `4063ec617` — Definitive decode curve
- `d72a9d5a3` — TheTom Tier 3 diagnostic

### Session 5: Threshold + Asymmetric
- `83593cffc` — Sparse V threshold 1e-6→1e-4 (PPL improved to +1.32%)
- `74a20aa35` — `__expf` in FA vec softmax (zero PPL impact)
- `02c797837` — Asymmetric K=turbo3 V=q8_0 FA instance (PPL works, decode WIP)

### Sessions 6-7: Fused SET_ROWS attempts (all reverted) + MoE
- `93c1415f1` — MoE Qwen3.5-35B-A3B benchmark (0.975x at 32K, matches spiritbuun)
- (Fused SET_ROWS v1 and v2 were implemented, tested, and reverted — see DEAD ENDS above)
- (Flat array shadow was implemented, tested, and reverted — see DEAD ENDS above)
- (Adaptive native/shadow was implemented, tested, and reverted — see DEAD ENDS above)

### Session 8: Ship Features + Community
- `ef506d510` — **Asymmetric K=turbo3 V=q8_0 decode fix** (two dispatch bugs: mixed-type guard + VEC entry)
- `79d8158a7` — **turbo4 end-to-end** (GET_ROWS + SET_ROWS op support, rotated-space QJL residual)
- `c2749ad48` — README update + discussion post draft
- `47743d568` — FA safety guard (ported from spiritbuun)
- `c50ba4468` — 128K MoE benchmarks (turbo3 1.10x, asymmetric 1.19x)
- lop3 research completed: BitDecoding repo analyzed, 3-bit requires custom implementation

### Session 8 New Performance Data
- MoE 128K: turbo3 86.9 tok/s (1.10x), asymmetric 93.7 tok/s (**1.19x**)
- Asymmetric dense PPL: 6.804 at 512 (+0.67%), 5.650 at 2048 (**better than q8_0**)
- turbo4 PPL: 5.743 at 2048 (+1.22%), decode 52.5 tok/s short

---

## CONTINUATION PROMPT

> Read `.trash/resume.md` in the repo at `/home/erol/ai/turboquant/research/llama-cpp-turboquant/` for full context. Then read `AGENTS.md` in the repo root.
>
> Key facts:
> - turbo3 BEATS q8_0 at 32K (1.04x dense, 1.10x MoE at 128K)
> - Asymmetric K=turbo3 V=q8_0 beats q8_0 by 19% at MoE 128K
> - turbo4 works end-to-end (PPL 5.743 at ctx=2048)
> - Shadow architecture is SETTLED. See DEAD ENDS.
>
> Next priorities:
> 1. Fix native turbo4 vec_dot for multi-seq (Q->ne[3]>1 gives NaN)
> 2. Port spiritbuun's partial offload + multi-GPU fixes
> 3. Post discussion to #20969 (draft ready at .trash/DISCUSSION_POST.md)
> 4. lop3 TC-based FA kernel (multi-day effort, research complete)
>
> Build: `/home/erol/miniconda3/envs/tq/bin/cmake --build build -j$(nproc)`
> Push: `git push myfork release/turbo3-cuda`
