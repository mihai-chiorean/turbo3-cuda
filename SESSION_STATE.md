# Session State — TurboQuant CUDA

**Last updated**: 2026-03-28 03:15 MST
**Branch**: `release/turbo3-cuda`
**Latest commit**: `b4a238d5c` (Tier 2 gate report)

## Commits This Session (6)
1. `8f91a5d75` — dequant-to-fp16 prefill+decode (per-call, fixed prefill regression)
2. `57f9a3bff` — persistent fp16 shadow cache (incremental dequant)
3. `91bb63001` — cleanup (L2 hints negative, single-row kernel no improvement)
4. `4b1a8fa1b` — layer-adaptive KV cache (TURBO_LAYER_ADAPTIVE modes 1-5)
5. `f15ea3462` — bf16 bypass validation (debunked)
6. `b4a238d5c` — Tier 2 gate report + CHANGELOG

## Current Performance

### Decode (tg128 tok/s ratio vs q8_0)
| Context | Ratio | Session 1 |
|---------|-------|-----------|
| short | 0.931x | 0.885x |
| 4K | 0.928x | 0.881x |
| 8K | 0.912x | 0.838x |
| 32K | ~0.94x | 0.719x |

### Prefill (pp512 ratio vs q8_0)
| Depth | Ratio | Session 1 |
|-------|-------|-----------|
| 0 | 0.977x | 0.624x |
| 8K | 0.979x | 0.108x |
| 32K | 0.970x | 0.038x |

### PPL
| Config | PPL | vs q8_0 |
|--------|-----|---------|
| turbo3 | 6.867 | +1.60% |
| turbo3 LA-1 | 6.804 | +0.67% |
| q8_0 | 6.759 | baseline |

## What's Done
- [x] Fix prefill regression (0.038x → 0.977x)
- [x] Persistent fp16 shadow cache
- [x] Layer-adaptive modes 1-5
- [x] bf16 bypass validation
- [x] Tested L2 hints (negative), single-row kernel (no help)
- [x] Tier 2 benchmark + gate report

## What's Next (Priority Order)
1. **turbo4 CUDA port** — spiritbuun's implementation studied, need to port:
   - turbo-quant-cuda.cuh: quantize_f32_turbo4_0_block, dequantize_turbo4_0, QJL sign arrays
   - fattn-common.cuh: FA vec_dot_turbo4_0, dequantize_V_turbo4_0
   - fattn.cu: dispatch + shadow for turbo4
   - set-rows.cu: turbo4 SET_ROWS dispatch (already has stub using template)
   - getrows.cu: turbo4 GET_ROWS
   - Template instances for turbo4
2. Backend-ops test integration
3. README update + GitHub push
4. TheTom Tier 3 diagnostic

## Architecture Notes
- **Shadow cache** keyed on turbo3 tensor data pointer (std::unordered_map)
- **Incremental dequant**: only new KV positions, O(1) per token
- **Multi-seq (ne[3]>1)** falls back to native turbo3 vec kernel
- **GGML_TURBO_DECODE_NATIVE=1** bypasses shadow, uses native turbo3
- **TURBO_LAYER_ADAPTIVE=N** promotes specific layers to q8_0
- **Prefill** uses per-call alloc+dequant+MMA+free (acceptable for bulk ops)

## What Failed (Don't Retry)
- L2 persistence hints (cudaAccessPolicyWindow) — net negative
- Single-row dequant kernel — no improvement over multi-block
- Hybrid native/shadow approach — worse at short context
- Fused compressed attention (vec_dot drop-in) — math correct but not faster
