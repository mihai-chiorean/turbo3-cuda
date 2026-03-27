# Session State — TurboQuant CUDA (Madreag/turbo3-cuda)

**Last updated**: 2026-03-28 Session 3
**Branch**: `release/turbo3-cuda`
**Latest commit**: `65da87a21`

## HEADLINE: turbo3 BEATS q8_0 at long context

Sparse V dequant (skip V for attention weight < 1e-6) inverts the context
scaling curve. turbo3 is now FASTER than q8_0 at 32K by 6.5%.

### Decode (tg128 tok/s)
| Context | q8_0 | turbo3 | Ratio |
|---------|------|--------|-------|
| short | 56.19 | 52.61 | 0.936x |
| 2K | 55.64 | 52.61 | 0.946x |
| 16K | 50.27 | 49.96 | 0.994x |
| 32K | 45.28 | 48.21 | **1.065x** |

### Prefill (pp512 tok/s)
| Depth | q8_0 | turbo3 | Ratio |
|-------|------|--------|-------|
| 0 | 3001 | 2931 | 0.977x |
| 32K | 2303 | 2235 | 0.970x |

### PPL
| Config | PPL | vs q8_0 |
|--------|-----|---------|
| turbo3 + sparse V | 6.854 | +1.41% |
| turbo3 + sparse V + LA-1 | ~6.79 | ~+0.5% |

## Key Optimizations Implemented
1. **Persistent fp16 shadow cache** — incremental dequant, O(1) per token
2. **Sparse V dequant** — skip 90%+ of V positions at long context
3. **Layer-adaptive KV** — promote sensitive layers to q8_0
4. **Norm correction** — exact L2 norm reconstruction
5. **16-byte struct padding** — GDDR7 coalescing
6. **Register LUT + batched bytes** — efficient FA inner loop
7. **Auto-enable FA** — prevent silent turbo3 failures
8. **Prefill dequant+MMA** — tensor core acceleration for prefill

## turbo4 Status
FA path complete (vec_dot, V dequant, shadow cache, dispatch, instances).
SET_ROWS quantize kernel complete.
NOT YET TESTABLE — needs additional ggml-cuda.cu op wiring for convert/cpy.
Crashes during graph reservation.

## What's Left
1. Debug turbo4 end-to-end (convert.cu, ggml-cuda.cu op dispatch)
2. Update README with benchmark table
3. Push to GitHub
4. TheTom Tier 3 diagnostic
5. Post results to llama.cpp discussion #20969
