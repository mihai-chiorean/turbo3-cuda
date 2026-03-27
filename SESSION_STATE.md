# Session State — TurboQuant CUDA

**Last updated**: 2026-03-28 02:00 MST
**Branch**: `release/turbo3-cuda`
**Latest commit**: `f15ea3462` (bf16 validation)

## Current Performance

### Decode (tg128 tok/s, persistent shadow cache)
| Context | q8_0 | turbo3 | Ratio |
|---------|------|--------|-------|
| short | 55.20 | 51.76 | 0.938x |
| 32K | 50.41 | 47.08 | 0.934x |

### Prefill (pp512 tok/s)
| Depth | q8_0 | turbo3 | Ratio |
|-------|------|--------|-------|
| 0 | 3046 | 2983 | 0.979x |
| 8K | 3029 | 2736 | 0.903x |
| 32K | 2484 | 2291 | 0.922x |

### PPL (ctx=512, 8 chunks)
| Config | PPL | vs q8_0 6.759 |
|--------|-----|---------------|
| turbo3 uniform | 6.867 | +1.60% |
| turbo3 LA-1 | 6.804 | +0.67% |
| q8_0 | 6.759 | baseline |
| f16 | 6.756 | -0.04% |

## Commits This Session
1. `8f91a5d75` — dequant-to-fp16 prefill+decode (per-call, fixed prefill regression)
2. `57f9a3bff` — persistent fp16 shadow cache (incremental dequant)
3. `91bb63001` — cleanup (removed dead kernels, L2 hints tested negative)
4. `4b1a8fa1b` — layer-adaptive KV cache (TURBO_LAYER_ADAPTIVE env var)
5. `f15ea3462` — bf16 bypass validation (debunked bf16 requirement)

## What's Done
- [x] Fix prefill regression (0.038x → 0.979x)
- [x] Persistent fp16 shadow cache (eliminates per-token bulk dequant)
- [x] Layer-adaptive modes 1-5 (PPL +1.6% → +0.67%)
- [x] bf16 bypass validation
- [x] Tested L2 persistence hints (net negative, removed)
- [x] Tested single-row dequant kernel (no improvement, removed)

## What's Next (in priority order)
1. FA instance coverage (D=192, 320, 576)
2. turbo4 CUDA port
3. Backend-ops test integration
4. Documentation updates (reasoning tokens, multi-GPU, FA-4)
5. Full Tier 2 benchmark suite
6. Update README with results
7. Push to GitHub
8. TheTom diagnostic

## Targets vs Current
| Metric | Current | Target | Status |
|--------|---------|--------|--------|
| Decode short | 0.938x | >0.95x | CLOSE |
| Decode 32K | 0.934x | >0.95x | CLOSE |
| Prefill | 0.979x | >0.95x | HIT |
| PPL (LA-1) | +0.67% | <+1% | HIT |
