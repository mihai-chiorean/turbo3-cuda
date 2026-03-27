# Session State — TurboQuant CUDA

**Last updated**: 2026-03-28 (Session 3)
**Branch**: `release/turbo3-cuda`
**Latest commit**: `37a48eabb` (CHANGELOG with sparse V)

## BREAKTHROUGH: turbo3 BEATS q8_0 at long context

Sparse V dequant (skip V for attention weight < 1e-6) inverts the context
scaling curve. turbo3 is now FASTER than q8_0 at 32K by 6.5%.

| Context | q8_0 | turbo3 | Ratio |
|---------|------|--------|-------|
| short | 56.19 | 52.61 | 0.936x |
| 2K | 55.64 | 52.61 | 0.946x |
| 16K | 50.27 | 49.96 | 0.994x |
| 32K | 45.28 | 48.21 | **1.065x** |

PPL: 6.854 (+1.41% vs q8_0) — improved from 6.867 (removing quant noise helps)
PPL with LA-1: 6.804 (+0.67%)

## What's Next
1. Continue with Phase 2 plan items (turbo4 SET_ROWS, README, push)
2. Try to improve short-context ratio (currently 0.936x, target 0.95x)
3. Run TheTom Tier 3 diagnostic with final numbers
