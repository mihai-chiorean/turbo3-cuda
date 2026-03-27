# Session State — TurboQuant CUDA (Madreag/turbo3-cuda)

**Last updated**: 2026-03-28 Session 4
**Branch**: `release/turbo3-cuda`
**Latest commit**: `291d29a04` (README update)
**Pushed to GitHub**: Yes (https://github.com/Madreag/turbo3-cuda)

## Current Performance

### Decode (turbo3 vs q8_0)
| Context | Ratio | Status |
|---------|-------|--------|
| short | 0.942x | 5.8% gap |
| 8K | 0.951x | TARGET HIT |
| 16K | 0.992x | Near parity |
| 32K | 1.063x | BEATS q8_0! |

### Other
- Prefill: 0.977x (target hit)
- PPL: +1.41% uniform, +0.67% with LA-1

## Session 4 Completed
- [x] Online research (RotorQuant, FP4 SM120, spiritbuun sparse V)
- [x] Research findings written to .trash/RESEARCH_FINDINGS.md
- [x] Confirmed shadow > native at ALL depths
- [x] README rewritten with full benchmarks
- [x] Pushed to GitHub
- [x] TheTom diagnostic running

## In Progress
- TheTom Tier 3 diagnostic (running in background)

## What's Next
1. Commit diagnostic results when done
2. Try more aggressive sparse V threshold (1e-4 instead of 1e-6)
3. Investigate short-context improvement approaches
4. turbo4 end-to-end debugging (convert.cu, ggml-cuda.cu wiring)
