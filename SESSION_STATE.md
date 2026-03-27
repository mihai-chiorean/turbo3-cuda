# Session State — TurboQuant CUDA

**Last updated**: 2026-03-28 Session 4
**Branch**: `release/turbo3-cuda`
**Latest commit**: `af8d95992`
**GitHub**: https://github.com/Madreag/turbo3-cuda (pushed)

## Performance (DEFINITIVE, stable 3-rep)

### Decode (turbo3 vs q8_0)
| Context | q8_0 | turbo3 | Ratio |
|---------|------|--------|-------|
| short | 58.60 | 55.18 | 0.942x |
| 8K | 53.66 | 51.04 | 0.951x |
| 16K | 50.26 | 49.85 | 0.992x |
| 32K | 45.29 | 48.15 | **1.063x** |

### Prefill: 0.977x | PPL: +1.41% uniform, +0.67% LA-1

## Key Optimizations (all committed)
1. Persistent fp16 shadow cache (incremental dequant)
2. Sparse V dequant (skip weight < 1e-6)
3. Layer-adaptive KV (TURBO_LAYER_ADAPTIVE)
4. Norm correction, 16-byte padding, register LUT
5. Prefill dequant+MMA

## Session 4 Completed
- [x] Online research (RotorQuant, FP4 SM120, spiritbuun updates)
- [x] README rewritten with benchmarks
- [x] Pushed to GitHub
- [x] TheTom diagnostic running (background)
- [x] Tested 1e-4 threshold (PPL improved but benchmarks unstable due to diagnostic running)
- [x] Confirmed shadow beats native at ALL depths

## What's Left
1. Wait for diagnostic to finish, commit zip
2. turbo4 end-to-end debugging (convert.cu wiring)
3. Try fusing dequant into SET_ROWS for short-context improvement
4. FP4 Tensor Core research (SM120 mma.sync.m16n8k64)
5. Post results to llama.cpp discussion #20969

## Continuation Prompt
> Continue TurboQuant CUDA. Read SESSION_STATE.md.
> Branch: release/turbo3-cuda, pushed to GitHub.
> turbo3 beats q8_0 at 32K (1.063x). Short context at 0.942x.
> TheTom diagnostic may still be running — check background process.
> Next: commit diagnostic, try fusing dequant into SET_ROWS, turbo4 debug.
> spiritbuun ref: /home/erol/projects/llama-cpp-turboquant-cuda/
