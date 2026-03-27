# TurboQuant Session State — Resume From Here

## Git Status
```
Branch: release/turbo3-cuda
Latest commit: d3119d33c (bench: document profiling tool attempts)
Working tree: clean
```

## Completed Items
- 0.1: FA dispatch safety (forces vec-only for turbo3) ✅
- 0.2: Auto-enable FA for turbo cache types ✅
- 0.3: Dead code removal in set-rows.cu ✅
- 0.4: Profiling analysis from decode curve data ✅
- 1.1: Norm correction in SET_ROWS quantizer ✅
- 1.2: 16-byte struct padding (+11.7% decode at 32K) ✅
- 1.3: __launch_bounds__ on turbo kernels ✅
- 1.4: Centroid access verified as constexpr in FA ✅
- 3.3: Register LUT + batched byte extraction (+2.1% additional at 32K) ✅
- Tier 2 gate: measured and documented ✅
- Tier 3 diagnostic: zip saved in benchmarks/community/ ✅

## Current Numbers (after all optimizations)
Decode (tg128 tok/s):
  short: q8_0=59.40, turbo3=52.59 (0.885x)
  4K:    q8_0=60.05, turbo3=52.87 (0.881x)
  8K:    q8_0=59.31, turbo3=49.73 (0.838x)
  16K:   q8_0=57.15, turbo3=45.83 (0.802x)
  32K:   q8_0=54.69, turbo3=39.30 (0.719x)

PPL: turbo3=6.867 vs q8_0=6.759 (+1.6%)
Prefill: turbo3=1949 vs q8_0=3124 (0.624x) — REGRESSED from FA dispatch fix

## Items NOT Done (continue from here)
1. **2.1 Fused FA kernel** — drop-in vec_dot prototype was reverted (slower).
   Needs: new fattn-vec-turbo3.cuh with outer KV loop, loads uint8 indices,
   Q binning + 8 centroid muls. This is the main decode speedup path.
   Register LUT was done as interim improvement.

2. **2.2 Dequant-then-MMA prefill** — prefill regressed to 0.624x from FA dispatch
   fix forcing vec-only kernel. Need bulk dequant → cuBLAS GEMM pipeline.

3. **3.1 turbo4 CUDA port** — port from spiritbuun's fork
4. **3.2 Asymmetric K/V** — K=turbo3, V=fp16 and K=turbo3, V=q8_0
5. **3.4 Layer-adaptive KV cache**
6. **4.1 L2 cache residency** (__ldg for turbo3 blocks)
7. **5.2 FA instance coverage** (D=192, 320, 576)
8. **5.3 Backend-ops test integration**
9. **5.4 bf16 bypass validation** — test turbo3 vs bf16 vs f16 PPL
10. **5.5-5.8 Documentation items**
11. **README update + GitHub push**

## Key Files
- Model: /home/erol/ai/turboquant/models/opus-v2-Q6_K.gguf
- Wikitext: wikitext-2-raw/wiki.test.raw
- Plan: IMPROVEMENT_PLAN.md (in repo root, also .trash/ copy)
- Benchmarks: benchmarks/CHANGELOG.md, benchmarks/gate/, benchmarks/community/
- spiritbuun's fork: /home/erol/projects/llama-cpp-turboquant-cuda/

## Build Command
```bash
/home/erol/miniconda3/envs/tq/bin/cmake --build build -j$(nproc)
```

## Critical Architecture Notes
- RTX 5090 SM120: 128KB smem/SM, 48 warps, NO WGMMA/TMEM
- Qwen 3.5 27B: 16 GA layers (with KV), 48 GDN layers (no KV), head_dim=256
- turbo3 block: 16 bytes (padded from 14), 32 values per block
- FA vec_dot centroids are constexpr C[8] in registers (not __constant__)
- llama-bench uses -d for context depth, NOT -c
