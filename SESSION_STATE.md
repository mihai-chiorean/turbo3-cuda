# Session State — TurboQuant CUDA

**Last updated**: 2026-03-28 03:45 MST
**Branch**: `release/turbo3-cuda`
**Latest commit**: `103805436` (turbo4 dequant kernels)

## Session 2 Commits (8 total)
1. `8f91a5d75` — perf: dequant-to-fp16 prefill+decode
2. `57f9a3bff` — perf: persistent fp16 shadow cache
3. `91bb63001` — refactor: cleanup
4. `4b1a8fa1b` — feat: layer-adaptive KV cache
5. `f15ea3462` — test: bf16 bypass validation
6. `b4a238d5c` — bench: Tier 2 gate report
7. `4ef291fde` — docs: session state update
8. `103805436` — feat: turbo4 dequant kernels (partial port)

## Current Performance
| Metric | Value | vs q8_0 |
|--------|-------|---------|
| Decode short | 50.35 tok/s | 0.931x |
| Decode 32K | 47.37 tok/s | ~0.94x |
| Prefill d0 | 2931 tok/s | 0.977x |
| Prefill d32K | 2235 tok/s | 0.970x |
| PPL (uniform) | 6.867 | +1.60% |
| PPL (LA-1) | 6.804 | +0.67% |

## turbo4 Port Status (IN PROGRESS)
### Done:
- dequantize_block_turbo4_0_kernel (contiguous fp16/fp32/bf16)
- turbo4_unpack_3bit helper
- QJL sign arrays (d_turbo_qjl_signs1/2) in turbo-quant.cu
- TURBO3_MIDPOINTS_C for nearest-centroid lookup
- Host launchers dequantize_row_turbo4_0_{fp16,fp32,bf16}_cuda

### Remaining turbo4 work:
1. **fattn-common.cuh**: Add vec_dot_fattn_vec_KQ_turbo4_0 and dequantize_V_turbo4_0
   - Reference: spiritbuun's fattn-common.cuh lines 340-400
   - turbo4 uses packed 3-bit indices (not 2+1 split like turbo3)
   - Needs QJL scale: qjl_scale = 1.2533141f / 128.0f * rnorm
2. **fattn.cu**: Add TURBO4_0 to vec dispatch + shadow cache support
   - fattn-vec dispatch: add FATTN_VEC_CASES_ALL_D(TURBO4_0, TURBO4_0)
   - get_best_fattn_kernel: add TURBO4_0 to allowed types
   - turbo_shadow_sync: handle TURBO4_0 (needs different dequant kernel)
3. **set-rows.cu**: turbo4 SET_ROWS already has template stub, need quantize function
   - spiritbuun uses quantize_f32_turbo4_0_block (in turbo-quant-cuda.cuh)
   - Need: FWHT rotation → 3-bit pack → QJL signs → norm correction with QJL
4. **dequantize.cuh**: Add turbo4 dequant for get_rows interleaving
5. **Template instances**: fattn-vec-instance-turbo4_0-turbo4_0.cu
6. **ggml-cuda.cu**: Add turbo4 dispatch entries for convert ops

## What's Left After turbo4
- Backend-ops test integration
- README update with comprehensive benchmarks
- Push to GitHub
- TheTom Tier 3 diagnostic

## Key Files Reference
- spiritbuun turbo4: `/home/erol/projects/llama-cpp-turboquant-cuda/ggml/src/ggml-cuda/turbo-quant-cuda.cuh`
- spiritbuun FA: `/home/erol/projects/llama-cpp-turboquant-cuda/ggml/src/ggml-cuda/fattn-common.cuh`
- spiritbuun set-rows: `/home/erol/projects/llama-cpp-turboquant-cuda/ggml/src/ggml-cuda/set-rows.cu`
- Our turbo-quant.cu: `ggml/src/ggml-cuda/turbo-quant.cu`
- Our fattn.cu: `ggml/src/ggml-cuda/fattn.cu`
- Our fattn-common.cuh: `ggml/src/ggml-cuda/fattn-common.cuh`

## Continuation Prompt
> Continue the TurboQuant CUDA implementation. Read SESSION_STATE.md for context.
> The turbo4 port is partially done (dequant kernels committed). Continue with:
> 1. Add turbo4 FA vec_dot + V dequant to fattn-common.cuh
> 2. Wire turbo4 into fattn.cu dispatch + shadow cache
> 3. Add turbo4 SET_ROWS quantize kernel
> 4. Add turbo4 to dequantize.cuh for get_rows
> 5. Create template instances
> 6. Test with -ctk turbo4 -ctv turbo4
> Reference spiritbuun's code at /home/erol/projects/llama-cpp-turboquant-cuda/
> After turbo4: README update, GitHub push, TheTom diagnostic.
