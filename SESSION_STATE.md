# Session State — TurboQuant CUDA

**Last updated**: 2026-03-28 04:15 MST
**Branch**: `release/turbo3-cuda`  
**Latest commit**: `ac033c695` (turbo4 GET_ROWS dequant)

## Session 2 Commits (11 total)
1. `8f91a5d75` — perf: dequant-to-fp16 prefill+decode (per-call approach)
2. `57f9a3bff` — perf: persistent fp16 shadow cache (incremental dequant)
3. `91bb63001` — refactor: cleanup (L2 hints negative, single-row no improvement)
4. `4b1a8fa1b` — feat: layer-adaptive KV cache (TURBO_LAYER_ADAPTIVE modes 1-5)
5. `f15ea3462` — test: bf16 bypass validation (debunked)
6. `b4a238d5c` — bench: Tier 2 gate report + CHANGELOG
7. `4ef291fde` — docs: session state update
8. `103805436` — feat: turbo4 dequant kernels + QJL sign arrays
9. `1f3875367` — docs: session state with turbo4 port progress
10. `abe0f3700` — feat: turbo4 FA integration (vec_dot, V dequant, shadow, dispatch)
11. `ac033c695` — feat: turbo4 GET_ROWS dequant in dequantize.cuh

## Current Performance (turbo3)
| Metric | Value | vs q8_0 |
|--------|-------|---------|
| Decode short | ~51 tok/s | ~0.93x |
| Decode 32K | ~47 tok/s | ~0.94x |
| Prefill d0 | ~2930 tok/s | 0.977x |
| Prefill d32K | ~2235 tok/s | 0.970x |
| PPL (uniform) | 6.867 | +1.60% |
| PPL (LA-1) | 6.804 | +0.67% |

## turbo4 Port Status
### DONE:
- Dequant kernels (contiguous fp16/fp32/bf16) in turbo-quant.cu
- QJL sign arrays (d_turbo_qjl_signs1/2) in turbo-quant.cu
- FA vec_dot_fattn_vec_KQ_turbo4_0 in fattn-common.cuh
- FA dequantize_V_turbo4_0 in fattn-common.cuh
- Shadow cache turbo4 dequant kernel in fattn.cu
- FA dispatch + kernel selection for turbo4 in fattn.cu
- GET_ROWS dequant in dequantize.cuh
- Template instance fattn-vec-instance-turbo4_0-turbo4_0.cu
- CMakeLists.txt turbo4 glob

### REMAINING for turbo4:
1. **SET_ROWS turbo4 quantize** — the quantize_f32_turbo4_0_block function
   (FWHT rotation → 3-bit pack → QJL cross-space residual → norm correction)
   Reference: spiritbuun's turbo-quant-cuda.cuh lines 140-188
2. **getrows.cu** — wire turbo4 into GET_ROWS dispatch
   (dequantize_turbo4_0 is defined but may not be registered in dispatch table)
3. **convert.cu** — turbo4 row dequant for tensor conversion
4. **ggml.c** — GGML_TYPE_TURBO4_0 type traits (may already exist)
5. **llama-bench** — add turbo4 to arg parser (like turbo3 was added)

## What's Left After turbo4
1. README update with comprehensive benchmarks
2. Push to GitHub release branch
3. TheTom Tier 3 diagnostic
4. Backend-ops test integration (if time)

## Continuation Prompt
> Continue the TurboQuant CUDA implementation. Read SESSION_STATE.md at
> `/home/erol/ai/turboquant/research/llama-cpp-turboquant/SESSION_STATE.md`
> 
> turbo4 FA path is complete. Remaining turbo4 work:
> 1. Add quantize_f32_turbo4_0_block to turbo-quant.cu (SET_ROWS quantize)
>    — reference: spiritbuun's turbo-quant-cuda.cuh lines 140-188
> 2. Wire turbo4 into getrows.cu dispatch table
> 3. Wire turbo4 into convert.cu for row dequant
> 4. Add turbo4 to llama-bench arg parser
> 5. Test with -ctk turbo4 -ctv turbo4
> 
> After turbo4: README update, GitHub push, TheTom diagnostic.
> spiritbuun reference: /home/erol/projects/llama-cpp-turboquant-cuda/
