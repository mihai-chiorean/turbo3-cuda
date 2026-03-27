# Session State — TurboQuant CUDA

**Updated**: 2026-03-28 Session 6
**Branch**: `release/turbo3-cuda`
**Latest commit**: `0f4f6fbdb`

## Performance (unchanged)
| Context | q8_0 | turbo3 | Ratio |
|---------|------|--------|-------|
| short | 55.05 | 51.95 | 0.944x |
| 32K | 45.96 | 47.76 | 1.039x |

PPL: 6.848 (+1.32% at 512), 5.736 (+1.08% at 2048)

## Session 6 Attempts
1. **Fused SET_ROWS fp16 write**: FAILED — shadow buffer lifecycle management is broken.
   The shadow doesn't properly track KV cache clears between perplexity chunks, causing
   stale data to be served. Need proper KV cache lifecycle hooks (not available in current
   ggml architecture). Reverted.

2. **MoE model download**: IN PROGRESS — downloading Qwen3.5-35B-A3B Q4_K_M (~19 GB).

## Key Learning
The fused SET_ROWS approach requires hooking into the KV cache clear/reset lifecycle.
Without that, the shadow serves stale data after cache clears. The current incremental sync
approach (turbo_shadow_sync) correctly handles this via ne[1] < filled detection.

The remaining 5.6% short-context gap is from:
1. fp16 shadow stride mismatch (capacity-based head stride vs tight stride)
2. unordered_map lookup overhead (32 lookups per token)
3. Per-layer incremental dequant kernel launch (32 launches per token)

## Continuation
> MoE model downloading. After download: benchmark both models.
> Then: try KV cache lifecycle hook approach for fused writes.
> Or: accept 0.944x short-context as the tradeoff for 4.6x compression.
