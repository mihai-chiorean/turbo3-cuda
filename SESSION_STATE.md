# Session State — TurboQuant CUDA

**Updated**: 2026-03-28 Session 6 (context limit approaching)
**Branch**: `release/turbo3-cuda`
**Latest commit**: `351758bd3`

## Performance
| Context | q8_0 | turbo3 | Ratio |
|---------|------|--------|-------|
| short | 55.05 | 51.95 | 0.944x |
| 32K | 45.96 | 47.76 | 1.039x |

PPL: 6.848 (+1.32% at 512), 5.736 (+1.08% at 2048)

## Session 6 Summary
1. **Sparse V threshold 1e-4**: PPL improved to +1.32%. Committed.
2. **__expf in FA softmax**: Zero PPL impact, helps absolute throughput. Committed.
3. **Flat array shadow cache**: FAILED (hash collisions → PPL 25.93). Reverted.
4. **Fused SET_ROWS fp16 write**: FAILED (KV cache lifecycle tracking broken → wrong PPL). Reverted.
5. **Asymmetric K=turbo3 V=q8_0**: PPL 6.804 (+0.67%). Decode not working (needs f16+q8_0 instance). Committed as partial.
6. **MoE model download**: IN PROGRESS (background Python download of Qwen3.5-35B-A3B Q4_K_M)

## Key Findings
- Fused SET_ROWS needs KV cache lifecycle hooks that don't exist in ggml
- The incremental sync (turbo_shadow_sync) is the correct approach — it properly handles cache clears
- 5.6% short-context gap is architectural, not easily fixable without ggml-level changes
- spiritbuun's 0.97x was on MoE with tiny KV — apples-to-oranges vs our dense model

## Continuation Prompt
> Read SESSION_STATE.md. Branch: release/turbo3-cuda.
> MoE model downloading to /home/erol/ai/turboquant/models/ — check if done:
>   ls -la /home/erol/ai/turboquant/models/Qwen3.5-35B-A3B-Q4_K_M.gguf
> If downloaded, benchmark MoE model: decode at 0/4K/8K/32K for turbo3 and q8_0.
> Then: update README with both model results, push to GitHub, TheTom diagnostic.
> Remaining: turbo4 debug, asymmetric decode fix (need f16+q8_0 FA instance).
