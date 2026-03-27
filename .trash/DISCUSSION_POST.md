# Discussion Post Draft for llama.cpp #20969

**Post this to**: https://github.com/ggml-org/llama.cpp/discussions/20969

---

## TurboQuant CUDA — RTX 5090 Results (turbo3 beats q8_0 at long context)

Sharing results from our RTX 5090 (SM120 Blackwell) implementation of TurboQuant KV cache compression.

**Repo**: https://github.com/Madreag/turbo3-cuda (branch: `release/turbo3-cuda`)

### Key Finding: turbo3 beats q8_0 at 16K+ context

On the RTX 5090 with Qwen 3.5 27B (dense, 16 attention layers):

| Context | q8_0 tok/s | turbo3 tok/s | Ratio |
|---------|-----------|-------------|-------|
| short | 55.05 | 51.95 | 0.944x |
| 8K | 54.79 | 51.92 | 0.951x |
| 16K | 50.26 | 49.85 | **0.993x** |
| 32K | 45.96 | 47.76 | **1.039x** |

PPL: turbo3 = 6.848 (+1.32% at ctx=512), 5.736 (+1.08% at ctx=2048). With layer-adaptive (first+last 4 layers at q8_0): 6.804 (+0.67%).

### MoE Model (apples-to-apples vs spiritbuun)

Qwen 3.5 35B-A3B Q4_K_M (MoE, tiny KV cache):

| Context | q8_0 tok/s | turbo3 tok/s | Ratio |
|---------|-----------|-------------|-------|
| short | 186 | 158 | 0.846x |
| 32K | 134 | 131 | **0.975x** |

This matches spiritbuun's 0.97x result on MoE — confirming our implementations are in agreement.

### Architecture: Persistent fp16 Shadow Cache

Instead of dequanting the entire KV cache per token (spiritbuun's approach), we maintain a persistent fp16 shadow buffer that's incrementally updated:

- **Decode**: Only dequant 1 new KV position per token (~2 KB instead of ~64 MB at 32K)
- **Prefill**: Bulk dequant to temp buffers + MMA Tensor Cores (0.977x of q8_0)
- **Sparse V skip**: Skip V positions with attention weight < 1e-4 (based on TheTom's research)

The crossover point where turbo3 beats q8_0 is around 16K context — at that point the 4.6x bandwidth reduction outweighs the ~5.6% shadow overhead.

### New in Session 8

- **Asymmetric K=turbo3, V=q8_0**: PPL only +0.67% at ctx=512, and actually better than q8_0 at ctx=2048 (5.650 vs 5.674)
- **turbo4 end-to-end**: 4.25 bpv with QJL residual correction. PPL 5.743 at ctx=2048.
- Both dense and MoE model validation

### Credits

Huge thanks to:
- **@TheTom** for the original Metal implementation, sparse V dequant research, and diagnostic scripts
- **@spiritbuun** for the CUDA reference implementation, norm correction, and layer-adaptive ideas
- **Google Research** for the TurboQuant paper (ICLR 2026)

### Looking for

- Community testing on other models and GPU architectures (RTX 4090, 3090)
- Feedback on the asymmetric K/V approach
- Anyone interested in the FP4 Tensor Core attention moonshot (SM120 natively supports mma.sync for FP4 E2M1)

Build instructions and full benchmarks in the README: https://github.com/Madreag/turbo3-cuda
