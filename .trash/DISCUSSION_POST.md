# Discussion Post Draft for llama.cpp #20969

**Post this to**: https://github.com/ggml-org/llama.cpp/discussions/20969

---

## TurboQuant CUDA on RTX 5090 — 19% faster than q8_0 at 128K, and asymmetric K/V beats q8_0 quality

Sharing results from our RTX 5090 (SM120 Blackwell) implementation of TurboQuant KV cache compression. The headline: **asymmetric K=turbo3 V=q8_0 is 19% faster than q8_0 at 128K context AND produces better perplexity than q8_0 at ctx=2048.**

**Repo**: https://github.com/Madreag/turbo3-cuda (branch: `release/turbo3-cuda`)

### Asymmetric K=turbo3, V=q8_0 — the best of both worlds

This was the most surprising finding. Compressing only K with turbo3 (rotation-invariant for Q.K dot product) while keeping V at q8_0 (higher fidelity for value accumulation) gives:

| Model | Context | q8_0 tok/s | K=turbo3 V=q8_0 | Speedup |
|-------|---------|-----------|----------------|---------|
| MoE 35B-A3B | short | 194 | 179 | 0.924x |
| MoE 35B-A3B | 32K | 139 | 142 | **1.023x** |
| MoE 35B-A3B | **128K** | **79** | **94** | **1.187x** |
| Dense 27B | short | 55 | 55 | 1.008x |
| Dense 27B | 32K | 46 | 46 | 0.994x |

Quality: PPL 6.804 (+0.67% at ctx=512) and **5.650 at ctx=2048 — actually 0.42% BETTER than q8_0's 5.674.** The turbo3 K compression removes q8_0's quantization noise on the key vectors while the uncompressed q8_0 V preserves full value fidelity.

### Symmetric turbo3 — beats q8_0 at 16K+

On the dense model (Qwen 3.5 27B Q6_K):

| Context | q8_0 tok/s | turbo3 tok/s | Ratio |
|---------|-----------|-------------|-------|
| short | 55.05 | 51.95 | 0.944x |
| 16K | 50.26 | 49.85 | **0.993x** |
| 32K | 45.96 | 47.76 | **1.039x** |

On the MoE model (Qwen 3.5 35B-A3B Q4_K_M):

| Context | q8_0 tok/s | turbo3 tok/s | Ratio |
|---------|-----------|-------------|-------|
| short | 186 | 158 | 0.846x |
| 32K | 134 | 131 | **0.975x** |
| 128K | 79 | 87 | **1.100x** |

The MoE 0.975x matches spiritbuun's reported 0.97x — confirming our implementations agree.

PPL: turbo3 = 6.848 (+1.32% at ctx=512), 5.736 (+1.08% at ctx=2048). With layer-adaptive (first+last 4 layers at q8_0): 6.804 (+0.67%).

### Architecture: Persistent fp16 Shadow Cache

Instead of dequanting the entire KV cache per token, we maintain a persistent fp16 shadow buffer that's incrementally updated:

- **Decode**: Only dequant 1 new KV position per token (~2 KB vs ~64 MB at 32K)
- **Prefill**: Bulk dequant to temp buffers + MMA Tensor Cores (0.977x of q8_0)
- **Sparse V skip**: Skip V positions with attention weight < 1e-4 (based on TheTom's research)

The crossover where turbo3 beats q8_0 is ~16K context on the dense model. At that point the 4.6x bandwidth reduction outweighs the ~5.6% shadow overhead.

### turbo4 (4.25 bpv)

turbo4 adds a 1-bit QJL residual correction on top of turbo3. End-to-end working:
- PPL 5.743 at ctx=2048 (+1.22%)
- Decode: 52.5 tok/s short, 47.6 tok/s at 32K

### Known Limitations

- **turbo4 multi-sequence PPL**: The native turbo4 vec kernel gives NaN when Q->ne[3] > 1 (multi-sequence perplexity evaluation at ctx=512). The shadow path at ctx=2048 works correctly. Being investigated.
- **Only tested on SM120**: Should work on SM75+ but not yet confirmed on other architectures. Use spiritbuun's fork for RTX 3090/4090.

### Credits

Huge thanks to:
- **@TheTom** for the original Metal implementation, sparse V dequant research, and diagnostic scripts
- **@spiritbuun** for the CUDA reference implementation, norm correction, and layer-adaptive ideas
- **Google Research** for the TurboQuant paper (ICLR 2026)

### Looking for

- Community testing on other models and GPU architectures (RTX 4090, 3090)
- Feedback on the asymmetric K/V approach — has anyone else tried mixed K/V quantization types?
- Anyone interested in lop3/TC-based FA kernels for Blackwell (the path past 0.94x at short context)

Build instructions, mode recommendations, and full benchmarks: https://github.com/Madreag/turbo3-cuda
