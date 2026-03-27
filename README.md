# llama.cpp + TurboQuant CUDA — Faster Than q8_0 at Long Context

The first production CUDA implementation of TurboQuant KV cache compression for NVIDIA GPUs. **turbo3 now beats q8_0 decode speed at 16K+ context** while using 4.6× less KV memory.

## Headline Results (RTX 5090, Qwen 3.5 27B)

| Context | q8_0 tok/s | turbo3 tok/s | Ratio | KV Memory |
|---------|-----------|-------------|-------|-----------|
| short | 58.60 | 55.18 | 0.942× | 4.6× smaller |
| 8K | 53.66 | 51.04 | 0.951× | 4.6× smaller |
| 16K | 50.26 | 49.85 | **0.992×** | 4.6× smaller |
| 32K | 45.29 | 48.15 | **1.063×** | 4.6× smaller |
| Prefill | 3001 | 2931 | 0.977× | — |

| Quality | q8_0 | turbo3 | turbo3 + LA-1 |
|---------|------|--------|---------------|
| PPL (512 ctx) | 6.759 | 6.854 (+1.4%) | 6.804 (+0.67%) |
| PPL (2048 ctx) | 5.674 | 5.736 (+1.1%) | — |

**turbo3 beats q8_0 by 6.3% at 32K context** — compressed KV moves less data over the memory bus, and sparse V dequant skips 90%+ of negligible attention positions.

## What Makes This Fork Unique

- **Persistent fp16 shadow cache** — KV data dequanted once, cached across tokens. Only new positions (1 per token) need dequanting. Zero per-token overhead for existing cache.
- **Sparse V dequantization** — Skip V dequant+accumulate for positions with attention weight < 1e-6. Eliminates 90%+ of V-path work at long context. Based on [TheTom's research](https://github.com/TheTom/turboquant_plus/blob/main/docs/papers/sparse-v-dequant.md).
- **Layer-adaptive KV cache** — Promote quality-sensitive layers (first+last) to q8_0. Cuts PPL gap from +1.4% to +0.67%.
- **Blackwell-first** — Tested and optimized for RTX 5090 (SM120, GDDR7).
- **Prefill dequant+MMA** — Tensor Core acceleration for prompt processing via bulk fp16 dequant.

## Hardware

- **Tested**: RTX 5090 32GB, CUDA 12.8, SM120 (Blackwell)
- **Should work**: Any NVIDIA GPU with SM ≥ 75 (Turing+) and Flash Attention support
- **Not tested**: RTX 3090/4090 (use [spiritbuun's fork](https://github.com/spiritbuun/llama-cpp-turboquant-cuda) for those)

## Quick Start

```bash
# Build
cmake -B build -DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=120
cmake --build build -j$(nproc)

# Run with turbo3 KV cache (Flash Attention auto-enabled)
./build/bin/llama-cli -hf your-model-GGUF -ctk turbo3 -ctv turbo3 -ngl 99

# Run server
./build/bin/llama-server -hf your-model-GGUF -ctk turbo3 -ctv turbo3 -ngl 99 --port 8080
```

## Configuration

| Environment Variable | Effect | Default |
|---------------------|--------|---------|
| `TURBO_LAYER_ADAPTIVE=1` | First+last 4 layers at q8_0 (best PPL) | 0 (disabled) |
| `TURBO_LAYER_ADAPTIVE=2` | Last 8 layers at q8_0 | — |
| `GGML_TURBO_DECODE_NATIVE=1` | Disable fp16 shadow, use native turbo3 vec | disabled |

Layer-adaptive modes (set `TURBO_LAYER_ADAPTIVE`):
- `0` — Uniform turbo3 (default)
- `1` — q8_0 for first 4 + last 4 layers (best PPL: +0.67%)
- `2` — q8_0 for last 8 layers
- `3` — q8_0 for last 4 layers
- `4` — q8_0 for first 4 layers
- `5` — q8_0 for first 2 + last 2 layers

## How It Works

[TurboQuant](https://arxiv.org/abs/2504.19874) (ICLR 2026) compresses KV cache vectors to 3.5 bits/value:

1. **Normalize** — compute L2 norm of 128-element groups
2. **Rotate** — Fast Walsh-Hadamard Transform with random sign flips (Gaussianizes the distribution)
3. **Quantize** — nearest of 8 Lloyd-Max centroids (3-bit index)
4. **Pack** — 2-bit qs[] + 1-bit signs[] arrays + fp16 norm
5. **Norm correction** — store `raw_norm / ||reconstruction||` for exact L2 norm

Block format: 16 bytes per 32 values (padded for GDDR7 coalescing).

### Decode Architecture

```
Token arrives → SET_ROWS stores turbo3 in KV cache
                → Shadow cache: incremental dequant of 1 new position to fp16
                                (positions 0..N-1 already cached from previous tokens)
Flash Attention → K path: read fp16 from shadow (fast dot product)
                → V path: sparse skip for weight < 1e-6 (90%+ skipped at long ctx)
Output → graph-level V inverse rotation
```

## Credits

- **[TheTom](https://github.com/TheTom/llama-cpp-turboquant)** — Original Metal implementation, sparse V dequant research, diagnostic scripts
- **[spiritbuun](https://github.com/spiritbuun/llama-cpp-turboquant-cuda)** — CUDA reference (RTX 3090), norm correction, layer-adaptive, prefill MMA
- **[Google Research](https://arxiv.org/abs/2504.19874)** — TurboQuant algorithm (ICLR 2026)
- **[Madreag](https://github.com/Madreag)** — This fork: RTX 5090 port, persistent shadow cache, sparse V CUDA, Blackwell optimization

## Related Projects

- [TheTom/turboquant_plus](https://github.com/TheTom/turboquant_plus) — Documentation, benchmarks, diagnostic scripts
- [spiritbuun/llama-cpp-turboquant-cuda](https://github.com/spiritbuun/llama-cpp-turboquant-cuda) — CUDA port for RTX 3090
- [ggml-org/llama.cpp Discussion #20969](https://github.com/ggml-org/llama.cpp/discussions/20969) — Main TurboQuant discussion

## License

MIT (matching upstream llama.cpp)
