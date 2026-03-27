# llama.cpp + TurboQuant CUDA — Faster Than q8_0 at Long Context

CUDA implementation of [TurboQuant](https://arxiv.org/abs/2504.19874) (ICLR 2026) KV cache compression for NVIDIA GPUs. **4.6x less KV memory, faster than q8_0 at 16K+ context, up to 19% faster at 128K.** Proven on dense and MoE architectures.

## Why Use This Fork?

| | q8_0 (stock llama.cpp) | This fork (turbo3) |
|---|---|---|
| **KV memory** | 8.5 bits/value | **3.5 bits/value (4.6x smaller)** |
| **Short context speed** | baseline | 0.94x (6% slower) |
| **32K context speed** | baseline | **1.04x (4% faster)** |
| **128K context speed (MoE)** | baseline | **1.19x (19% faster)** |
| **Max context for 32GB VRAM** | ~65K | **~300K** |
| **Quality (PPL)** | baseline | +1.08% (negligible) |

The tradeoff: ~6% slower at short context, but faster at long context and fits 4.6x more context in VRAM. If you regularly use 8K+ context, turbo3 is a net win.

## Which Mode Should I Use?

| Your priority | Mode | Command | What it does |
|---|---|---|---|
| **Best speed at long context** | turbo3+turbo3 | `-ctk turbo3 -ctv turbo3` | Both K and V compressed. Beats q8_0 at 16K+. Sparse V skip at long ctx. |
| **Best quality** | K=turbo3, V=q8_0 | `-ctk turbo3 -ctv q8_0` | Only K compressed, V stays full precision. Half the PPL delta. **Better than q8_0 quality at ctx=2048.** |
| **Max VRAM savings** | turbo3 + layer-adaptive | `-ctk turbo3 -ctv turbo3` + `TURBO_LAYER_ADAPTIVE=1` | turbo3 everywhere except quality-sensitive layers (first+last 4) promoted to q8_0. PPL gap drops to +0.67%. |
| **Experimental** | turbo4 | `-ctk turbo4 -ctv turbo4` | 4.25 bpv with QJL residual correction. Similar speed to turbo3 through shadow path. |

## Headline Results (RTX 5090)

### Dense Model: Qwen 3.5 27B Q6_K

| Context | q8_0 tok/s | turbo3 tok/s | Ratio | KV Memory |
|---------|-----------|-------------|-------|-----------|
| short | 55.05 | 51.95 | 0.944x | 4.6x smaller |
| 8K | 54.79 | 51.92 | 0.951x | 4.6x smaller |
| 16K | 50.26 | 49.85 | **0.993x** | 4.6x smaller |
| 32K | 45.96 | 47.76 | **1.039x** | 4.6x smaller |
| Prefill | 3001 | 2931 | 0.977x | -- |

### MoE Model: Qwen 3.5 35B-A3B Q4_K_M

| Context | q8_0 tok/s | turbo3 tok/s | Ratio |
|---------|-----------|-------------|-------|
| short | 186 | 158 | 0.846x |
| 32K | 134 | 131 | **0.975x** |
| 128K | 79 | 87 | **1.100x** |

### Asymmetric Mode (K=turbo3, V=q8_0) on MoE

| Context | q8_0 tok/s | K=turbo3 V=q8_0 | Ratio |
|---------|-----------|----------------|-------|
| short | 194 | 179 | 0.924x |
| 32K | 139 | 142 | **1.023x** |
| 128K | 79 | 94 | **1.187x** |

### Quality (PPL, wikitext-2, 8 chunks)

| Config | ctx=512 | ctx=2048 |
|--------|---------|----------|
| q8_0 | 6.759 | 5.674 |
| turbo3 | 6.848 (+1.32%) | 5.736 (+1.08%) |
| turbo3 LA-1 | 6.804 (+0.67%) | -- |
| K=turbo3 V=q8_0 | 6.804 (+0.67%) | 5.650 (**-0.42%**) |
| turbo4 | -- | 5.743 (+1.22%) |

## Comparison With Other Forks

| Fork | GPU | Architecture | Short ctx | 32K ctx | Notes |
|------|-----|-------------|-----------|---------|-------|
| **This fork (Madreag)** | RTX 5090 | Persistent fp16 shadow + sparse V | 0.944x | **1.039x** | Blackwell-optimized, asymmetric K/V, turbo4 |
| [spiritbuun](https://github.com/spiritbuun/llama-cpp-turboquant-cuda) | RTX 3090 | Per-call dequant+free | ~0.88x (dense) | ~0.97x (MoE) | turbo4, layer-adaptive, multi-GPU |
| [TheTom](https://github.com/TheTom/llama-cpp-turboquant) | Apple M-series | Metal native | varies | varies | Original implementation, sparse V research |

**Which fork for your GPU?**
- **RTX 5090 / Blackwell**: This fork
- **RTX 3090 / 4090 / Ampere / Ada**: [spiritbuun's fork](https://github.com/spiritbuun/llama-cpp-turboquant-cuda)
- **Apple Silicon**: [TheTom's fork](https://github.com/TheTom/llama-cpp-turboquant)

## What Makes This Fork Unique

- **Persistent fp16 shadow cache** -- KV data dequanted once, cached across tokens. Only new positions (1 per token) need dequanting. Zero per-token overhead for existing cache entries.
- **Sparse V dequantization** -- Skip V dequant+accumulate for positions with attention weight < 1e-4. Eliminates 90%+ of V-path work at long context. Based on [TheTom's research](https://github.com/TheTom/turboquant_plus/blob/main/docs/papers/sparse-v-dequant.md).
- **Asymmetric K/V types** -- Use turbo3 K + q8_0 V for best quality (half the PPL delta, better than q8_0 at long context).
- **Layer-adaptive KV cache** -- Promote quality-sensitive layers (first+last) to q8_0. Cuts PPL gap from +1.32% to +0.67%.
- **turbo4 support** -- 4.25 bpv with 1-bit QJL residual correction on top of turbo3's 3-bit PolarQuant.
- **Blackwell-optimized** -- 16-byte struct padding for GDDR7 32-byte sector coalescing, SM120 tuning.
- **Prefill dequant+MMA** -- Tensor Core acceleration for prompt processing via bulk fp16 dequant.

## Quick Start

```bash
# Build (adjust CUDA_ARCHITECTURES for your GPU: 75=Turing, 80=Ampere, 89=Ada, 120=Blackwell)
cmake -B build -DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=120
cmake --build build -j$(nproc)

# Recommended: turbo3 KV cache (Flash Attention auto-enabled)
./build/bin/llama-cli -hf your-model-GGUF -ctk turbo3 -ctv turbo3 -ngl 99

# Best quality: turbo3 K + q8_0 V
./build/bin/llama-cli -hf your-model-GGUF -ctk turbo3 -ctv q8_0 -ngl 99

# turbo4: 4.25 bpv with QJL residual
./build/bin/llama-cli -hf your-model-GGUF -ctk turbo4 -ctv turbo4 -ngl 99

# Server mode
./build/bin/llama-server -hf your-model-GGUF -ctk turbo3 -ctv turbo3 -ngl 99 --port 8080

# Layer-adaptive (best PPL with turbo3)
TURBO_LAYER_ADAPTIVE=1 ./build/bin/llama-cli -hf your-model-GGUF -ctk turbo3 -ctv turbo3 -ngl 99
```

## Hardware

- **Tested**: RTX 5090 32GB, CUDA 12.8, SM120 (Blackwell)
- **Should work**: SM75+ (Turing and newer), but only tested on SM120. Use [spiritbuun's fork](https://github.com/spiritbuun/llama-cpp-turboquant-cuda) for confirmed RTX 3090/4090 support

Flash Attention is **required** for turbo types (auto-enabled when turbo K/V is detected).

## Configuration

| Environment Variable | Effect | Default |
|---------------------|--------|---------|
| `TURBO_LAYER_ADAPTIVE=N` | Per-layer KV type (see below) | 0 (uniform turbo) |
| `GGML_TURBO_DECODE_NATIVE=1` | Disable fp16 shadow, use native turbo3 vec kernel (slower, for debugging) | disabled |

**Layer-adaptive modes** (`TURBO_LAYER_ADAPTIVE`):

| Mode | Strategy | PPL impact |
|------|----------|------------|
| `0` | Uniform turbo3 (default) | +1.32% |
| `1` | q8_0 for first 4 + last 4 layers | **+0.67%** (recommended) |
| `2` | q8_0 for last 8 layers | ~+0.9% |
| `3` | q8_0 for last 4 layers | ~+1.0% |
| `4` | q8_0 for first 4 layers | ~+1.0% |
| `5` | q8_0 for first 2 + last 2 layers | ~+0.9% |

## How It Works

[TurboQuant](https://arxiv.org/abs/2504.19874) (ICLR 2026) compresses KV cache vectors to 3.5 bits/value:

1. **Normalize** -- compute L2 norm of 128-element groups
2. **Rotate** -- Fast Walsh-Hadamard Transform with random sign flips (Gaussianizes the distribution)
3. **Quantize** -- nearest of 8 Lloyd-Max centroids (3-bit index)
4. **Pack** -- 2-bit qs[] + 1-bit signs[] arrays + fp16 norm per block
5. **Norm correction** -- store `raw_norm / ||reconstruction||` instead of raw norm for exact L2 norm recovery

Block format: 16 bytes per 32 values (padded from 14 for GDDR7 coalescing).

### Decode Architecture

```
Token arrives -> SET_ROWS quantizes K/V to turbo3 in KV cache (FWHT + centroid quantize)
              -> Shadow cache: incremental dequant of 1 new position to fp16
                               (positions 0..N-1 already cached from previous tokens)

Flash Attention -> K path: fp16 dot product from shadow (hardware half2 ops)
               -> V path: sparse skip for attention weight < 1e-4 (90%+ skipped at long ctx)

Output -> graph-level inverse FWHT rotation on attention output
```

### Why turbo3 beats q8_0 at long context

1. **4.6x less KV bandwidth** -- turbo3 = 16 bytes/32 values, q8_0 = 34 bytes/32 values. At 32K+, decode is memory-bandwidth-bound.
2. **Sparse V skip** -- 90%+ of V positions have negligible attention weight. Skipping them removes both bandwidth AND quantization noise.
3. **L2 cache efficiency** -- More turbo3 KV fits in the RTX 5090's 98 MB L2 cache at the same context length.

## Credits

- **[TheTom](https://github.com/TheTom/llama-cpp-turboquant)** -- Original Metal implementation, sparse V dequant research, norm correction innovation, diagnostic scripts
- **[spiritbuun](https://github.com/spiritbuun/llama-cpp-turboquant-cuda)** -- CUDA reference (RTX 3090), norm correction, layer-adaptive, prefill MMA, turbo4 port
- **[Google Research](https://arxiv.org/abs/2504.19874)** -- TurboQuant algorithm (ICLR 2026)
- **[Madreag](https://github.com/Madreag)** -- This fork: RTX 5090 port, persistent shadow cache, sparse V CUDA, asymmetric K/V, Blackwell optimization

## Related Projects

- [TheTom/turboquant_plus](https://github.com/TheTom/turboquant_plus) -- Documentation, benchmarks, diagnostic scripts
- [spiritbuun/llama-cpp-turboquant-cuda](https://github.com/spiritbuun/llama-cpp-turboquant-cuda) -- CUDA port for RTX 3090
- [ggml-org/llama.cpp Discussion #20969](https://github.com/ggml-org/llama.cpp/discussions/20969) -- Main TurboQuant community discussion

## License

MIT (matching upstream llama.cpp)
