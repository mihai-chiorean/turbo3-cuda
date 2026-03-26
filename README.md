# llama.cpp + TurboQuant turbo3 CUDA

The first CUDA implementation of TurboQuant turbo3 KV cache compression with Flash Attention support for NVIDIA GPUs.

Compresses the KV cache to **3.5 bits per value** (4.6× vs fp16) with near-zero quality loss, enabling **700K+ token context on a single RTX 5090** (32GB).

## What This Is

A fork of [TheTom/llama-cpp-turboquant](https://github.com/TheTom/llama-cpp-turboquant) (which itself forks [ggml-org/llama.cpp](https://github.com/ggml-org/llama.cpp)) that adds **CUDA GPU support** for the `turbo3` KV cache type. TheTom's original implementation had Metal (Apple Silicon) kernels only — this port brings full CUDA support including Flash Attention integration.

## Key Results

| Metric | Value |
|--------|-------|
| KV compression | 4.6× (3.5 bits/value vs 16-bit fp16) |
| Max context (RTX 5090 32GB) | 700,000+ tokens |
| Generation speed | 48 tok/s at 524K context |
| Quality (NIAH) | 6/6 exact retrieval |
| Quality (math/factual) | All tests passing |
| Flash Attention | Enabled for both K and V |

## How It Works

TurboQuant ([arXiv:2504.19874](https://arxiv.org/abs/2504.19874), ICLR 2026) compresses KV cache vectors using:

1. **Random orthogonal rotation** (Fast Walsh-Hadamard Transform) — makes coordinates nearly independent
2. **Lloyd-Max optimal scalar quantization** — 3-bit per coordinate using precomputed codebooks
3. **Per-vector norm storage** — preserves magnitude information in fp16

The `turbo3` block format stores 32 values in 14 bytes: 8 bytes of 2-bit indices + 4 bytes of 1-bit sign extensions + 2 bytes fp16 norm.

## Requirements

- NVIDIA GPU with compute capability ≥ 8.0 (tested on RTX 5090 sm_120)
- CUDA Toolkit 12.8 (NOT 13.x — MMQ kernel segfaults on 13.1)
- CMake 3.21+
- C++17 compiler

## Quick Start

```bash
# Clone
git clone https://github.com/erolgermain/llama-cpp-turbo3-cuda.git
cd llama-cpp-turbo3-cuda

# Build
cmake -B build -DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=120 \
  -DGGML_CUDA_FORCE_CUBLAS=OFF
cmake --build build --config Release -j $(nproc)

# Run with turbo3 KV cache
./build/bin/llama-server \
  -m your-model.gguf \
  --cache-type-k turbo3 --cache-type-v turbo3 \
  -c 524288 --port 8080 --host 0.0.0.0 -ngl 99 --no-mmap \
  --jinja --reasoning-format deepseek
```

Or use the launch script:

```bash
# Set model path and run
TURBO_MODEL_PATH=./models/your-model.gguf ./launch_server.sh

# Or pass as argument
./launch_server.sh ./models/your-model.gguf 524288
```

## VRAM Budget

Measured on Qwen3-32B Q6_K (~21 GB weights), RTX 5090 32GB:

| Component | Size |
|-----------|------|
| Model weights (Q6_K) | ~21 GB |
| KV cache turbo3 (per 100K tokens) | ~1.4 GB |
| CUDA overhead | ~1.5 GB |
| **Total at 524K context** | **~29.8 GB** |
| **Total at 700K context** | **~32.3 GB** |

**VRAM formula:**

```
KV_per_token = 2 × n_layers × n_kv_heads × head_dim × (14 / 32)
```

The per-token cost depends on your model's architecture (`n_layers`, `n_kv_heads`, `head_dim`). For the Qwen3-32B tested here, measured KV consumption is **~14 KB/token** with turbo3, vs ~64 KB/token with fp16 — a 4.6× reduction.

## Files Modified (CUDA Port)

**New files:**
| File | Description |
|------|-------------|
| `ggml/src/ggml-cuda/turbo-quant.cu` | Dequantize kernels, FWHT rotation kernel, SET_ROWS quantize kernel |
| `ggml/src/ggml-cuda/turbo-quant.cuh` | Header declarations for turbo3 CUDA ops |
| `ggml/src/ggml-cuda/template-instances/fattn-vec-instance-turbo3_0-f16.cu` | FA template instances: turbo3 K + fp16 V |
| `ggml/src/ggml-cuda/template-instances/fattn-vec-instance-turbo3_0-turbo3_0.cu` | FA template instances: turbo3 K + turbo3 V |

**Modified files:**
| File | Change |
|------|--------|
| `fattn-common.cuh` | `vec_dot_fattn_vec_KQ_turbo3_0` (K dot product) + `dequantize_V_turbo3_0` (V dequant) |
| `fattn-vec.cuh` | Thread/block config for turbo3 (matches f16 path), extern declarations |
| `fattn.cu` | FATTN_VEC_CASES_ALL_D entries, K type switch, K≠V mixed-type exception |
| `convert.cu` | Dequantize dispatch for fp16/fp32/bf16, contiguous + non-contiguous |
| `set-rows.cu` | Custom turbo3 quantize kernel dispatch |
| `getrows.cu` | turbo3 get_rows dispatch |
| `cpy.cu` | Same-type raw byte copy for turbo3/turbo4 |
| `cpy-utils.cuh` | `quantize_f32_turbo3_0_block` device function |
| `dequantize.cuh` | `dequantize_turbo3_0` for get_rows template |
| `ggml-cuda.cu` | TURBO_WHT op dispatch, supports_op entries, MUL_MAT turbo exclusion |
| `CMakeLists.txt` | Glob turbo3 FA template instances into build |

## Known Limitations

- **CUDA 13.1 + MMQ kernel = segfault** — use CUDA 12.8
- **turbo4** (4.25-bit with QJL correction) not yet ported to CUDA
- **No CPU fallback** for turbo3 dequant during inference (GPU required)
- **Flash Attention required** for turbo3 V cache (non-FA path would materialize O(n²) attention matrix)
- **Blackwell-tested only** — should work on Ampere/Hopper but not validated

## Benchmarks

See [BENCHMARKS.md](BENCHMARKS.md) for detailed quality and performance results with reproduction commands.

## Documentation

- **[GUIDE.md](GUIDE.md)** — Step-by-step setup, build, and deployment guide
- **[ARCHITECTURE.md](ARCHITECTURE.md)** — Technical deep-dive into the turbo3 block format, FWHT rotation, and CUDA kernel design
- **[BENCHMARKS.md](BENCHMARKS.md)** — Quality and performance benchmarks with reproduction commands

## Credits

This project builds on the work of:

- **[TheTom/turboquant_plus](https://github.com/TheTom/turboquant_plus)** — Original TurboQuant implementation with Metal kernels, llama.cpp integration, turbo3/turbo4 block formats, WHT rotation, and the complete test suite. The CUDA kernels in this repo are a direct port of TheTom's Metal shaders.
- **[ggml-org/llama.cpp](https://github.com/ggml-org/llama.cpp)** — The inference engine this is built on.
- **[Google Research](https://research.google/blog/turboquant-redefining-ai-efficiency-with-extreme-compression/)** — The TurboQuant algorithm (Zandieh, Daliri, Hadian, Mirrokni). Paper: [arXiv:2504.19874](https://arxiv.org/abs/2504.19874).
- **[tonbistudio/turboquant-pytorch](https://github.com/tonbistudio/turboquant-pytorch)** — PyTorch reference implementation with Lloyd-Max solver and test suite.
- **[dejan.ai](https://dejan.ai/blog/turboquant/)** — Triton fused kernel implementation that informed the Flash Attention integration approach.
- **[unixsysdev/llama-turboquant](https://github.com/unixsysdev/llama-turboquant)** — Independent CUDA port with fused MMVQ approach (referenced during development).
- **[Prince Canuma (@Blaizzy)](https://github.com/Blaizzy)** — MLX TurboQuant implementation validating 6/6 NIAH on Qwen3.5 architecture.

## Algorithm Reference

- TurboQuant: [arXiv:2504.19874](https://arxiv.org/abs/2504.19874) (ICLR 2026)
- PolarQuant (Stage 1): [arXiv:2502.02617](https://arxiv.org/abs/2502.02617) (AISTATS 2026)
- QJL: [arXiv:2406.03482](https://arxiv.org/abs/2406.03482)

## License

MIT (matching upstream llama.cpp)
