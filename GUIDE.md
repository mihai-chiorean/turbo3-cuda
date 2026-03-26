# Setup Guide — turbo3 CUDA for llama.cpp

Step-by-step guide for building, running, and testing the turbo3 CUDA port.

## 1. Prerequisites

### Hardware
- NVIDIA GPU with compute capability ≥ 8.0 (Ampere, Hopper, Blackwell)
- Minimum 16 GB VRAM for small models, 32 GB recommended for long-context
- Tested on: RTX 5090 (sm_120, 32 GB)

### Software
- **CUDA Toolkit 12.8** — **Not** 13.x (MMQ kernel segfaults on CUDA 13.1)
- CMake 3.21+
- GCC 11+ or Clang 14+ (C++17 required)
- Git

### Verify CUDA installation
```bash
nvcc --version        # Should show 12.8.x
nvidia-smi            # Should show your GPU
```

If you have multiple CUDA versions installed:
```bash
export PATH=/usr/local/cuda-12.8/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda-12.8/lib64:$LD_LIBRARY_PATH
```

## 2. Building from Source

```bash
git clone https://github.com/erolgermain/llama-cpp-turbo3-cuda.git
cd llama-cpp-turbo3-cuda

cmake -B build \
  -DGGML_CUDA=ON \
  -DCMAKE_CUDA_ARCHITECTURES=120 \
  -DGGML_CUDA_FORCE_CUBLAS=OFF

cmake --build build --config Release -j $(nproc)
```

### Build flags explained

| Flag | Purpose |
|------|---------|
| `-DGGML_CUDA=ON` | Enable CUDA backend |
| `-DCMAKE_CUDA_ARCHITECTURES=120` | Target SM architecture (120=Blackwell, 89=Ada, 80=Ampere) |
| `-DGGML_CUDA_FORCE_CUBLAS=OFF` | Use custom CUDA kernels instead of cuBLAS for everything |
| `-DCUDAToolkit_ROOT=/usr/local/cuda-12.8` | Optional: specify CUDA path if not on default PATH |

### Architecture values
| GPU Family | SM Value |
|-----------|----------|
| Ampere (A100, RTX 3090) | 80, 86 |
| Hopper (H100) | 90 |
| Ada Lovelace (RTX 4090) | 89 |
| Blackwell (RTX 5090) | 120 |

Set `CMAKE_CUDA_ARCHITECTURES` to match your GPU. You can specify multiple: `-DCMAKE_CUDA_ARCHITECTURES="89;120"`.

### Verify build
```bash
./build/bin/llama-server --help | grep cache-type
# Should list: turbo3, turbo4, q8_0, q4_0, f16, etc.
```

## 3. Downloading a Model

You need a GGUF model file. Recommended: **Qwen3-32B** in Q6_K quantization for best quality/VRAM balance.

```bash
# Install huggingface-cli if needed
pip install huggingface-hub

# Download a Q6_K model (adjust for your preferred model)
mkdir -p models
huggingface-cli download bartowski/Qwen3-32B-GGUF \
  Qwen3-32B-Q6_K.gguf \
  --local-dir models/
```

### Model size vs VRAM budget
| Quantization | ~Size (32B params) | Remaining VRAM for KV (32GB GPU) |
|-------------|-------------------|----------------------------------|
| Q4_K_M | ~18 GB | ~12.5 GB → ~890K tokens turbo3 |
| Q6_K | ~21 GB | ~9.5 GB → ~680K tokens turbo3 |
| Q8_0 | ~32 GB | ~0 GB (won't fit with long context) |

## 4. Running the Server

### Using the launch script (recommended)
```bash
# Basic usage
./launch_server.sh ./models/Qwen3-32B-Q6_K.gguf

# With custom context
./launch_server.sh ./models/Qwen3-32B-Q6_K.gguf 262144

# Using environment variables
export TURBO_MODEL_PATH=./models/Qwen3-32B-Q6_K.gguf
export TURBO_CONTEXT=524288
export TURBO_PORT=8080
./launch_server.sh
```

### Direct invocation
```bash
./build/bin/llama-server \
  -m ./models/Qwen3-32B-Q6_K.gguf \
  --cache-type-k turbo3 \
  --cache-type-v turbo3 \
  -c 524288 \
  --port 8080 \
  --host 0.0.0.0 \
  -ngl 99 \
  --no-mmap \
  --jinja \
  --reasoning-format deepseek
```

### Server flags explained
| Flag | Purpose |
|------|---------|
| `--cache-type-k turbo3` | Use turbo3 compression for K cache |
| `--cache-type-v turbo3` | Use turbo3 compression for V cache |
| `-c 524288` | Context window size in tokens |
| `-ngl 99` | Offload all layers to GPU |
| `--no-mmap` | Don't memory-map model (avoids page faults on long runs) |
| `--jinja` | Enable Jinja2 chat templates |
| `--reasoning-format deepseek` | Parse `<think>` blocks in reasoning models |

### Health check
```bash
curl http://localhost:8080/health
# Should return: {"status":"ok"}
```

## 5. Connecting to Open WebUI / oh-my-pi

Add to your `models.yml`:
```yaml
turbo3-qwen32b:
  api_base: http://YOUR_SERVER_IP:8080/v1
  api_key: ""
  model: qwen3-32b
  context_length: 524288
```

## 6. Connecting to Claude Code (claude-code-proxy)

In your claude-code-proxy `config.yml`:
```yaml
providers:
  turbo3:
    type: openai
    base_url: http://YOUR_SERVER_IP:8080/v1
    api_key: "not-needed"
    model: qwen3-32b
```

## 7. Testing

### Quick quality test
```bash
# Math test
curl -s http://localhost:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"messages":[{"role":"user","content":"What is 15*23+7? Just the number."}],"max_tokens":300,"temperature":0}' \
  | python3 -m json.tool

# Expected: 352

# Factual test
curl -s http://localhost:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"messages":[{"role":"user","content":"Capital of France? One word."}],"max_tokens":200,"temperature":0}' \
  | python3 -m json.tool

# Expected: Paris
```

### NIAH (Needle in a Haystack)
Generate a long context with a hidden fact, then ask about it:
```bash
# Create a test with 4K context
python3 -c "
import json
filler = 'The quick brown fox jumps over the lazy dog. ' * 200
needle = 'The secret code is TURBO3WORKS. '
prompt = filler + needle + filler
msg = {'messages': [
    {'role': 'user', 'content': prompt + '\n\nWhat is the secret code?'}
], 'max_tokens': 50, 'temperature': 0}
print(json.dumps(msg))
" | curl -s http://localhost:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d @-
```

### Server metrics
```bash
curl -s http://localhost:8080/metrics
# Shows tokens/second, context usage, VRAM usage
```

## 8. Troubleshooting

### CUDA 13.1 segfault
**Symptom:** Server crashes immediately or on first inference with `SIGSEGV` inside MMQ kernels.
**Fix:** Downgrade to CUDA 12.8. The MMQ (matrix multiply quantized) kernels have a bug in CUDA 13.1.

### Out of memory (OOM)
**Symptom:** `CUDA error: out of memory`
**Fix:** Reduce context size. Use the VRAM formula:
```
Available_KV = GPU_VRAM - Model_Size - 1.5GB_overhead
Max_tokens = Available_KV / (2 × n_layers × n_kv_heads × head_dim × 14/32)
```

### Flash Attention not used
**Symptom:** Extremely slow inference, high VRAM usage.
**Fix:** Flash Attention is required for turbo3 V cache. It should be auto-selected. Check that you're NOT passing `--no-flash-attn`.

### Model doesn't load
**Symptom:** `error: failed to load model`
**Fix:** Ensure:
1. The `.gguf` file is not corrupted (re-download if needed)
2. You have enough VRAM for the model weights alone
3. `-ngl 99` is set (partial offload can cause issues)

### Build fails: "turbo3 not found"
**Symptom:** Compilation errors about undefined `GGML_TYPE_TURBO3_0`
**Fix:** Make sure you're building from this fork, not upstream llama.cpp. The turbo3 type definitions are in TheTom's fork.

### Slow generation at high context
**Symptom:** Generation speed drops below 10 tok/s.
**Likely cause:** KV cache is consuming most of VRAM, causing memory pressure. Reduce `-c` or use a smaller model quantization (Q4_K_M instead of Q6_K).

## 9. Performance Tuning

### Choosing context size based on VRAM
1. Check available VRAM: `nvidia-smi`
2. Subtract model weight size (see quantization table above)
3. Subtract 1.5 GB for CUDA overhead
4. Divide remaining by per-token KV cost

### Q4_K_M vs Q6_K tradeoffs
| | Q4_K_M | Q6_K |
|---|--------|------|
| Model quality | Good | Better |
| Model size (32B) | ~18 GB | ~21 GB |
| Max turbo3 context (32GB GPU) | ~890K | ~680K |
| Best for | Maximum context | Best quality |

### Multi-GPU
Not yet tested. llama.cpp supports tensor splitting across GPUs — should work with turbo3 but is unvalidated.

## 10. VRAM Calculator

```
KV_per_token = 2 × n_layers × n_kv_heads × head_dim × (14/32)
             = 2 × n_layers × n_kv_heads × head_dim × 0.4375 bytes

Available_KV = GPU_VRAM - Model_Size - 1.5 GB
Max_tokens   = Available_KV / KV_per_token
```

### Examples

**Qwen3-32B on RTX 5090 (32GB):**
- Q6_K weights: 21 GB
- Available for KV: 32 - 21 - 1.5 = 9.5 GB
- Per token: 2 × 64 × 8 × 128 × 0.4375 = 57,344 bytes ≈ 56 KB
- Max context: 9.5 GB / 56 KB ≈ **~170K tokens** per KV, but turbo3 compresses to ~14 KB/token → **~680K tokens**

**Llama-3.1-70B on 2× RTX 5090 (64GB):**
- Q4_K_M weights: ~38 GB
- Available for KV: 64 - 38 - 3 = 23 GB
- 80 layers, 8 KV heads, 128 head_dim
- Max context: ~1.6M tokens (theoretical)
