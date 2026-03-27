---
description: 
alwaysApply: true
---

# CLAUDE.md — TurboQuant CUDA (Madreag/turbo3-cuda)

## Who You Are Working For

Erol Germain (@erolgermain, GitHub: Madreag). Manufacturing Engineer. Direct communicator. Does NOT tolerate:
- Skipping items
- Deferring work to "later" or "next session"
- Stopping to ask "want me to continue?"
- Implementing things halfway
- Moving to the next task before the current one is PROVEN with measurements
- Compromises, fallbacks, or "good enough"

When Erol says "figure it out" — that means investigate, debug, try multiple approaches, and solve the problem yourself. Do not give up. Do not suggest alternatives that avoid the hard work.

## Project Overview

CUDA port of Google's TurboQuant (ICLR 2026) KV cache compression for llama.cpp, targeting NVIDIA RTX 5090 (SM120 Blackwell). Goal: be the definitive TurboQuant CUDA implementation for Blackwell GPUs.

- **Repo**: `/home/erol/ai/turboquant/research/llama-cpp-turboquant/`
- **Branch**: `release/turbo3-cuda`
- **Dense model**: `/home/erol/ai/turboquant/models/opus-v2-Q6_K.gguf` (Qwen 3.5 27B, ~21 GB)
- **MoE model**: `/home/erol/ai/turboquant/models/Qwen3.5-35B-A3B-Q4_K_M.gguf` (~21 GB)
- **Hardware**: RTX 5090 32GB GDDR7, SM120, CUDA 12.8, WSL2 Ubuntu 24.04
- **Competitor reference**: `/home/erol/projects/llama-cpp-turboquant-cuda/` (spiritbuun's fork, RTX 3090)
- **Full context dump**: `.trash/resume.md` — READ THIS for complete project history, architecture decisions, and dead ends

## ⚠️ SETTLED ARCHITECTURE — DO NOT REVISIT

The decode architecture uses a **persistent fp16 shadow cache with incremental dequant**. This has been proven optimal across 7 sessions. The following alternatives have been thoroughly tested and FAILED — do NOT retry them:

1. **Fused SET_ROWS fp16 write** — speed regression (0.944x→0.919x). Tested 3 times (Sessions 2, 6, 7).
2. **Flat array shadow cache** — hash collisions corrupt data (PPL 25.93). Tested Session 5.
3. **Adaptive native/shadow** — native turbo3 vec is SLOWER than fp16 vec at ALL depths. Tested Session 7.
4. **L2 cache persistence hints** — net negative. Tested Session 2.
5. **Fused compressed attention** — register spill from dynamic indexing. Tested Session 1.

See `.trash/resume.md` for detailed failure analysis of each.

## Current Performance

| Context | q8_0 | turbo3 | Ratio |
|---------|------|--------|-------|
| short | ~55 | ~52 | **0.944x** |
| 16K | ~50 | ~50 | **0.99x** |
| 32K | ~46 | ~48 | **1.04x** (beats q8_0!) |

PPL: turbo3=6.848 (+1.32% vs q8_0 at ctx=512), 5.736 (+1.08% at ctx=2048)

## ABSOLUTE RULES

### 1. NEVER skip an item. NEVER defer.
If the plan says to do something, do it. If it's blocked, find another way.

### 2. MEASURE before and after EVERY change.
No commit without benchmark data. PPL at ctx=512 AND ctx=2048 for every code change. Record in `benchmarks/CHANGELOG.md`.

### 3. PPL REJECT THRESHOLDS — non-negotiable.
- ctx=512: turbo3 PPL > 6.89 → **REJECT and revert immediately**
- ctx=2048: turbo3 PPL > 5.77 → **REJECT and revert immediately**

### 4. ONE commit per logical change.
Implement → rebuild → measure → commit → next item.

### 5. Fix regressions IMMEDIATELY.
If any metric gets worse, stop and fix before moving on.

### 6. Read reference implementations BEFORE reimplementing.
spiritbuun's fork at `/home/erol/projects/llama-cpp-turboquant-cuda/` is the CUDA reference.

### 7. NEVER ask "want me to continue?"
The answer is always yes. Save state to `.trash/resume.md` when context is low.

### 8. Understand the ARCHITECTURE before writing code.
Read the decode data flow in `.trash/resume.md` before modifying fattn.cu or turbo-quant.cu.

## Build & Test Commands

```bash
# Build
cd /home/erol/ai/turboquant/research/llama-cpp-turboquant
/home/erol/miniconda3/envs/tq/bin/cmake --build build -j$(nproc)

# PPL gate (run for EVERY code change):
MODEL=/home/erol/ai/turboquant/models/opus-v2-Q6_K.gguf
WIKI=wikitext-2-raw/wiki.test.raw
./build/bin/llama-perplexity -m $MODEL -f $WIKI -c 512 -ctk turbo3 -ctv turbo3 -fa on --chunks 8 -ngl 99
./build/bin/llama-perplexity -m $MODEL -f $WIKI -c 2048 -ctk turbo3 -ctv turbo3 -fa on --chunks 8 -ngl 99

# Full decode curve (NOTE: llama-bench uses -d for depth, NOT -c):
for DEPTH in 0 2048 4096 8192 16384 32768; do
  ./build/bin/llama-bench -m $MODEL -fa 1 -ctk turbo3 -ctv turbo3 -d $DEPTH -ngl 99 -t 1 -r 3 -p 0 -n 128
  ./build/bin/llama-bench -m $MODEL -fa 1 -ctk q8_0 -ctv q8_0 -d $DEPTH -ngl 99 -t 1 -r 3 -p 0 -n 128
done

# Prefill:
for DEPTH in 0 8192 32768; do
  ./build/bin/llama-bench -m $MODEL -fa 1 -ctk turbo3 -ctv turbo3 -d $DEPTH -ngl 99 -t 1 -p 512 -n 0 -r 1
done
```

## Architecture Knowledge

### TurboQuant turbo3
- 3.25 bits/value, 4.6x compression vs fp16
- Block: 16 bytes per 32 values (padded from 14), rotation group = 128 values = 4 blocks
- Lloyd-Max 8 centroids, FWHT rotation for Gaussianization
- Norm correction: `corrected_norm = raw_norm / ||reconstruction||`
- Centroids: `{-0.190685, -0.117832, -0.065717, -0.021460, 0.021460, 0.065717, 0.117832, 0.190685}`

### SM120 (RTX 5090) Constraints
- 128 KB shared memory/SM, 99 KB max per block
- 48 max warps/SM (NOT 64 like SM100 datacenter)
- NO WGMMA, NO TMEM — only extended `mma.sync`
- 98 MB L2 cache, 1.79 TB/s GDDR7 bandwidth
- FlashAttention-3/4 does NOT work on SM120
- HAS native FP4 E2M1 Tensor Core support via `mma.sync.m16n8k64`
- CUDA 12.8 only (13.1 segfaults on MMQ kernels)

### Key Files
```
ggml/src/ggml-cuda/turbo-quant.cu   — CUDA dequant, WHT, SET_ROWS quantizer
ggml/src/ggml-cuda/fattn-common.cuh — FA vec_dot for turbo3 + V dequant + sparse V
ggml/src/ggml-cuda/fattn-vec.cuh    — sparse V threshold, __expf, nthreads
ggml/src/ggml-cuda/fattn.cu         — FA dispatch: shadow cache, prefill MMA
src/llama-kv-cache.cpp              — Layer-adaptive (TURBO_LAYER_ADAPTIVE)
src/llama-graph.cpp                 — Graph-level WHT rotation
```

## Common Mistakes To Avoid

### ❌ Trying to eliminate the shadow cache
The persistent fp16 shadow IS the optimal architecture. See "SETTLED ARCHITECTURE" above. The 5.6% short-context gap is inherent to the shadow approach and cannot be fixed without a fundamentally new FA kernel (e.g., FP4 Tensor Core, BitDecoding lop3).

### ❌ Re-dequanting the entire KV cache every token
O(context_length) work per token when only O(1) new data was added. Use incremental dequant.

### ❌ Reimplementing from scratch when reference code exists
spiritbuun's fork is cloned locally. Read it first.

### ❌ Batching multiple changes without measuring between them
One change → one measurement → one commit.

### ❌ Skipping PPL validation
Every code change needs PPL at ctx=512 AND ctx=2048. If it exceeds reject thresholds → revert.

### ❌ Using cudaEventSynchronize inside graph compute
CUDA graphs are enabled (USE_GRAPHS=1). Sync inside graph replay = crash.

### ❌ Stopping early to "save state"
Do NOT stop to write SESSION_STATE.md unless you literally cannot make another tool call.
Push context to the absolute limit. Save state only when forced.

## Commit Message Format

```
<type>: <short description>

<Detailed explanation of what changed and why>

<Benchmark results — REQUIRED>
  PPL impact: turbo3=X.XXX, q8_0=X.XXX, delta=+X.XX% (ctx=512)
  Decode: XX.XX tok/s (ratio vs q8_0)

Co-Authored-By: spiritbuun <271142774+spiritbuun@users.noreply.github.com>  # if adapted
```

Types: `feat`, `fix`, `perf`, `docs`, `test`, `refactor`

## Environment Variables

| Variable | Effect |
|----------|--------|
| `TURBO_LAYER_ADAPTIVE=N` | Per-layer KV type: 0=uniform, 1=first4+last4→q8_0 (best PPL) |
| `GGML_TURBO_DECODE_NATIVE=1` | Disable fp16 shadow, use native turbo3 vec (slower, for debug) |

## Full Context

**Read `.trash/resume.md`** for the complete 500-line context dump including:
- All 15 implemented optimizations with files and impacts
- Complete decode data flow (decode + prefill paths)
- Shadow cache lifecycle details
- All 6 dead ends with detailed failure analysis
- Hardware specs, competitor analysis, commit history
- turbo4 port status, asymmetric K/V status
- Research paper summaries and frontier opportunities (FP4 TC, BitDecoding, cp.async)
- WSL2 caveats

## Remember

- turbo3 BEATS q8_0 at 32K context (1.04x) while using 4.6x less KV memory
- Short context gap (0.944x) is inherent to shadow architecture — don't waste time on it
- The genuine frontiers are: turbo4 completion, FP4 Tensor Core attention, BitDecoding lop3
- Every optimization must be MEASURED. Intuition is not data.
- DO NOT STOP. DO NOT DEFER. DO NOT SKIP. FINISH THE WORK.
