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
- **Model**: `/home/erol/ai/turboquant/models/opus-v2-Q6_K.gguf` (Qwen 3.5 27B, ~21 GB)
- **Hardware**: RTX 5090 32GB GDDR7, SM120, CUDA 12.8, WSL2 Ubuntu 24.04
- **Competitor reference**: `/home/erol/projects/llama-cpp-turboquant-cuda/` (spiritbuun's fork, RTX 3090)

## ABSOLUTE RULES

### 1. NEVER skip an item. NEVER defer.
If the plan says to do something, do it. If it's blocked, find another way. "Blocked" means you haven't tried hard enough. There is always a path forward.

### 2. MEASURE before and after EVERY change.
No commit without benchmark data. Run smoke test (PPL + speed) after every kernel change. Run full decode curve before milestone commits. Record results in `benchmarks/CHANGELOG.md`.

### 3. ONE commit per logical change.
Do not batch multiple items into one commit. Implement → rebuild → measure → commit → next item. This makes it possible to bisect regressions.

### 4. Fix regressions IMMEDIATELY.
If any metric gets worse after a change, stop and fix it before moving on. Speed went down? Fix it. PPL went up? Fix it. Prefill broke? Fix it. Do NOT continue building on top of a regression.

### 5. Read reference implementations CAREFULLY before reimplementing.
spiritbuun's fork is at `/home/erol/projects/llama-cpp-turboquant-cuda/`. When the plan says to implement something spiritbuun already has, READ THEIR CODE FIRST. Understand their approach, their edge cases, their buffer management. Do not reimplement from scratch and get it wrong. Adapt their proven approach to our codebase.

### 6. NEVER ask "want me to continue?"
The answer is always yes. If you're running low on context, save state to `SESSION_STATE.md` with exact checklist status, current metrics, and what comes next. Then tell me to start a new session with that file.

### 7. Understand the ARCHITECTURE before writing code.
When implementing a performance optimization:
- First, understand the data flow (what gets allocated when, what persists, what's temporary)
- Second, understand the hot path (what runs per-token vs per-prompt vs once)
- Third, identify where the cost is (memory bandwidth? compute? allocation overhead? kernel launch?)
- THEN write code. Do not write code first and debug performance second.

## Build & Test Commands

```bash
# Build
cd /home/erol/ai/turboquant/research/llama-cpp-turboquant
/home/erol/miniconda3/envs/tq/bin/cmake --build build -j$(nproc)

# Quick smoke test (~3 min) — after EVERY kernel change
MODEL=/home/erol/ai/turboquant/models/opus-v2-Q6_K.gguf
WIKI=wikitext-2-raw/wiki.test.raw
./build/bin/llama-perplexity -m $MODEL -f $WIKI -c 512 -ctk turbo3 -ctv turbo3 -fa on --chunks 8 -ngl 99
./build/bin/llama-bench -m $MODEL -fa 1 -ctk turbo3 -ctv turbo3 -d 0 -ngl 99 -t 1 -p 0 -n 128

# Full decode curve — before milestone commits
# NOTE: llama-bench uses -d for depth, NOT -c
for DEPTH in 0 2048 4096 8192 16384 32768; do
  ./build/bin/llama-bench -m $MODEL -fa 1 -ctk q8_0 -ctv q8_0 -d $DEPTH -ngl 99 -t 1 -r 3 -p 0 -n 128
  ./build/bin/llama-bench -m $MODEL -fa 1 -ctk turbo3 -ctv turbo3 -d $DEPTH -ngl 99 -t 1 -r 3 -p 0 -n 128
done

# Full PPL at multiple depths
for CTX in 512 2048 8192; do
  ./build/bin/llama-perplexity -m $MODEL -f $WIKI -c $CTX -ctk turbo3 -ctv turbo3 -fa on --chunks 8 -ngl 99
  ./build/bin/llama-perplexity -m $MODEL -f $WIKI -c $CTX -ctk q8_0 -ctv q8_0 -fa on --chunks 8 -ngl 99
done

# Prefill at depths
for DEPTH in 0 8192 32768; do
  ./build/bin/llama-bench -m $MODEL -fa 1 -ctk turbo3 -ctv turbo3 -d $DEPTH -ngl 99 -t 1 -p 512 -n 0 -r 1
  ./build/bin/llama-bench -m $MODEL -fa 1 -ctk q8_0 -ctv q8_0 -d $DEPTH -ngl 99 -t 1 -p 512 -n 0 -r 1
done
```

## Performance Targets

| Metric | Target | Competitor (spiritbuun RTX 3090) |
|--------|--------|----------------------------------|
| Decode ratio vs q8_0 (all depths) | >0.95x | 0.970x at 42K |
| Prefill ratio vs q8_0 | >0.95x | 0.988x |
| PPL vs q8_0 | <+1% | -1.17% (beats q8_0) |
| Context scaling (short→32K) | FLAT | FLAT |

We are on BETTER hardware (RTX 5090 vs RTX 3090). Our numbers must match or beat spiritbuun's ratios.

## Architecture Knowledge

### Qwen 3.5 27B
- 64 layers: 48 GatedDeltaNet (no KV) + 16 GatedAttention (with KV)
- Attention: head_dim=256, Q_heads=24, KV_heads=4 (GQA 6:1)
- Only 16 layers have KV cache — turbo3 compression applies to these only

### TurboQuant turbo3
- 3.25 bits/value, 4.6x compression vs fp16
- Block: 16 bytes per 32 values (padded from 14), rotation group = 128 values = 4 blocks
- Lloyd-Max 8 centroids, FWHT rotation for Gaussianization
- Norm correction: `corrected_norm = raw_norm / ||reconstruction||`

### SM120 (RTX 5090) Constraints
- 128 KB shared memory/SM, 99 KB max per block
- 48 max warps/SM (NOT 64 like SM100 datacenter)
- NO WGMMA, NO TMEM — only extended `mma.sync` (Ampere-style)
- 98 MB L2 cache, 1.79 TB/s GDDR7 bandwidth
- FlashAttention-3/4 does NOT work on SM120
- HAS native FP4 E2M1 Tensor Core support via `mma.sync.m16n8k64`

### Key Files
```
ggml/src/ggml-common.h              — block_turbo3_0 struct
ggml/src/ggml-cuda/turbo-quant.cu   — CUDA dequant, WHT, SET_ROWS quantizer
ggml/src/ggml-cuda/fattn-common.cuh — FA vec_dot for turbo3 + V dequant
ggml/src/ggml-cuda/fattn.cu         — FA dispatch (turbo3 prefill/decode routing)
ggml/src/ggml-cuda/set-rows.cu      — SET_ROWS dispatch
src/llama-context.cpp               — Auto-enable FA for turbo
src/llama-graph.cpp                 — Graph-level WHT rotation (Q forward, V inverse)
```

## Common Mistakes To Avoid

### ❌ Re-dequanting the entire KV cache every token
If you allocate a temp fp16 buffer, dequant all K/V, run the kernel, then free the buffer on every decode call — you're doing O(context_length) work per token when only O(1) new data was added. The fp16 data must be cached persistently or dequanted incrementally.

### ❌ Reimplementing from scratch when reference code exists
spiritbuun's fork is cloned locally. Read it. Understand it. Adapt it. Do not rebuild from first principles and miss critical details like buffer lifetime management, multi-sequence edge cases, or rotation handling.

### ❌ Optimizing the wrong layer
Before optimizing, identify WHERE the time is spent. The bottleneck shifts with context length. At short context, it's compute. At long context, it's memory bandwidth. Profile (or use decode curve shape as a proxy) before deciding what to optimize.

### ❌ Batching multiple changes without measuring between them
If you change the struct layout AND the dequant path AND the dispatch logic in one commit, you can't tell which change helped and which hurt. One change → one measurement → one commit.

### ❌ Skipping edge cases
Multi-sequence batching (ne[3]>1), head_dim != 256, models without FWHT rotation, FA disabled by user — all of these must work. Test the edge cases, not just the happy path.

## Commit Message Format

```
<type>: <short description>

<Detailed explanation of what changed and why>

<Benchmark results — REQUIRED>
  Metric: before → after (change%)

Co-Authored-By: spiritbuun <271142774+spiritbuun@users.noreply.github.com>  # if adapted from their code
Made-with: Cursor
```

Types: `feat`, `fix`, `perf`, `docs`, `test`, `refactor`

## Benchmarks Directory Structure

```
benchmarks/
├── smoke/          # Quick per-change checks
├── gate/           # Multi-context milestone gates
├── nsight/         # Profiling data (if available)
├── community/      # TheTom diagnostic zips
└── CHANGELOG.md    # Living log: date, commit, what changed, before/after numbers
```

Every entry in CHANGELOG.md must have: date, git commit hash, what changed, before/after metrics table.

## Current Plan

Read `IMPROVEMENT_PLAN.md` in the repo root (or `.trash/IMPROVEMENT_PLAN.md`) for the full V3.2 plan. Read `SESSION_STATE.md` for what was completed and what's next.

## Remember

- We are competing with spiritbuun (RTX 3090, 97% decode ratio). We must beat them.
- We are on RTX 5090 — better hardware. No excuses for worse numbers.
- Blackwell-specific optimizations (L2 set-aside, FP4 TC, cp.async.bulk) are our differentiators.
- Every optimization must be MEASURED. Intuition is not data. Profile, benchmark, prove it.
- DO NOT STOP. DO NOT DEFER. DO NOT SKIP. FINISH THE WORK.
