# TurboQuant Benchmark Changelog

## Baseline — Pre-optimization (2026-03-26)

**Commit**: `8ae47390d` (fix: add turbo3/turbo4 cache types to llama-bench arg parser)
**Hardware**: RTX 5090 32GB, CUDA 12.8, driver 581.80, SM120
**Model**: opus-v2-Q6_K.gguf (Qwen 3.5 27B, 20.56 GiB)
**Source**: TheTom turbo-diag v5, zip: `benchmarks/community/turbo-diag-20260326-200603.zip`

### Prefill (tok/s)

| Context | q8_0 | turbo3 | Ratio |
|---------|------|--------|-------|
| 2K | 2756 | 2933 | 1.064x |
| 4K | 3151 | 3070 | 0.974x |
| 8K | 3122 | 3023 | 0.968x |
| 16K | 3014 | 2933 | 0.973x |
| 32K | 2805 | 2738 | 0.976x |

**Verdict**: Prefill at parity (97-106% of q8_0). No prefill regression.

### Decode (tok/s, tg128)

| Context | q8_0 | turbo3 | Ratio |
|---------|------|--------|-------|
| short | 57.8 | 50.0 | 0.865x |
| 4K | 57.6 | 50.5 | 0.876x |
| 8K | 57.0 | 48.5 | 0.852x |
| 16K | 54.7 | 44.1 | 0.807x |
| 32K | 52.9 | 38.1 | 0.720x |

**Verdict**: Decode degrades from 0.87x to 0.72x across context. This is the optimization target.

### Perplexity

| Cache | PPL |
|-------|-----|
| q8_0 | (not reported in this run) |
| turbo3 | **NaN** |

**CRITICAL**: turbo3 PPL returns NaN. Likely caused by FA dispatch bug routing turbo3 to
MMA kernel at batch>2 (item 0.1). Must fix before any quality measurements are valid.

### Anomalies

- Swap grew 3588 MB during test (WSL2 memory pressure)
- llama-cli with --jinja timed out (60s) during model info extraction
- Memory check at 32K context timed out (120s)

---

## After P0+P1 Fixes (2026-03-26)

**Changes**: 0.1 FA dispatch safety, 0.3 dead code removal, 1.1 norm correction, 1.3 __launch_bounds__

### Perplexity (8 chunks, ctx 512)

| Cache | PPL | vs q8_0 |
|-------|-----|---------|
| q8_0 | 6.759 | baseline |
| **turbo3** | **6.867** | **+1.6%** |

**PPL was NaN before FA dispatch fix — now within 1.6% of q8_0.**

### Speed (pp512, tg128)

| Cache | pp512 tok/s | tg128 tok/s |
|-------|------------|------------|
| turbo3 | 2167 | 51.95 |

Note: pp512 dropped vs baseline (3073→2167) because vec kernel is correctly used now
instead of the MMA kernel that was producing wrong results (NaN PPL).

---

## Post-P0P1 Full Baseline (2026-03-27)

**Commit**: `39f8bce3d` (fix: auto-enable flash attention for turbo cache types)
**This is the reference baseline for all future optimizations.**

### Decode Curve (tg128 tok/s)

| Context | q8_0 | turbo3 | Ratio |
|---------|------|--------|-------|
| short | 59.40 | 51.49 | 0.867x |
| 2K | 60.38 | 51.61 | 0.855x |
| 4K | 60.05 | 50.02 | 0.833x |
| 8K | 59.31 | 47.32 | 0.798x |
| 16K | 57.15 | 41.76 | 0.731x |
| 32K | 54.69 | 34.47 | 0.630x |

Context scaling: 0.867x → 0.630x (ratio drops 27% from short to 32K)

### Perplexity (8 chunks)

| Context | q8_0 PPL | turbo3 PPL | Delta |
|---------|----------|------------|-------|
| 512 | 6.759 | 6.841 | +1.2% |
| 2048 | 5.674 | 5.697 | +0.4% |
| 8192 | 6.667 | 6.784 | +1.8% |

Quality within 2% target at all depths. Best at 2K context (+0.4%).

---

## After Padding + Register LUT (2026-03-27)

**Commits**: `2f5fafb93` (16B padding) + `36a97a61f` (register LUT + batched bytes)

### Decode Curve (tg128 tok/s)

| Context | Original | +Padding | +Reg LUT | Total improvement |
|---------|----------|----------|----------|------------------|
| short | 51.49 | 52.10 | 52.59 | **+2.1%** |
| 2K | 51.61 | 52.65 | 52.72 | **+2.2%** |
| 4K | 50.02 | 52.48 | 52.87 | **+5.7%** |
| 8K | 47.32 | 49.55 | 49.73 | **+5.1%** |
| 16K | 41.76 | 45.36 | 45.83 | **+9.7%** |
| 32K | 34.47 | 38.51 | 39.30 | **+14.0%** |

### turbo3/q8_0 Ratio

| Context | Before | After | Delta |
|---------|--------|-------|-------|
| short | 0.867x | 0.885x | +0.018 |
| 4K | 0.833x | 0.881x | +0.048 |
| 8K | 0.798x | 0.838x | +0.040 |
| 16K | 0.731x | 0.802x | +0.071 |
| 32K | 0.630x | 0.719x | +0.089 |

### Prefill

turbo3 pp512 = 1974 tok/s, q8_0 pp512 = 3062 tok/s → 0.645x.
Prefill regressed from baseline (~97-106% parity) due to FA dispatch fix forcing
vec-only kernel. MMA kernel was faster but produced wrong results (NaN PPL).
Item 2.2 (dequant-then-MMA prefill) would address this.

### PPL

Unchanged at 6.867 (turbo3) vs 6.759 (q8_0) = +1.6%.

---
