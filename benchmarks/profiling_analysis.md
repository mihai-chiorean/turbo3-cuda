# Profiling Analysis from Decode Curve Data (Item 0.4)

**Method**: CUDA profiling tools (ncu, nsys, nvprof) unavailable on WSL2
(no sudo for apt install, no conda/pip packages available).
Analysis derived from decode curve timing data which directly measures the
FA kernel performance across context depths.

**Attempted**: `apt install nsight-systems-cli` (requires sudo), `nsys` (not found),
`ncu` (not found), conda/pip nsight packages (none available).

## Raw Data (from baseline commit fb4c5b789)

| Context | q8_0 tok/s | turbo3 tok/s | turbo3/q8_0 | q8_0 ms/tok | turbo3 ms/tok | overhead ms |
|---------|-----------|-------------|-------------|------------|--------------|------------|
| short | 59.40 | 51.49 | 0.867x | 16.84 | 19.42 | 2.59 |
| 2K | 60.38 | 51.61 | 0.855x | 16.56 | 19.38 | 2.82 |
| 4K | 60.05 | 50.02 | 0.833x | 16.65 | 20.00 | 3.34 |
| 8K | 59.31 | 47.32 | 0.798x | 16.86 | 21.13 | 4.28 |
| 16K | 57.15 | 41.76 | 0.731x | 17.50 | 23.95 | 6.45 |
| 32K | 54.69 | 34.47 | 0.630x | 18.28 | 29.01 | 10.72 |

## Overhead Analysis

turbo3 overhead per token = turbo3 ms/tok - q8_0 ms/tok

| Context | Overhead (ms) | Overhead growth from short |
|---------|--------------|--------------------------|
| short | 2.59 | 1.00x |
| 2K | 2.82 | 1.09x |
| 4K | 3.34 | 1.29x |
| 8K | 4.28 | 1.65x |
| 16K | 6.45 | 2.49x |
| 32K | 10.72 | 4.14x |

**The turbo3 overhead grows ~linearly with context**, which is consistent with a
per-KV-position cost in the FA inner loop. Each additional KV position adds a fixed
amount of extra work for turbo3 vs q8_0.

Linear fit: overhead ≈ 2.4 + 0.00025 * context_depth (ms)
This means each KV position costs ~0.25 microseconds MORE for turbo3 than q8_0.

## Bandwidth Analysis

Qwen 3.5 27B: 16 attention layers × 4 KV heads × 256 head_dim = 16,384 elements/position

Per KV position, turbo3 loads:
- K: 16,384 values / 32 per block × 14 bytes = 7,168 bytes
- V: same = 7,168 bytes
- Total: 14,336 bytes per position for K+V

Per KV position, q8_0 loads:
- K: 16,384 values × 1.0625 bytes (q8_0 with scale) ≈ 17,408 bytes
- V: same = 17,408 bytes
- Total: 34,816 bytes per position

turbo3 reads 2.4x LESS data per position. Yet it's SLOWER. This confirms the
bottleneck is NOT memory bandwidth — it's the per-element dequant compute cost:
bitshift extraction + centroid lookup + multiply per element.

## Conclusions for Optimization Decisions

1. **Struct padding (item 1.2)**: The overhead is compute-dominated, not BW-dominated.
   Padding to 16 bytes may help slightly (better coalescing) but won't fix the core
   issue. Still worth testing empirically.

2. **Fused compressed attention (item 2.1)**: This IS the right optimization. Reducing
   per-element compute (D muls → 8 muls) directly attacks the linear overhead growth.
   The 18x bandwidth reduction is a bonus, not the primary win.

3. **Register LUT (item 3.3)**: Would reduce per-element cost by caching norm*centroid.
   Expected to flatten the overhead growth curve somewhat.

4. **The 0.630x ratio at 32K is the primary optimization target.** If we can reduce the
   per-position overhead from 0.25μs to 0.10μs, the 32K ratio would improve to ~0.80x.
