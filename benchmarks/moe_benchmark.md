# MoE Model Benchmark — Qwen3.5-35B-A3B Q4_K_M

**Date**: 2026-03-28
**Hardware**: RTX 5090 32GB, CUDA 12.8, SM120
**Model**: Qwen3.5-35B-A3B Q4_K_M (20.49 GiB, MoE 35B total / 3B active)

## Decode (tg128 tok/s, 3-rep)

| Context | q8_0 | turbo3 | Ratio |
|---------|------|--------|-------|
| short | 186.18 | 157.50 | 0.846x |
| 4K | 174.57 | 150.69 | 0.863x |
| 8K | 174.44 | 147.69 | 0.847x |
| 32K | 134.48 | 131.15 | **0.975x** |

## PPL (ctx=512, 8 chunks)

| Type | PPL | Delta |
|------|-----|-------|
| q8_0 | 6.125 | baseline |
| turbo3 | 6.215 | +1.47% |

## vs spiritbuun (RTX 3090)

spiritbuun reports 0.970x decode at 42K on this same model architecture.
We get 0.975x at 32K — **matching their claim** on better hardware.

## vs Dense Model (Qwen 3.5 27B)

| Context | MoE Ratio | Dense Ratio |
|---------|-----------|-------------|
| short | 0.846x | 0.944x |
| 32K | 0.975x | 1.039x |

Dense model shows better turbo3 performance at all depths because:
- Lower absolute throughput = shadow overhead is proportionally smaller
- Sparse V has more impact on the dense model's attention patterns
