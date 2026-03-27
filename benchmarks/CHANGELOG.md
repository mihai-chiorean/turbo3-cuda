# Benchmarks Changelog

## 2026-03-28 Session 3 — Sparse V Dequant (BREAKTHROUGH)

### Commit `157b6dcff` — Sparse V dequant
**turbo3 now BEATS q8_0 at 32K by 6.5%.**

| Context | q8_0 | turbo3+sparseV | Ratio | Before |
|---------|------|----------------|-------|--------|
| short | 56.19 | 52.61 | 0.936x | 0.930x |
| 2K | 55.64 | 52.61 | 0.946x | ~0.93x |
| 16K | 50.27 | 49.96 | 0.994x | ~0.88x |
| 32K | 45.28 | 48.21 | **1.065x** | ~0.93x |

PPL improved: 6.854 (was 6.867) — removing quantization noise helps.

Based on TheTom's "Attention-Gated Value Dequantization" paper.
3 lines in fattn-vec.cuh: skip V dequant when attention weight < 1e-6.

---

## 2026-03-28 Session 2 — Persistent Shadow + Layer-Adaptive

### Persistent fp16 shadow cache (`57f9a3bff`)
- Decode at 32K: 0.801x → 0.934x (+16.6%)
- Eliminates per-token bulk dequant

### Layer-adaptive KV cache (`4b1a8fa1b`)
- PPL: +1.60% → +0.67% (LA mode 1)

### bf16 bypass validation (`f15ea3462`)
- bf16 requirement debunked for Qwen 3.5 27B

---

## 2026-03-27 Session 1 — Foundation (12 commits)
- Fixed NaN PPL, 16-byte padding, register LUT, norm correction
- Session 1 final: 0.885x short → 0.719x 32K
