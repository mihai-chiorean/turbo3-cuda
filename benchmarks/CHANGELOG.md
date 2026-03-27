# Benchmarks Changelog

## 2026-03-28 Session 3 — DEFINITIVE RESULTS

### turbo3 BEATS q8_0 at long context

| Context | q8_0 | turbo3 | Ratio |
|---------|------|--------|-------|
| short | 58.60 | 55.18 | **0.942x** |
| 8K | 53.66 | 51.04 | **0.951x** |
| 16K | 50.26 | 49.85 | **0.992x** |
| 32K | 45.29 | 48.15 | **1.063x** |

PPL: 6.854 (+1.41% vs q8_0), 6.804 with LA-1 (+0.67%)
Prefill: 0.977x at d0, 0.970x at d32K

Key: sparse V dequant (`157b6dcff`) + persistent shadow cache (`57f9a3bff`)

### Commit `157b6dcff` — Sparse V dequant
Based on TheTom's "Attention-Gated Value Dequantization" paper.
Skip V for attention weight < 1e-6. At 32K, 90%+ positions skipped.

### Commit `65da87a21` — turbo4 SET_ROWS (completes turbo4 pipeline)

---

## 2026-03-28 Session 2

### `57f9a3bff` — Persistent fp16 shadow cache
- Decode 32K: 0.801x → 0.934x

### `4b1a8fa1b` — Layer-adaptive KV cache
- PPL: +1.60% → +0.67% (LA-1)

---

## 2026-03-27 Session 1 — Foundation (12 commits)
- NaN PPL fix, 16-byte padding, register LUT, norm correction
- Session 1 final: 0.885x short → 0.719x 32K
