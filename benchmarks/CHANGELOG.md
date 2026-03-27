# Benchmarks Changelog

## 2026-03-28 Session 2 — Persistent Shadow Cache + Layer-Adaptive

### Commit `f15ea3462` — bf16 bypass validation
- f16 PPL: 6.756, bf16 PPL: 6.763, q8_0: 6.759, turbo3: 6.867
- bf16 requirement for Qwen 3.5 is NOT confirmed

### Commit `4b1a8fa1b` — Layer-adaptive KV cache
- Mode 0 (uniform): PPL 6.867 (+1.60% vs q8_0)
- Mode 1 (first4+last4): PPL 6.804 (+0.67% vs q8_0)
- Mode 2 (last8): PPL 6.836 (+1.15%)
- Decode speed unchanged (2 of 16 layers promoted)

### Commit `91bb63001` — Cleanup (L2 hints tested negative)
- L2 persistence hints: NET NEGATIVE (~1-2% worse at short ctx)
- Single-row dequant kernel: NO IMPROVEMENT
- Both removed, decode unchanged at 0.938x short, 0.934x 32K

### Commit `57f9a3bff` — Persistent fp16 shadow cache
- BEFORE (per-call dequant): 0.880x short, 0.801x 32K
- AFTER (persistent shadow): 0.938x short, 0.934x 32K
- Eliminates per-token bulk dequant (~64 MB → ~2 KB per token)

### Commit `8f91a5d75` — Dequant-to-fp16 prefill+decode
- Prefill: 0.624x → 0.985x at depth 0, 0.038x → 0.922x at 32K
- Decode: 0.885x → 0.930x short, 0.719x → 0.804x 32K
- Multi-seq guard (ne[3]>1) falls back to native turbo3

---

## 2026-03-27 Session 1 — Foundation

### Commits 8ae47390d through d3119d33c (12 commits)
- Fixed NaN PPL (FA dispatch safety)
- 16-byte struct padding (+11.7% at 32K)
- Register LUT + batched bytes (+2.1% at 32K)
- Norm correction (zero-cost PPL improvement)
- Auto-enable FA for turbo cache types
- llama-bench turbo3 arg parser fix

Session 1 final: decode 0.885x short → 0.719x 32K, PPL 6.867
