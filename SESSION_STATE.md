# Session State — TurboQuant CUDA

**Updated**: 2026-03-28 Session 5
**Branch**: `release/turbo3-cuda`
**Latest commit**: `02c797837`
**GitHub**: pushed

## Performance

| Context | q8_0 | turbo3 | Ratio |
|---------|------|--------|-------|
| short | 55.05 | 51.95 | 0.944x |
| 32K | 45.96 | 47.76 | 1.039x |

PPL: 6.848 (+1.32% at 512), 5.736 (+1.08% at 2048)
PPL with K=turbo3 V=q8_0: 6.804 (+0.67%)

## Session 5 Commits
1. `83593cffc` — sparse V threshold 1e-6→1e-4 (PPL improved: +1.32%)
2. `74a20aa35` — __expf in FA softmax (zero PPL impact)
3. `02c797837` — asymmetric K=turbo3 V=q8_0 FA instance

## Key Findings
- spiritbuun's 0.97x was measured on MoE model (tiny KV). Per-call dequant.
- Our persistent shadow + sparse V is the correct architecture for dense models.
- 5.6% short-context gap is inherent to fp16 shadow approach (stride mismatch + dispatch overhead).
- Flat array shadow cache caused hash collisions → REVERTED.
- __expf helps both turbo3 and q8_0 equally (doesn't change ratio).
- K=turbo3 V=q8_0 gives same PPL as LA-1 (6.804, +0.67%).

## Continuation
> Read SESSION_STATE.md.
> Key: short-context gap (5.6%) is understood and largely inherent to shadow approach.
> Next: fix asymmetric decode (needs f16+q8_0 instance or shadow bypass for V),
> continue with turbo4 debug, then Phase 3 experiments (lop3, FP4 TC).
