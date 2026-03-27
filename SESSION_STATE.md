# Session State — TurboQuant CUDA

**Updated**: 2026-03-27 Session 8
**Branch**: `release/turbo3-cuda`
**Latest commit**: `c2749ad48`

## Performance
| Context | q8_0 | turbo3 | Ratio |
|---------|------|--------|-------|
| short | 55.05 | 51.95 | 0.944x |
| 32K | 45.96 | 47.76 | 1.039x |

PPL: 6.848 (+1.32% at 512), 5.736 (+1.08% at 2048)

## Session 8 Summary

### Phase 1A: Asymmetric K=turbo3 V=q8_0 decode — FIXED
- **Two bugs**: mixed-type guard in get_best_fattn_kernel (f16+q8_0 not allowed), and missing VEC dispatch entry + CMake glob
- PPL: 6.804 (+0.67% at 512), 5.650 (-0.42% at 2048) — better than q8_0 at long context!
- Decode: ~55.5 tok/s short (0.969x)
- **Committed**: ef506d510

### Phase 1B: turbo4 end-to-end — WORKING
- **Bug 1**: TURBO4_0 missing from GET_ROWS and SET_ROWS supports_op in ggml-cuda.cu
- **Bug 2**: turbo4 ctx=512 PPL NaN was actually a batch_size issue (n_seq>1 → native turbo4 vec path broken)
- Shadow path works perfectly: PPL 5.743 at ctx=2048
- Decode: 52.5 tok/s short, 47.6 tok/s at 32K
- **Known issue**: native turbo4 vec_dot gives NaN with Q->ne[3]>1 (multi-seq PPL at ctx=512)
- **Committed**: 79d8158a7

### Phase 2: README + Discussion Post
- README updated with MoE table, turbo4, asymmetric K/V
- Discussion post draft at .trash/DISCUSSION_POST.md
- **Committed**: c2749ad48

### Phase 3: lop3 Research
- Cloned BitDecoding to /tmp/bitdecoding
- Key finding: no direct 3-bit lop3 template (only 2-bit and 4-bit)
- lop3 benefits require full TC-based FA kernel, not vec_dot optimization
- Centroid lookup (8 arbitrary floats) can't be done bitwise
- The 0x64006400 constant embeds values in bf16 format for TC fragments
- Multi-day effort — deferred to dedicated session

## Key Findings This Session
- `-b 512` flag produces bad PPL (14.79) for ALL turbo types — perplexity evaluator issue with batch=ctx
- `Q->ne[3] != 1` bypass is required for multi-seq correctness (removing it breaks turbo3 too)
- turbo4 SET_ROWS cross-space residual (normalized-recon) vs rotated-space (x-recon) makes no meaningful difference
- The f16+q8_0 FA instance existed upstream but wasn't compiled without FA_ALL_QUANTS

## Continuation Prompt
> Read SESSION_STATE.md. Branch: release/turbo3-cuda.
>
> Phase 1-2 complete. Push: already done.
>
> Remaining from Phase 4:
> 1. Test turbo4 on MoE model
> 2. Pull spiritbuun + TheTom for new commits
> 3. Run 128K context test on MoE with turbo3
> 4. Profile individual kernel times (chrono approach)
> 5. Write technical documentation of shadow cache architecture
>
> Future (separate sessions):
> - lop3 TC-based FA kernel (the real moonshot past 0.94x)
> - Fix native turbo4 vec_dot for multi-seq (Q->ne[3]>1)
> - Backend-ops test integration
