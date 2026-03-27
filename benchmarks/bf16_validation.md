# bf16 KV Cache Bypass Validation

**Date**: 2026-03-28
**Model**: Qwen 3.5 27B (opus-v2-Q6_K.gguf)
**Context**: 512 tokens, 8 chunks
**Hardware**: RTX 5090, CUDA 12.8

## Results

| KV Type | PPL | vs bf16 | vs f16 |
|---------|-----|---------|--------|
| f16 | 6.756 | -0.10% | baseline |
| q8_0 | 6.759 | -0.06% | +0.04% |
| bf16 | 6.763 | baseline | +0.10% |
| turbo3 | 6.867 | +1.54% | +1.64% |
| turbo3 LA-1 | 6.804 | +0.61% | +0.71% |

## Conclusion

The "Qwen 3.5 requires bf16 KV cache" claim is **NOT confirmed** for this model
at short context. f16 actually marginally outperforms bf16 (+0.10%). The PPL
ordering is f16 < q8_0 < bf16 < turbo3, suggesting the bf16 vs f16 difference
is within noise and Qwen 3.5 27B does NOT require bf16 specifically.

The turbo3 PPL gap (+1.54% vs bf16) is entirely from 3-bit quantization effects,
not from bf16 format incompatibility. turbo3's FWHT rotation Gaussianizes the
distribution, making the format irrelevant.

With layer-adaptive mode 1 (first+last attention layers at q8_0), the gap
narrows to +0.61% — well within the <1% target.
