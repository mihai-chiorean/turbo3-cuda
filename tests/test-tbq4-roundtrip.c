/*
 * test-tbq4-roundtrip.c -- TBQ4_0 quantization round-trip unit tests
 *
 * Tests the CPU reference quantizer (quantize_row_tbq4_0_ref) and
 * dequantizer (dequantize_row_tbq4_0) for correctness.
 *
 * Build:
 *   cd build && cmake --build . --target test-tbq4-roundtrip
 * Or manually:
 *   gcc -O2 -I../ggml/include -I../ggml/src tests/test-tbq4-roundtrip.c \
 *       -Lbuild/bin -lggml-base -lm -o build/bin/test-tbq4-roundtrip
 */

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>

#include "ggml.h"

/* Declarations -- these are exported from libggml-base */
typedef struct {
    uint16_t d;                  /* ggml_half scale */
    uint8_t  qs[64];            /* 4-bit codebook indices */
} block_tbq4_0_test;

extern void quantize_row_tbq4_0_ref(const float * x, void * y, long long k);
extern void dequantize_row_tbq4_0 (const void * x, float * y, long long k);

#define QK 128  /* TBQ4_0 block size */

/* ── Helpers ─────────────────────────────────────────────────────────── */

static float rmse(const float * a, const float * b, int n) {
    double sum = 0;
    for (int i = 0; i < n; i++) {
        double d = a[i] - b[i];
        sum += d * d;
    }
    return (float)sqrt(sum / n);
}

static float cosine_sim(const float * a, const float * b, int n) {
    double dot = 0, na = 0, nb = 0;
    for (int i = 0; i < n; i++) {
        dot += a[i] * b[i];
        na  += a[i] * a[i];
        nb  += b[i] * b[i];
    }
    if (na == 0 || nb == 0) return 0;
    return (float)(dot / sqrt(na) / sqrt(nb));
}

static float l2_norm(const float * a, int n) {
    double s = 0;
    for (int i = 0; i < n; i++) s += a[i] * a[i];
    return (float)sqrt(s);
}

/* ── Test Cases ──────────────────────────────────────────────────────── */

static int num_pass = 0;
static int num_fail = 0;

#define CHECK(cond, fmt, ...) do { \
    if (cond) { \
        num_pass++; \
        printf("  PASS: " fmt "\n", ##__VA_ARGS__); \
    } else { \
        num_fail++; \
        printf("  FAIL: " fmt "\n", ##__VA_ARGS__); \
    } \
} while(0)

static void test_sinusoid(void) {
    printf("\nTest: sinusoidal vector (typical KV cache values)\n");
    float input[QK], output[QK];
    char buf[256];  /* plenty of room for one block (66 bytes) */

    for (int i = 0; i < QK; i++)
        input[i] = sinf(i * 0.1f + 0.5f) * 2.0f;

    quantize_row_tbq4_0_ref(input, buf, QK);
    dequantize_row_tbq4_0(buf, output, QK);

    float err = rmse(input, output, QK);
    float cos = cosine_sim(input, output, QK);
    float norm_in = l2_norm(input, QK);
    float norm_out = l2_norm(output, QK);
    float norm_ratio = norm_out / norm_in;

    printf("  RMSE=%.6f  Cosine=%.6f  NormRatio=%.4f\n", err, cos, norm_ratio);
    CHECK(err < 0.15f, "RMSE < 0.15 (got %.6f)", err);
    CHECK(cos > 0.99f, "Cosine > 0.99 (got %.6f)", cos);
    CHECK(norm_ratio > 0.90f && norm_ratio < 1.10f,
          "Norm ratio in [0.90, 1.10] (got %.4f)", norm_ratio);
}

static void test_large_values(void) {
    printf("\nTest: large-magnitude vector\n");
    float input[QK], output[QK];
    char buf[256];

    for (int i = 0; i < QK; i++)
        input[i] = cosf(i * 0.3f) * 100.0f;

    quantize_row_tbq4_0_ref(input, buf, QK);
    dequantize_row_tbq4_0(buf, output, QK);

    float cos = cosine_sim(input, output, QK);
    float norm_ratio = l2_norm(output, QK) / l2_norm(input, QK);

    printf("  Cosine=%.6f  NormRatio=%.4f\n", cos, norm_ratio);
    CHECK(cos > 0.98f, "Cosine > 0.98 for large values (got %.6f)", cos);
    CHECK(norm_ratio > 0.85f && norm_ratio < 1.15f,
          "Norm ratio in [0.85, 1.15] (got %.4f)", norm_ratio);
}

static void test_small_values(void) {
    printf("\nTest: small-magnitude vector\n");
    float input[QK], output[QK];
    char buf[256];

    for (int i = 0; i < QK; i++)
        input[i] = sinf(i * 0.7f) * 0.001f;

    quantize_row_tbq4_0_ref(input, buf, QK);
    dequantize_row_tbq4_0(buf, output, QK);

    float cos = cosine_sim(input, output, QK);
    printf("  Cosine=%.6f\n", cos);
    /* Small values are harder to quantize accurately */
    CHECK(cos > 0.90f, "Cosine > 0.90 for small values (got %.6f)", cos);
}

static void test_zero_vector(void) {
    printf("\nTest: zero vector\n");
    float input[QK], output[QK];
    char buf[256];

    memset(input, 0, sizeof(input));

    quantize_row_tbq4_0_ref(input, buf, QK);
    dequantize_row_tbq4_0(buf, output, QK);

    float norm_out = l2_norm(output, QK);
    printf("  Output norm=%.8f\n", norm_out);
    CHECK(norm_out < 0.01f, "Zero vector stays near-zero (norm=%.8f)", norm_out);
}

static void test_basis_vector(void) {
    printf("\nTest: basis vector e_0 = [1, 0, 0, ...]\n");
    float input[QK], output[QK];
    char buf[256];

    memset(input, 0, sizeof(input));
    input[0] = 1.0f;

    quantize_row_tbq4_0_ref(input, buf, QK);
    dequantize_row_tbq4_0(buf, output, QK);

    float cos = cosine_sim(input, output, QK);
    float norm_out = l2_norm(output, QK);
    printf("  out[0]=%.6f  Cosine=%.6f  Norm=%.6f\n", output[0], cos, norm_out);
    /* Basis vectors are worst-case for rotation-based quantizers */
    CHECK(cos > 0.80f, "Cosine > 0.80 for basis vector (got %.6f)", cos);
    CHECK(norm_out > 0.5f, "Norm > 0.5 (got %.6f)", norm_out);
}

static void test_multi_block(void) {
    printf("\nTest: multi-block (4 blocks = 512 elements)\n");
    const int N = 4 * QK;
    float input[4 * QK], output[4 * QK];
    char buf[4 * 256];  /* 4 blocks */

    for (int i = 0; i < N; i++)
        input[i] = sinf(i * 0.05f) * 3.0f + cosf(i * 0.13f);

    quantize_row_tbq4_0_ref(input, buf, N);
    dequantize_row_tbq4_0(buf, output, N);

    float err = rmse(input, output, N);
    float cos = cosine_sim(input, output, N);
    printf("  RMSE=%.6f  Cosine=%.6f\n", err, cos);
    CHECK(err < 0.20f, "Multi-block RMSE < 0.20 (got %.6f)", err);
    CHECK(cos > 0.99f, "Multi-block cosine > 0.99 (got %.6f)", cos);
}

static void test_alternating_signs(void) {
    printf("\nTest: alternating sign pattern\n");
    float input[QK], output[QK];
    char buf[256];

    for (int i = 0; i < QK; i++)
        input[i] = (i % 2 == 0) ? 1.5f : -1.5f;

    quantize_row_tbq4_0_ref(input, buf, QK);
    dequantize_row_tbq4_0(buf, output, QK);

    float cos = cosine_sim(input, output, QK);
    float norm_ratio = l2_norm(output, QK) / l2_norm(input, QK);
    printf("  Cosine=%.6f  NormRatio=%.4f\n", cos, norm_ratio);
    CHECK(cos > 0.95f, "Alternating cosine > 0.95 (got %.6f)", cos);
}

static void test_deterministic(void) {
    printf("\nTest: deterministic (two quantizations of same input match)\n");
    float input[QK];
    char buf1[256], buf2[256];

    for (int i = 0; i < QK; i++)
        input[i] = sinf(i * 0.2f) * 5.0f;

    quantize_row_tbq4_0_ref(input, buf1, QK);
    quantize_row_tbq4_0_ref(input, buf2, QK);

    /* Compare the raw quantized bytes (66 bytes per block) */
    int match = (memcmp(buf1, buf2, 66) == 0);
    CHECK(match, "Quantization is deterministic");
}

/* ── Main ────────────────────────────────────────────────────────────── */

int main(void) {
    printf("=== TBQ4_0 Round-Trip Unit Tests ===\n");

    test_sinusoid();
    test_large_values();
    test_small_values();
    test_zero_vector();
    test_basis_vector();
    test_multi_block();
    test_alternating_signs();
    test_deterministic();

    printf("\n=== Results: %d passed, %d failed ===\n", num_pass, num_fail);
    return num_fail > 0 ? 1 : 0;
}
