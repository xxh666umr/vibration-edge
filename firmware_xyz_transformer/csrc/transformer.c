/* tiny Transformer (d=16, 1 layer, 2 heads) —— int8 权重 + float 激活
 *
 * 与 tools/export_transformer_c.py 的 numpy 量化实现逐样本一致。
 * 输入：3×1024 float（已按窗口 RMS 归一化，clip ±6）
 * 输出：probs[3]（softmax 后）
 */
#include <math.h>
#include <stdint.h>
#include <string.h>

#include "transformer.h"
#include "transformer_model.h"

#ifdef TF_HOST
static inline uint32_t tf_tick(void) { return 0u; }
#else
#include "stm32f1xx.h"
static inline uint32_t tf_tick(void) { return tf_tick(); }
#endif

uint32_t tf_cycles[6];

#define TOK   T_TOKENS_N
#define STRIDE T_PATCH_STRIDE
#define DIM   16
#define HDS   2
#define HD    8
#define FF    32
#define EPSLN 1e-5f

/* 自动生成的数组（在 transformer_model.c 中） */
extern const int8_t  T_PATCH_W[], T_QKV_W[], T_OUT_W[], T_FC1_W[], T_FC2_W[], T_HEAD_W[];
extern const float   T_PATCH_B[], T_POS[], T_QKV_B[], T_OUT_B[], T_FC1_B[], T_FC2_B[];
extern const float   T_LN1_G[], T_LN1_B[], T_LN2_G[], T_LN2_B[], T_LNF_G[], T_LNF_B[], T_HEAD_B[];

static inline int8_t q_i8(float x, float s) {
    int32_t v = (int32_t)(x / s + (x >= 0.0f ? 0.5f : -0.5f));
    if (v > 127) v = 127;
    if (v < -127) v = -127;
    return (int8_t)v;
}

/* float 输入 → int8；int8 权重 → int32 累加 → float 输出 */
static void matmul_i8w(const int8_t *w, const float *bias, const float *x,
                       int in, int out, float sw, float sx, float *y) {
    int8_t xq[64];
    const float inv_sx = 1.0f / sx;
    for (int i = 0; i < in; ++i) {
        float v = x[i] * inv_sx;
        int32_t q = (int32_t)(v + (v >= 0.0f ? 0.5f : -0.5f));
        if (q > 127) q = 127; if (q < -127) q = -127;
        xq[i] = (int8_t)q;
    }
    for (int o = 0; o < out; ++o) {
        int32_t acc = 0;
        const int8_t *wp = w + (size_t)o * in;
        for (int i = 0; i < in; ++i) acc += (int32_t)wp[i] * (int32_t)xq[i];
        y[o] = (float)acc * (sw * sx) + bias[o];
    }
}

static void layernorm(float *x, const float *g, const float *b, int n) {
    float mu = 0.0f;
    for (int i = 0; i < n; ++i) mu += x[i];
    mu /= (float)n;
    float var = 0.0f;
    for (int i = 0; i < n; ++i) { float d = x[i] - mu; var += d * d; }
    var /= (float)n;
    float inv = 1.0f / sqrtf(var + EPSLN);
    for (int i = 0; i < n; ++i) x[i] = (x[i] - mu) * inv * g[i] + b[i];
}

static inline float fast_tanh(float x) {
    if (x > 3.0f) return 1.0f;
    if (x < -3.0f) return -1.0f;
    float x2 = x * x;
    return x * (27.0f + x2) / (27.0f + 9.0f * x2);
}

static float gelu(float x) {
    return 0.5f * x * (1.0f + fast_tanh(0.7978845608f * (x + 0.044715f * x * x * x)));
}

static inline float fast_expf_neg(float x) {   /* x <= 0 */
    union { float f; int32_t i; } v;
    v.i = (int32_t)(12102203.0f * x + 1064866805.0f);
    return v.f;
}

static void softmax(float *x, int n) {
    float mx = x[0];
    for (int i = 1; i < n; ++i) if (x[i] > mx) mx = x[i];
    float sum = 0.0f;
    for (int i = 0; i < n; ++i) { x[i] = fast_expf_neg(x[i] - mx); sum += x[i]; }
    float inv = 1.0f / sum;
    for (int i = 0; i < n; ++i) x[i] *= inv;
}

static float g_h[TOK * DIM];            /* 主干 */
static int8_t g_xq[3 * T_INPUT_LEN];    /* patch 卷积的 int8 输入 */
static float g_qkv[TOK * 3 * DIM];      /* QKV */
static float g_scores[TOK * TOK];       /* 注意力分数（单头） */
static float g_ctx[TOK * DIM];          /* 注意力输出 */
static float g_f[TOK * FF];             /* FFN 隐层 */

int transformer_forward(const float *input, float *probs) {
    const uint32_t t0 = tf_tick();
    /* 1) patch conv: 3×1024 → 31×16，int8 权重 */
    {
        const float inv_p = 1.0f / T_PATCH_X_S;
        for (int i = 0; i < 3 * T_INPUT_LEN; ++i) {
            float v = input[i] * inv_p;
            int32_t q = (int32_t)(v + (v >= 0.0f ? 0.5f : -0.5f));
            if (q > 127) q = 127; if (q < -127) q = -127;
            g_xq[i] = (int8_t)q;
        }
    }
    for (int t = 0; t < TOK; ++t) {
        for (int o = 0; o < DIM; ++o) {
            int32_t acc = 0;
            for (int c = 0; c < 3; ++c) {
                const int8_t *wp = T_PATCH_W + ((size_t)o * 3 + c) * 64;
                const int8_t *xp = g_xq + c * T_INPUT_LEN + t * STRIDE;
                for (int k = 0; k < 64; ++k) acc += (int32_t)wp[k] * (int32_t)xp[k];
            }
            g_h[t * DIM + o] = (float)acc * (T_PATCH_S * T_PATCH_X_S) + T_PATCH_B[o] + T_POS[t * DIM + o];
        }
    }

    tf_cycles[0] = tf_tick() - t0;   /* patch */
    const uint32_t t1 = tf_tick();
    /* 2) QKV */
    for (int t = 0; t < TOK; ++t)
        matmul_i8w(T_QKV_W, T_QKV_B, g_h + t * DIM, DIM, 3 * DIM, T_QKV_S, T_ATTN_X_S, g_qkv + t * 3 * DIM);

    tf_cycles[1] = tf_tick() - t1;   /* qkv */
    const uint32_t t2 = tf_tick();
    /* 3) 每头：scores → softmax → ctx */
    for (int h = 0; h < HDS; ++h) {
        for (int i = 0; i < TOK; ++i) {
            for (int j = 0; j < TOK; ++j) {
                float s = 0.0f;
                for (int k = 0; k < HD; ++k)
                    s += g_qkv[i * 48 + h * HD + k] * g_qkv[j * 48 + 16 + h * HD + k];
                g_scores[i * TOK + j] = s * 0.3535533906f;  /* 1/sqrt(8) */
            }
            softmax(g_scores + i * TOK, TOK);
        }
        for (int i = 0; i < TOK; ++i)
            for (int k = 0; k < HD; ++k) {
                float a = 0.0f;
                for (int j = 0; j < TOK; ++j) a += g_scores[i * TOK + j] * g_qkv[j * 48 + 32 + h * HD + k];
                g_ctx[i * DIM + h * HD + k] = a;
            }
    }

    tf_cycles[2] = tf_tick() - t2;   /* attention */
    const uint32_t t3 = tf_tick();
    /* 4) out_proj + 残差 + LN1 */
    for (int t = 0; t < TOK; ++t) {
        float out[16];
        matmul_i8w(T_OUT_W, T_OUT_B, g_ctx + t * DIM, DIM, DIM, T_OUT_S, T_CTX_X_S, out);
        for (int i = 0; i < DIM; ++i) g_h[t * DIM + i] += out[i];
        layernorm(g_h + t * DIM, T_LN1_G, T_LN1_B, DIM);
    }

    tf_cycles[3] = tf_tick() - t3;   /* out+ln1 */
    const uint32_t t4 = tf_tick();
    /* 5) FFN: fc1 → GELU → fc2 → 残差 + final LN */
    for (int t = 0; t < TOK; ++t) {
        matmul_i8w(T_FC1_W, T_FC1_B, g_h + t * DIM, DIM, FF, T_FC1_S, T_FFN_X_S, g_f + t * FF);
        for (int i = 0; i < FF; ++i) g_f[t * FF + i] = gelu(g_f[t * FF + i]);
        float out[16];
        matmul_i8w(T_FC2_W, T_FC2_B, g_f + t * FF, FF, DIM, T_FC2_S, T_GELU_X_S, out);
        for (int i = 0; i < DIM; ++i) g_h[t * DIM + i] += out[i];
        layernorm(g_h + t * DIM, T_LNF_G, T_LNF_B, DIM);
    }

    tf_cycles[4] = tf_tick() - t4;   /* ffn */
    const uint32_t t5 = tf_tick();
    /* 6) mean pool → head */
    float pooled[16];
    for (int i = 0; i < DIM; ++i) {
        float s = 0.0f;
        for (int t = 0; t < TOK; ++t) s += g_h[t * DIM + i];
        pooled[i] = s / (float)TOK;
    }
    float logits[3];
    matmul_i8w(T_HEAD_W, T_HEAD_B, pooled, DIM, 3, T_HEAD_S, T_HEAD_X_S, logits);
    softmax(logits, 3);
    tf_cycles[5] = tf_tick() - t5;   /* head */
    int best = 0;
    for (int i = 0; i < 3; ++i) probs[i] = logits[i];
    if (probs[1] > probs[best]) best = 1;
    if (probs[2] > probs[best]) best = 2;
    return best;
}

const char *transformer_class_name(int c) {
    static const char *n[3] = {"initial_wear", "steady_wear", "accelerating_wear"};
    return (c >= 0 && c < 3) ? n[c] : "?";
}
