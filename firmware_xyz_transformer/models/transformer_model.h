/* 自动生成：tiny Transformer d=16 接口 */
#ifndef TRANSFORMER_MODEL_H
#define TRANSFORMER_MODEL_H
#include <stdint.h>

#define T_INPUT_C   3
#define T_INPUT_LEN 1024
#define T_NUM_CLASSES 3
#define T_PARAM_COUNT 5891
#define T_PATCH_X_S  4.724409449e-02f
#define T_ATTN_X_S   4.102928807e-02f
#define T_CTX_X_S    2.900036030e-02f
#define T_FFN_X_S    3.620703232e-02f
#define T_GELU_X_S   3.413073159e-02f
#define T_HEAD_X_S   8.339886067e-03f

extern const float T_PATCH_S, T_QKV_S, T_OUT_S, T_FC1_S, T_FC2_S, T_HEAD_S;
extern const int T_D, T_HEADS, T_HD, T_FF, T_TOKENS;

int transformer_forward(const float *input, float *probs);

#endif
