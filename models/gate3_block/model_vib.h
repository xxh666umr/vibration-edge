/* 自动生成：tools/quantize_vib.py（VibNet 工业振动 4 分类，int8） */
#ifndef MODEL_VIB_H
#define MODEL_VIB_H
#include <stdint.h>

#define MODEL_NAME "vib_cnn_gate3"
#define MODEL_INPUT_C      1
#define MODEL_INPUT_LEN    1024
#define MODEL_NUM_CLASSES  3
#define MODEL_INPUT_SCALE  0.0441606364f
#define MODEL_PARAM_COUNT  8435
#define MODEL_CONV1_FX_MUL   62862
#define MODEL_CONV1_FX_SHIFT 24
#define MODEL_CONV2_FX_MUL   119394
#define MODEL_CONV2_FX_SHIFT 24
#define MODEL_CONV3_FX_MUL   61626
#define MODEL_CONV3_FX_SHIFT 24

#ifdef __cplusplus
extern "C" {
#endif

int model_forward(const int8_t *input, float *probs);

#ifdef __cplusplus
}
#endif
#endif /* MODEL_VIB_H */
