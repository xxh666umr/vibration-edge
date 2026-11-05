#ifndef TRANSFORMER_H
#define TRANSFORMER_H
#include <stdint.h>
#include "transformer_model.h"
int transformer_forward(const float *input, float *probs);
const char *transformer_class_name(int c);
#endif
