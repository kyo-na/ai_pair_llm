#pragma once
#include "tensor4d.h"
#include "layers/layernorm4d.h"
#include "layers/attention4d.h"
#include "layers/linear4d.h"

struct TransformerBlock4D {
    LayerNorm4D ln1;
    Attention4D attn;
    LayerNorm4D ln2;
    Linear4D    ffn;

    Tensor4D last_x;

    TransformerBlock4D(int dim)
        : ln1(dim), attn(dim), ln2(dim), ffn(dim, dim), last_x() {}

    Tensor4D forward(const Tensor4D& x);
    Tensor4D backward(const Tensor4D& x, const Tensor4D& dout);
    void step(float lr);
};