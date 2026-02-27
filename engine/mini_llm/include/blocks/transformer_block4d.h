#pragma once
#include "layers/attention4d.h"
#include "layers/rmsnorm4d.h"
#include "layers/swiglu4d.h"

class TransformerBlock4D {
public:
    TransformerBlock4D(
        int heads,
        int head_dim,
        int ffn_hidden);

    Tensor4D forward(const Tensor4D& x);
    Tensor4D backward(const Tensor4D& grad);

private:
    Attention4D attn_;
    RMSNorm4D norm1_;
    RMSNorm4D norm2_;
    SwiGLU4D ffn_;
};