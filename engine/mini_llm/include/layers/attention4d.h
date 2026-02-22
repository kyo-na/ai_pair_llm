#pragma once
#include <vector>
#include <cmath>
#include "tensor4d.h"
#include "layers/linear4d.h"

struct Attention4D {

    int dim = 0;

    Linear4D Wq;
    Linear4D Wk;
    Linear4D Wv;

    // 保存（backward 用）
    Tensor4D Q;
    Tensor4D K;
    Tensor4D V;

    // attention weights (B,T,T)
    std::vector<float> attn;

    Attention4D() = default;
    Attention4D(int d);

    Tensor4D forward(const Tensor4D& x);
    Tensor4D backward(const Tensor4D& dout);

    void step(float lr);
};