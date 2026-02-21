#pragma once
#include "tensor4d.h"
#include <vector>

struct LayerNorm4D {
    int dim = 0;
    float eps = 1e-5f;

    // γ, β
    std::vector<float> gamma, beta;
    std::vector<float> dgamma, dbeta;
    std::vector<float> mg, vg, mb, vb;

    int step_count = 0;

    // forward保存用
    Tensor4D x_hat;
    std::vector<float> mean;
    std::vector<float> var;

    LayerNorm4D() = default;
    LayerNorm4D(int dim);

    Tensor4D forward(const Tensor4D& x);
    Tensor4D backward(const Tensor4D& dout);

    void zero_grad();
    void step(float lr);
};