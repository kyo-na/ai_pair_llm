#pragma once
#include <vector>
#include "tensor4d.h"

struct Linear4D {
    int in_dim = 0;
    int out_dim = 0;

    // W: (in_dim, out_dim), b: (out_dim)
    std::vector<float> W, dW, mW, vW;
    std::vector<float> b, db, mb, vb;

    int step_count = 0;

    Tensor4D last_x; // forward入力保存

    Linear4D() = default;
    Linear4D(int in_dim, int out_dim);

    Tensor4D forward(const Tensor4D& x);          // x: (B,T,H,in) -> y: (B,T,H,out)
    Tensor4D backward(const Tensor4D& dout);      // dout same as y, returns dx same as x

    void zero_grad();
    void step(float lr);
};