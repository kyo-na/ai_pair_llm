#pragma once
#include "tensor4d.h"
#include <vector>

class SwiGLU4D {
public:
    SwiGLU4D(int H, int D, int hidden);

    Tensor4D forward(const Tensor4D& x);
    Tensor4D backward(const Tensor4D& grad_out);

    std::vector<Tensor4D*> parameters();

private:
    int H_;
    int D_;
    int hidden_;

    Tensor4D W1_; // (1,1,D,hidden)
    Tensor4D W2_; // (1,1,D,hidden)

    Tensor4D last_x_;
    Tensor4D A_;   // xW1
    Tensor4D B_;   // xW2
    Tensor4D silu_; // silu(B)
};