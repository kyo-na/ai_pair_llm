#pragma once
#include "tensor4d.h"
#include <vector>

class LinearVocab4D {
public:
    LinearVocab4D(int d_model, int vocab);

    Tensor4D forward(const Tensor4D& x);           // (B,T,1,d_model)->(B,T,1,V)
    Tensor4D backward(const Tensor4D& grad_out);   // grad w.r.t x

    std::vector<Tensor4D*> parameters();

private:
    int d_model_;
    int vocab_;

    Tensor4D W_;  // (1,1,d_model,V)
    Tensor4D b_;  // (1,1,1,V)

    Tensor4D last_x_;
};