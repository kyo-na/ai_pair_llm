#pragma once
#include "tensor4d.h"
#include "layers/linear4d.h"

struct FFN4D {
    Linear4D fc1,fc2;
    Tensor4D h;

    FFN4D(int H,int D);

    Tensor4D forward(const Tensor4D& x);
    Tensor4D backward(const Tensor4D& x,const Tensor4D& dout);

    void step(float lr);
};