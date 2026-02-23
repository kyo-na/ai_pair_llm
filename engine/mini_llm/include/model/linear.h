#pragma once
#include "../tensor4d.h"

struct Linear {
    Tensor4D weight; // [Dout, Din, 1, 1]
    Tensor4D last_x;

    Linear(int in,int out)
        : weight(out,in,1,1), last_x(1,in,1,1) {}

    Tensor4D forward(const Tensor4D& x);
    Tensor4D backward(const Tensor4D& gy);
};