// linear.h
#pragma once
#include "tensor4d.h"

struct Linear {
    Tensor4D W;
    Linear(int in,int out):W(1,1,in,out){}
    Tensor4D forward(const Tensor4D& x);
    Tensor4D backward(const Tensor4D& gy);
};