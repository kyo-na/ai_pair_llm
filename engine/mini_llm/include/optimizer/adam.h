#pragma once
#include "tensor4d.h"
#include <vector>

struct Adam {
    float lr, b1, b2, eps;
    int step;
    std::vector<float> m, v;

    Adam(int n, float lr_=1e-3f);

    void update(Tensor4D& w);
};