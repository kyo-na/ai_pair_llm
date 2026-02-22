#pragma once
#include <vector>
#include <algorithm>

struct Tensor {
    std::vector<float> data;
    std::vector<float> grad;

    // Adam 用（★追加）
    std::vector<float> m;
    std::vector<float> v;

    int n;

    Tensor() : n(0) {}

    Tensor(int n)
        : data(n, 0.0f),
          grad(n, 0.0f),
          m(n, 0.0f),
          v(n, 0.0f),
          n(n) {}

    void zero_grad() {
        std::fill(grad.begin(), grad.end(), 0.0f);
    }
};