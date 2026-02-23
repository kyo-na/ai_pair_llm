// engine/mini_llm/src/layers/ffn4d.cpp
#include "../../include/layers/ffn4d.h"
#include <algorithm>

namespace mini_llm {

// ----------------------------
// Forward
// ----------------------------
Tensor4D FFN4D::forward(const Tensor4D& x) {
    cache_x = x;

    Tensor4D h(x.B, x.T, x.H, hidden);
    Tensor4D out(x.B, x.T, x.H, x.D);

    // Linear1 + ReLU
    for (int i = 0; i < x.size(); ++i) {
        float v = 0.0f;
        for (int d = 0; d < x.D; ++d)
            v += x.data[i * x.D + d] * w1[d];
        h.data[i * hidden] = std::max(0.0f, v);
    }

    // Linear2
    for (int i = 0; i < out.size(); ++i) {
        float v = 0.0f;
        for (int d = 0; d < hidden; ++d)
            v += h.data[i * hidden + d] * w2[d];
        out.data[i] = v;
    }

    cache_h = h;
    return out;
}

// ----------------------------
// Backward
// ----------------------------
Tensor4D FFN4D::backward(const Tensor4D& grad_out) {
    Tensor4D grad_x = cache_x;

    // grad w2
    for (int i = 0; i < grad_out.size(); ++i)
        for (int d = 0; d < hidden; ++d)
            gw2[d] += grad_out.data[i] * cache_h.data[i * hidden + d];

    // grad w1
    for (int i = 0; i < grad_out.size(); ++i) {
        float g = 0.0f;
        for (int d = 0; d < hidden; ++d)
            g += grad_out.data[i] * w2[d];
        if (cache_h.data[i * hidden] <= 0) g = 0;

        for (int d = 0; d < cache_x.D; ++d)
            gw1[d] += g * cache_x.data[i * cache_x.D + d];
    }

    return grad_x;
}

// ----------------------------
// Step
// ----------------------------
void FFN4D::step(float lr) {
    for (int i = 0; i < hidden; ++i) {
        w2[i] -= lr * gw2[i];
        gw2[i] = 0;
    }
    for (int i = 0; i < cache_x.D; ++i) {
        w1[i] -= lr * gw1[i];
        gw1[i] = 0;
    }
}

} // namespace mini_llm