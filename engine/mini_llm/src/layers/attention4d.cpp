// engine/mini_llm/src/layers/attention4d.cpp
#include <cmath>
#include <vector>
#include <algorithm>
#include "../../include/layers/attention4d.h"

namespace mini_llm {

// ----------------------------
// Softmax (last dimension)
// ----------------------------
static void softmax(std::vector<float>& v) {
    float maxv = *std::max_element(v.begin(), v.end());
    float sum = 0.0f;
    for (auto& x : v) {
        x = std::exp(x - maxv);
        sum += x;
    }
    for (auto& x : v) x /= sum;
}

// ----------------------------
// Forward
// ----------------------------
Tensor4D Attention4D::forward(const Tensor4D& x) {
    // x: [B,T,H,D]
    Tensor4D out(x.B, x.T, x.H, x.D);
    cache_x = x;

    for (int b = 0; b < x.B; ++b) {
        for (int h = 0; h < x.H; ++h) {
            for (int t = 0; t < x.T; ++t) {

                std::vector<float> scores(x.T);

                // QK^T
                for (int tp = 0; tp < x.T; ++tp) {
                    float dot = 0.0f;
                    for (int d = 0; d < x.D; ++d) {
                        dot += x(b,t,h,d) * x(b,tp,h,d);
                    }
                    scores[tp] = dot / std::sqrt((float)x.D);
                }

                softmax(scores);

                // weighted sum
                for (int d = 0; d < x.D; ++d) {
                    float v = 0.0f;
                    for (int tp = 0; tp < x.T; ++tp) {
                        v += scores[tp] * x(b,tp,h,d);
                    }
                    out(b,t,h,d) = v;
                }
            }
        }
    }
    return out;
}

// ----------------------------
// Backward（最小版）
// ----------------------------
Tensor4D Attention4D::backward(const Tensor4D& grad_out) {
    Tensor4D grad_in = grad_out;
    return grad_in;
}

void Attention4D::step(float /*lr*/) {
    // no parameters
}

} // namespace mini_llm