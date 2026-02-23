#include <iostream>
#include <vector>
#include <cmath>
#include <random>
#include <cassert>

// ================================
// Tensor4D (B, T, H, D)
// ================================
struct Tensor4D {
    int B, T, H, D;
    std::vector<float> data;

    Tensor4D(int b, int t, int h, int d)
        : B(b), T(t), H(h), D(d),
          data(b * t * h * d, 0.0f) {}

    inline float& at(int b, int t, int h, int d) {
        return data[((b * T + t) * H + h) * D + d];
    }

    inline float at(int b, int t, int h, int d) const {
        return data[((b * T + t) * H + h) * D + d];
    }
};

// ================================
// Softmax (last dim)
// ================================
void softmax_last_dim(Tensor4D& x) {
    for (int b = 0; b < x.B; ++b)
    for (int t = 0; t < x.T; ++t)
    for (int h = 0; h < x.H; ++h) {

        float maxv = -1e9f;
        for (int d = 0; d < x.D; ++d)
            maxv = std::max(maxv, x.at(b,t,h,d));

        float sum = 0.0f;
        for (int d = 0; d < x.D; ++d) {
            float e = std::exp(x.at(b,t,h,d) - maxv);
            x.at(b,t,h,d) = e;
            sum += e;
        }
        for (int d = 0; d < x.D; ++d)
            x.at(b,t,h,d) /= sum;
    }
}

// ================================
// Simple Attention (Q=K=V)
// ================================
Tensor4D attention4d(const Tensor4D& x) {
    Tensor4D out(x.B, x.T, x.H, x.D);

    for (int b = 0; b < x.B; ++b)
    for (int h = 0; h < x.H; ++h)
    for (int t = 0; t < x.T; ++t)
    for (int d = 0; d < x.D; ++d) {

        float sum = 0.0f;
        for (int tp = 0; tp < x.T; ++tp) {
            float dot = 0.0f;
            for (int k = 0; k < x.D; ++k)
                dot += x.at(b,tp,h,k) * x.at(b,t,h,k);

            float w = std::exp(dot / std::sqrt((float)x.D));
            sum += w * x.at(b,tp,h,d);
        }
        out.at(b,t,h,d) = sum / x.T;
    }
    return out;
}

// ================================
// FFN
// ================================
Tensor4D ffn4d(const Tensor4D& x) {
    Tensor4D y = x;
    for (auto& v : y.data)
        v = std::tanh(v);
    return y;
}

// ================================
// Training demo
// ================================
int main() {
    constexpr int B = 1;
    constexpr int T = 4;
    constexpr int H = 2;
    constexpr int D = 8;

    Tensor4D x(B, T, H, D);

    std::mt19937 rng(42);
    std::uniform_real_distribution<float> dist(-0.5f, 0.5f);

    for (auto& v : x.data)
        v = dist(rng);

    std::cout << "Forward start\n";

    auto attn = attention4d(x);
    softmax_last_dim(attn);
    auto out = ffn4d(attn);

    std::cout << "Output sample:\n";
    for (int d = 0; d < D; ++d)
        std::cout << out.at(0,0,0,d) << " ";
    std::cout << "\nDone\n";
}