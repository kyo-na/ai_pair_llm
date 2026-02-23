#include "layers/embedding4d.h"
#include <random>
#include <algorithm>

static float frand_uniform(float a, float b) {
    static std::mt19937 rng(1234);
    std::uniform_real_distribution<float> dist(a, b);
    return dist(rng);
}

Embedding4D::Embedding4D(int vocab_size, int dim_) : vocab(vocab_size), dim(dim_) {
    W.resize((size_t)vocab * dim);
    dW.assign((size_t)vocab * dim, 0.0f);

    // small init
    float scale = 0.02f;
    for (auto &x : W) x = frand_uniform(-scale, scale);
}

Tensor4D Embedding4D::forward_ids(int B, int T, const std::vector<int32_t>& ids) {
    last_ids = ids;
    Tensor4D out(B, T, 1, dim);
    std::fill(out.data.begin(), out.data.end(), 0.0f);

    for (int b = 0; b < B; b++) {
        for (int t = 0; t < T; t++) {
            int idx = b*T + t;
            int32_t id = ids[idx];
            if (id < 0) id = 0;
            if (id >= vocab) id = vocab - 1;

            float* dst = &out.data[((b*out.T + t)*out.H + 0)*out.D];
            const float* src = &W[(size_t)id * dim];
            for (int d = 0; d < dim; d++) dst[d] = src[d];
        }
    }
    return out;
}

void Embedding4D::backward(const Tensor4D& dout) {
    // accumulate gradients into dW
    for (int b = 0; b < dout.B; b++) {
        for (int t = 0; t < dout.T; t++) {
            int idx = b*dout.T + t;
            int32_t id = last_ids[idx];
            if (id < 0) id = 0;
            if (id >= vocab) id = vocab - 1;

            const float* g = &dout.grad[((b*dout.T + t)*dout.H + 0)*dout.D];
            float* dw = &dW[(size_t)id * dim];
            for (int d = 0; d < dim; d++) dw[d] += g[d];
        }
    }
}

void Embedding4D::step(float lr) {
    for (size_t i = 0; i < W.size(); i++) {
        W[i] -= lr * dW[i];
        dW[i] = 0.0f;
    }
}