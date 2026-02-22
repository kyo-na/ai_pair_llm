#include "model/mini_AI.h"

MiniAI::MiniAI(int layers, int dim, int vocab_size)
    : vocab_head(dim, vocab_size), dim(dim)
{
    for (int i = 0; i < layers; ++i)
        blocks.emplace_back(dim);
}

Tensor4D MiniAI::forward(const Tensor4D& x) {
    Tensor4D h = x;

    for (auto& b : blocks)
        h = b.forward(h);

    // vocab projection
    h = vocab_head.forward(h);

    return h;
}

void MiniAI::step(float lr) {
    for (auto& b : blocks)
        b.step(lr);

    vocab_head.step(lr);
}