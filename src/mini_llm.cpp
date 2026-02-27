#include "mini_llm.h"
#include <fstream>
#include <iostream>
#include <cmath>

MiniLLM::MiniLLM(size_t vocab, size_t c)
    : C(c), W_embed(vocab * c, 0.01f) {}

void MiniLLM::load_weights(const char* emb, const char*) {
    std::ifstream f(emb, std::ios::binary);
    if (!f) {
        std::cerr << "failed to open " << emb << "\n";
        return;
    }
    f.read(reinterpret_cast<char*>(W_embed.data()),
           W_embed.size() * sizeof(float));
}

void MiniLLM::step(
    World4D& world,
    size_t t,
    const std::vector<unsigned>& input_ids
) {
    for (size_t d = 0; d < world.D; ++d) {
        for (size_t c = 0; c < world.C; ++c) {
            float sum = 0.f;
            for (auto id : input_ids)
                sum += W_embed[id * C + c];

            float prev = (t > 0)
                ? world.at(0, t - 1, d, c)
                : 0.f;

            world.at(0, t, d, c) =
                std::tanh(sum + 0.5f * prev);
        }
    }
}