#include <iostream>
#include <vector>
#include "world4d.h"
#include "mini_llm.h"

int main() {
    constexpr size_t B = 1, T = 128, D = 4, C = 16, VOCAB = 256;

    World4D world(B, T, D, C);
    MiniLLM mini(VOCAB, C);

    mini.load_weights(
    "C:\\Users\\spenc\\Downloads\\ai_pair_llm\\engine\\mini_llm\\train\\weights_emb.bin",
    "C:\\Users\\spenc\\Downloads\\ai_pair_llm\\engine\\mini_llm\\train\\weights_proj.bin"
);

    for (size_t t = 0; t < T; ++t) {
        std::vector<unsigned> input_ids = {
            static_cast<unsigned>(t % VOCAB)
        };

        mini.step(world, t, input_ids);

        float energy = 0.0f;
        for (size_t c = 0; c < C; ++c)
            energy += world.at(0, t, 0, c);

        std::cout << "t=" << t << " energy=" << energy << "\n";
    }
}