#include <iostream>
#include <string>
#include <algorithm>
#include <vector>

#include "engine/mini_llm/include/model/mini_AI.h"
#include "engine/mini_llm/include/world/world_model.h"
#include "engine/mini_llm/include/tensor4d.h"

#include "engine/mini_llm/tokenizer/tokenizer.h"
#include "engine/mini_llm/tokenizer/vocab.h"
#include "engine/mini_llm/tokenizer/utf8.h"

using namespace mini_llm;

int argmax(const std::vector<float>& v) {
    int best = 0;
    for (int i = 1; i < (int)v.size(); ++i)
        if (v[i] > v[best]) best = i;
    return best;
}

int main() {
    const int B = 1;
    const int FIXED_T = 32;
    const int H = 1;
    const int D = 64;

    Tokenizer tokenizer;
    Vocab vocab;
    int vocab_size = vocab.size();

    MiniAI ai(2, D, vocab_size);
    WorldModel world(B, FIXED_T, H, D);
    world.init();

    std::cout << "ai_pair_llm ready. exitで終了\n";

    while (true) {
        std::cout << "あなた> ";
        std::string input;
        std::getline(std::cin, input);
        if (input == "exit") break;

        auto cps = tokenizer.encode(input);

        Tensor4D x(B, FIXED_T, H, D);
        std::fill(x.data.begin(), x.data.end(), 0.0f);

        int len = std::min((int)cps.size(), FIXED_T);
        for (int t = 0; t < len; ++t) {
            int id = vocab.token_to_id(cps[t]);
            x.data[t * D] = float(id) / vocab_size;
        }

        std::cout << "AI> ";

        for (int step = 0; step < 20; ++step) {
            Tensor4D logits = ai.forward(x);

            int last_t = len > 0 ? len - 1 : 0;

            // ★ 正しい base 計算
            int base = last_t * vocab_size;

            std::vector<float> row(vocab_size);
            for (int i = 0; i < vocab_size; ++i)
                row[i] = logits.data[base + i];

            int next_id = argmax(row);

            char32_t cp = vocab.id_to_token(next_id);
            std::string out = codepoint_to_utf8(cp);
            std::cout << out;

            if (len < FIXED_T) {
                x.data[len * D] = float(next_id) / vocab_size;
                len++;
            }
        }

        std::cout << "\n";
    }

    return 0;
}