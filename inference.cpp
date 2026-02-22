// inference.cpp
#include <iostream>
#include <string>
#include <vector>

// ★ これが足りなかった
#include <fstream>

#include "engine/mini_llm/tokenizer/tokenizer.h"
#include "engine/mini_llm/tokenizer/vocab.h"
#include "engine/mini_llm/tokenizer/utf8.h"

#include "engine/mini_llm/model/embedding.h"
#include "engine/mini_llm/model/linear.h"

using namespace mini_llm;

int main() {
    Tokenizer tokenizer;
    Vocab vocab;

    // mini_llm と同じ次元
    constexpr int DIM = 64;

    // ★ load はしない（現時点では）
    Embedding emb(4096, DIM);
    Linear proj(DIM, DIM);

    std::cout << "ai_pair_llm ready. 日本語で入力してください（exitで終了）\n";

    while (true) {
        std::cout << "あなた> ";
        std::string input;
        std::getline(std::cin, input);
        if (input == "exit") break;

        auto cps = tokenizer.encode(input);

        std::vector<char32_t> context = cps;

        std::cout << "AI> ";

        for (int step = 0; step < 16; ++step) {
            if (context.empty()) break;

            int last_id = vocab.token_to_id(context.back());
            auto x = emb.forward(last_id);
            auto y = proj.forward(x);

            // 最近傍トークン探索（最小）
            int best = 0;
            float best_d = 1e30f;

            for (int i = 0; i < vocab.size(); ++i) {
                auto t = emb.forward(i);
                float d = 0.0f;
                for (int k = 0; k < DIM; ++k) {
                    float diff = y[k] - t[k];
                    d += diff * diff;
                }
                if (d < best_d) {
                    best_d = d;
                    best = i;
                }
            }

            char32_t cp = vocab.id_to_token(best);
            std::cout << codepoint_to_utf8(cp);
            context.push_back(cp);
        }

        std::cout << "\n";
    }

    return 0;
}