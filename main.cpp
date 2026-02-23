#include <iostream>
#include <string>
#include <vector>

#include "../tokenizer/tokenizer.h"
#include "../tokenizer/vocab.h"
#include "../model/embedding.h"
#include "../model/linear.h"

using namespace mini_llm;

int main() {
    std::cout << "mini_llm chat start\n";

    Tokenizer tokenizer;
    Vocab vocab;

    constexpr int DIM = 64;
    Embedding emb(4096, DIM);
    Linear proj(DIM, DIM);

    // 重みロード（学習済み）
    emb.load("weights_emb.bin");
    proj.load("weights_proj.bin");

    while (true) {
        std::string input;
        std::cout << "\n> ";
        std::getline(std::cin, input);
        if (input == "exit") break;

        // 1. tokenize
        auto cps = tokenizer.encode(input);
        if (cps.empty()) continue;

        std::vector<int> ids;
        for (auto cp : cps)
            ids.push_back(vocab.token_to_id(cp));

        // 2. 最後の token から次を予測
        int last_id = ids.back();

        auto x = emb.forward(last_id);
        auto y = proj.forward(x);

        // 3. argmax
        int best = 0;
        float best_v = y[0];
        for (int i = 1; i < (int)y.size(); ++i) {
            if (y[i] > best_v) {
                best_v = y[i];
                best = i;
            }
        }

        // 4. token → char
        char32_t cp = vocab.id_to_token(best);
        std::string out = tokenizer.decode({cp});

        std::cout << out << std::endl;
    }

    return 0;
}