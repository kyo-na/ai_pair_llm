#include <iostream>
#include <vector>
#include <random>

#include "../tokenizer/tokenizer.h"
#include "../tokenizer/vocab.h"
#include "../model/embedding.h"
#include "../model/linear.h"
#include "../model/loss.h"

using namespace mini_llm;

// 自動生成FSMデータ
std::vector<std::string> generate_data(int n) {
    std::vector<std::string> states =
        {"idle","walk","run","eat","sleep"};
    std::vector<std::pair<int,int>> rules =
        {{0,1},{1,2},{2,0},{0,3},{3,4},{4,0}};
    std::mt19937 rng(42);

    std::vector<std::string> out;
    for (int i=0;i<n;i++) {
        auto r = rules[rng()%rules.size()];
        out.push_back(states[r.first] + " -> " +
                      states[r.second] + ".");
    }
    return out;
}

int main() {
    Tokenizer tokenizer;
    Vocab vocab;

    constexpr int DIM = 64;
    Embedding emb(4096, DIM); // vocabは動的拡張
    Linear proj(DIM, DIM);

    auto corpus = generate_data(2000);

    float total_loss=0.0f;
    int steps=0;

    for (auto& line : corpus) {
        auto cps = tokenizer.encode(line);
        std::vector<int> ids;
        for (auto cp : cps)
            ids.push_back(vocab.token_to_id(cp));

        for (size_t i=0;i+1<ids.size();++i) {
            auto x = emb.forward(ids[i]);
            auto y = proj.forward(x);
            auto t = emb.forward(ids[i+1]);

            float loss = mse_loss(y,t);
            total_loss += loss;
            steps++;

            auto dy = mse_grad(y,t);
            auto dx = proj.backward(dy);
            emb.backward(ids[i],dx);

            emb.step(0.05f);
            proj.step(0.05f);
        }
    }

    std::cout<<"avg_loss="<<(total_loss/steps)<<std::endl;
}