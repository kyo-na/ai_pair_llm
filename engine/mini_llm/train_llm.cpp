// train_llm.cpp
// Full-scratch Japanese pretraining (LEGAL / NO external data)
// C++17 / STL only

#include <iostream>
#include <vector>
#include <string>
#include <unordered_map>
#include <random>
#include <cmath>
#include <cstdint>
#include <cassert>

// =============================
// UTF-8 → Unicode codepoint
// =============================
std::vector<uint32_t> utf8_to_codepoints(const std::string& s) {
    std::vector<uint32_t> cps;
    for (size_t i = 0; i < s.size();) {
        uint8_t c = s[i];
        if ((c & 0x80) == 0) {
            cps.push_back(c);
            i += 1;
        } else if ((c & 0xE0) == 0xC0) {
            cps.push_back(((c & 0x1F) << 6) | (s[i+1] & 0x3F));
            i += 2;
        } else if ((c & 0xF0) == 0xE0) {
            cps.push_back(((c & 0x0F) << 12) |
                          ((s[i+1] & 0x3F) << 6) |
                           (s[i+2] & 0x3F));
            i += 3;
        } else {
            cps.push_back(((c & 0x07) << 18) |
                          ((s[i+1] & 0x3F) << 12) |
                          ((s[i+2] & 0x3F) << 6) |
                           (s[i+3] & 0x3F));
            i += 4;
        }
    }
    return cps;
}

// =============================
// 合法・完全自作 日本語合成データ
// =============================
std::vector<std::string> generate_synthetic_japanese(int n) {
    std::vector<std::string> subjects = {"私", "彼", "彼女", "猫", "犬"};
    std::vector<std::string> objects  = {"リンゴ", "バナナ", "水", "本"};
    std::vector<std::string> verbs    = {"食べる", "見る", "飲む", "読む"};
    std::vector<std::string> times    = {"今日", "昨日", "明日"};

    std::mt19937 rng(1234);
    std::vector<std::string> out;

    for (int i = 0; i < n; ++i) {
        std::string s =
            times[rng() % times.size()] + "、" +
            subjects[rng() % subjects.size()] + "は" +
            objects[rng() % objects.size()] + "を" +
            verbs[rng() % verbs.size()] + "。";
        out.push_back(s);
    }
    return out;
}

// =============================
// Hash Embedding
// =============================
constexpr int EMBED_DIM = 64;
constexpr int HASH_SIZE = 1 << 16; // 65536

struct HashEmbedding {
    std::vector<float> w;
    std::vector<float> grad;

    HashEmbedding() : w(HASH_SIZE * EMBED_DIM), grad(w.size()) {
        std::mt19937 rng(1);
        std::uniform_real_distribution<float> d(-0.01f, 0.01f);
        for (auto& x : w) x = d(rng);
    }

    float* forward(uint32_t token) {
        return &w[(token % HASH_SIZE) * EMBED_DIM];
    }

    float* backward(uint32_t token) {
        return &grad[(token % HASH_SIZE) * EMBED_DIM];
    }

    void zero_grad() {
        std::fill(grad.begin(), grad.end(), 0.0f);
    }
};

// =============================
// Linear layer
// =============================
struct Linear {
    int in, out;
    std::vector<float> w, b, gw, gb;

    Linear(int i, int o)
        : in(i), out(o),
          w(i * o), b(o),
          gw(i * o), gb(o) {
        std::mt19937 rng(2);
        std::uniform_real_distribution<float> d(-0.01f, 0.01f);
        for (auto& x : w) x = d(rng);
    }

    std::vector<float> forward(const float* x) {
        std::vector<float> y(out);
        for (int o = 0; o < out; ++o) {
            float s = b[o];
            for (int i = 0; i < in; ++i)
                s += w[o*in + i] * x[i];
            y[o] = s;
        }
        return y;
    }

    void backward(const float* x, const std::vector<float>& dy, float* dx) {
        for (int o = 0; o < out; ++o) {
            gb[o] += dy[o];
            for (int i = 0; i < in; ++i) {
                gw[o*in + i] += dy[o] * x[i];
                dx[i] += w[o*in + i] * dy[o];
            }
        }
    }

    void zero_grad() {
        std::fill(gw.begin(), gw.end(), 0.0f);
        std::fill(gb.begin(), gb.end(), 0.0f);
    }
};

// =============================
// Softmax + CrossEntropy
// =============================
float softmax_xent(std::vector<float>& logits, uint32_t target, std::vector<float>& dlogits) {
    float maxv = *std::max_element(logits.begin(), logits.end());
    float sum = 0.0f;
    for (auto& v : logits) sum += std::exp(v - maxv);

    float loss = -std::log(std::exp(logits[target] - maxv) / sum);

    dlogits.resize(logits.size());
    for (size_t i = 0; i < logits.size(); ++i) {
        float p = std::exp(logits[i] - maxv) / sum;
        dlogits[i] = p - (i == target ? 1.0f : 0.0f);
    }
    return loss;
}

// =============================
// SGD Optimizer
// =============================
void sgd(std::vector<float>& w, std::vector<float>& g, float lr) {
    for (size_t i = 0; i < w.size(); ++i) {
        w[i] -= lr * g[i];
    }
}

// =============================
// MAIN: 本当に学習する
// =============================
int main() {
    auto corpus = generate_synthetic_japanese(2000);

    HashEmbedding embed;
    Linear linear(EMBED_DIM, HASH_SIZE);

    float lr = 0.05f;

    for (int epoch = 0; epoch < 5; ++epoch) {
        float total_loss = 0.0f;
        int steps = 0;

        for (auto& s : corpus) {
            auto tokens = utf8_to_codepoints(s);
            if (tokens.size() < 2) continue;

            for (size_t t = 0; t + 1 < tokens.size(); ++t) {
                embed.zero_grad();
                linear.zero_grad();

                float* x = embed.forward(tokens[t]);
                auto logits = linear.forward(x);

                std::vector<float> dlogits;
                float loss = softmax_xent(logits, tokens[t+1] % HASH_SIZE, dlogits);

                std::vector<float> dx(EMBED_DIM, 0.0f);
                linear.backward(x, dlogits, dx.data());

                float* gemb = embed.backward(tokens[t]);
                for (int i = 0; i < EMBED_DIM; ++i)
                    gemb[i] += dx[i];

                sgd(embed.w, embed.grad, lr);
                sgd(linear.w, linear.gw, lr);
                sgd(linear.b, linear.gb, lr);

                total_loss += loss;
                steps++;
            }
        }

        std::cout << "Epoch " << epoch
                  << " loss=" << (total_loss / steps) << std::endl;
    }

    return 0;
}