#include "../include/tensor4d.h"
#include <iostream>
#include <random>
#include "../include/tensor4d.h"
#include "../include/adam.h"

// ダミー「学習」
// embedding と projection を Adam で少し動かすだけ

int main() {
    const int vocab = 16;
    const int dim   = 8;

    // Embedding: [1,1,vocab,dim]
    Tensor4D emb(1,1,vocab,dim);

    // Projection: [1,1,dim,vocab]
    Tensor4D proj(1,1,dim,vocab);

    // 初期化
    std::mt19937 rng(0);
    std::uniform_real_distribution<float> dist(-0.1f, 0.1f);

    for (auto& x : emb.data)  x = dist(rng);
    for (auto& x : proj.data) x = dist(rng);

    // ダミー勾配
    for (int step=0; step<10; step++) {
        emb.zero_grad();
        proj.zero_grad();

        for (int i=0;i<emb.size();i++)  emb.grad[i]  = 0.01f;
        for (int i=0;i<proj.size();i++) proj.grad[i] = -0.01f;

        // Adam
        static Adam opt_emb(emb.size(), 1e-2f);
        static Adam opt_proj(proj.size(), 1e-2f);
        opt_emb.update(emb);
        opt_proj.update(proj);
    }

    emb.save("weights_emb.bin");
    proj.save("weights_proj.bin");

    std::cout << "saved: weights_emb.bin / weights_proj.bin\n";
    return 0;
}