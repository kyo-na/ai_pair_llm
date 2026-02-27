#include <iostream>
#include <vector>
#include <cstdint>

#include "model/transformer_stack_infer4d.h"
#include "decode/infer_sampling.h"

int main()
{
    std::cout << "infer start\n";

    int vocab=16;
    int layers=2;
    int H=1;
    int D=16;
    int maxT=256;

    TransformerStackInfer4D model(layers, vocab, H, D, maxT);
    InferContext ctx(8*1024*1024); // 8MB arena

    InferSamplingConfig cfg;
    cfg.temperature = 1.0f;
    cfg.top_k = 8;
    cfg.top_p = 0.9f;
    cfg.repetition_penalty = 1.1f;
    cfg.rng_seed = 1234;

    int32_t token = 1; // BOS想定
    std::vector<int32_t> recent;

    for(int step=0; step<50; ++step){
        auto logits = model.forward_step(token, ctx);
        int next = sample_next_token(logits, recent, cfg);

        recent.push_back(token);
        if((int)recent.size() > 64) recent.erase(recent.begin());

        std::cout << next << " ";
        token = next;
    }

    std::cout << "\ninfer done\n";
    return 0;
}