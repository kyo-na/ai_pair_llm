#include <iostream>
#include <vector>

#include "tokenizer.h"
#include "vocab.h"

#include "model/mini_AI.h"
#include "world/world_model.h"
#include "tensor4d.h"

using namespace mini_llm;

void cpu_run_once() {
    Tokenizer tokenizer;
    Vocab vocab; // ロードなし（未実装）

    auto cps = tokenizer.encode("こんにちは");

    Tensor4D x(1, 8, 1, 64);
    for (auto& v : x.data) v = 0.2f;

    MiniAI ai(2, 64);

    WorldModel world(1, 8, 1, 64);
    world.init();

    Tensor4D h = ai.forward(x);
    world.inject_observation(h);
    world.step_forward();

    std::cout << "cpu_run_once OK\n";
}