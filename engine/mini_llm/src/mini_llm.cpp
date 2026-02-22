#include "api/mini_llm_api.h"
#include "world/world_model.h"

MiniLLM::MiniLLM() {
    ai_ = new MiniAI(2, 64);
    world_ = new WorldModel(1, 8, 1, 64);
    world_->init();
}

MiniLLMResult MiniLLM::forward(const MiniLLMTask& task) {
    MiniLLMResult r{};
    if (!task.input) return r;

    Tensor4D h = ai_->forward(*task.input);
    world_->inject_observation(h);
    world_->step_forward();

    r.loss = 0.0f;
    return r;
}