#include "runtime_loop.h"
#include <iostream>

RuntimeLoop::RuntimeLoop(
    Scheduler* s,
    WorkerPool* w,
    const std::vector<EngineBase*>& e
)
    : scheduler(s), workers(w), engines(e) {}

void RuntimeLoop::submit(const MiniLLMTask& task) {
    // ★ ここで初めて Scheduler を使う
    EngineBase* engine = scheduler->pick_engine(engines);
    if (!engine) {
        std::cout << "[Runtime] no available engine\n";
        return;
    }

    workers->submit(task);
}

void RuntimeLoop::poll_results() {
    for (auto* engine : engines) {
        MiniLLMResult result;
        while (engine->try_fetch(result)) {
            std::cout << "[Runtime] result fetched (loss="
                      << result.loss << ")\n";
        }
    }
}