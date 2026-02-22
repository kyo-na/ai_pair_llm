#pragma once
#include <vector>
#include "runtime/scheduler/scheduler.h"
#include "runtime/threads/worker_pool.h"

// runtime 全体を駆動する制御ループ
class RuntimeLoop {
public:
    RuntimeLoop(
        Scheduler* scheduler,
        WorkerPool* workers,
        const std::vector<EngineBase*>& engines
    );

    // タスク投入
    void submit(const MiniLLMTask& task);

    // 結果回収（非ブロッキング）
    void poll_results();

private:
    Scheduler* scheduler;
    WorkerPool* workers;
    std::vector<EngineBase*> engines;
};