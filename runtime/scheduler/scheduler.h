#pragma once
#include "runtime/threads/worker_pool.h"

class Scheduler {
public:
    explicit Scheduler(size_t cpu_threads);

    void submit(std::function<void()> task);

private:
    WorkerPool cpu_pool;
};