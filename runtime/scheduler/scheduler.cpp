#include "runtime/scheduler/scheduler.h"

Scheduler::Scheduler(size_t cpu_threads)
    : cpu_pool(cpu_threads) {}

void Scheduler::submit(std::function<void()> task) {
    cpu_pool.submit(std::move(task));
}