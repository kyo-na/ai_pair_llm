#include "runtime/threads/worker_pool.h"

WorkerPool::WorkerPool(size_t num_threads) : running(true) {
    for (size_t i = 0; i < num_threads; ++i) {
        workers.emplace_back(&WorkerPool::worker_loop, this);
    }
}

WorkerPool::~WorkerPool() {
    stop();
}

void WorkerPool::submit(std::function<void()> task) {
    {
        std::lock_guard<std::mutex> lock(mtx);
        tasks.push(std::move(task));
    }
    cv.notify_one();
}

void WorkerPool::stop() {
    running = false;
    cv.notify_all();
    for (auto& t : workers) {
        if (t.joinable()) t.join();
    }
}

void WorkerPool::worker_loop() {
    while (running) {
        std::function<void()> task;
        {
            std::unique_lock<std::mutex> lock(mtx);
            cv.wait(lock, [&] { return !tasks.empty() || !running; });

            if (!running && tasks.empty()) return;

            task = std::move(tasks.front());
            tasks.pop();
        }
        task();
    }
}