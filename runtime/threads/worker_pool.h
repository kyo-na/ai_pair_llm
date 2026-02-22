#pragma once
#include <vector>
#include <thread>
#include <queue>
#include <functional>
#include <mutex>
#include <condition_variable>
#include <atomic>

class WorkerPool {
public:
    explicit WorkerPool(size_t num_threads);
    ~WorkerPool();

    void submit(std::function<void()> task);
    void stop();

private:
    void worker_loop();

    std::vector<std::thread> workers;
    std::queue<std::function<void()>> tasks;

    std::mutex mtx;
    std::condition_variable cv;
    std::atomic<bool> running;
};