#include <thread>
#include <vector>
#include <queue>
#include <mutex>
#include <condition_variable>
#include <atomic>

#include "model/mini_AI.h"
#include "world/world_model.h"
#include "tensor4d.h"

struct TransformerTask {
    Tensor4D input;
};

class CPUMonitor {
private:
    std::queue<TransformerTask> task_queue;
    std::mutex mtx;
    std::condition_variable cv;
    std::atomic<bool> running{true};

    std::vector<std::thread> cpu_workers;

public:
    CPUMonitor() {
        unsigned n = std::thread::hardware_concurrency();
        for (unsigned i = 0; i < n; ++i) {
            cpu_workers.emplace_back(&CPUMonitor::cpu_worker, this);
        }
    }

    ~CPUMonitor() {
        running = false;
        cv.notify_all();
        for (auto& t : cpu_workers)
            t.join();
    }

    void enqueue(const TransformerTask& task) {
        {
            std::lock_guard<std::mutex> lock(mtx);
            task_queue.push(task);
        }
        cv.notify_one();
    }

private:
    void cpu_worker() {
        MiniAI ai(2, 64);

        while (running) {
            TransformerTask task;

            {
                std::unique_lock<std::mutex> lock(mtx);
                cv.wait(lock, [&] {
                    return !task_queue.empty() || !running;
                });

                if (!running) return;

                task = task_queue.front();
                task_queue.pop();
            }

            // ---- 実行 ----
            Tensor4D h = ai.forward(task.input);

            // 必要ならWorld
            WorldModel world(
                task.input.B,
                task.input.T,
                task.input.H,
                task.input.D
            );
            world.init();
            world.inject_observation(h);
            world.step_forward();
        }
    }
};