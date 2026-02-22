#include <thread>
#include <chrono>
#include <iostream>
#include "engine/cpu_engine/cpu_engine.h"
#include "runtime/threads/worker_pool.h"

int main() {
    CPUEngine cpu;
    WorkerPool pool(1, &cpu);

    MiniLLMTask task{};
    task.training = false;
    task.input_text = "hello world";
    task.on_done = [](std::string out){
        std::cout << "OUT: " << out << "\n";
    };

    pool.submit(task);

    // ちゃんと待つ（本当は future が理想）
    std::this_thread::sleep_for(std::chrono::milliseconds(200));
    pool.stop();
    return 0;
}