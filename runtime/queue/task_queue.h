#pragma once
#include <queue>
#include <mutex>
#include <condition_variable>

template<typename T>
class TaskQueue {
public:
    void push(const T& task) {
        std::unique_lock<std::mutex> lock(mtx);
        q.push(task);
        cv.notify_one();
    }

    bool pop(T& out) {
        std::unique_lock<std::mutex> lock(mtx);
        cv.wait(lock, [&]{ return !q.empty() || stopped; });

        if (q.empty()) return false;

        out = q.front();
        q.pop();
        return true;
    }

    void stop() {
        std::unique_lock<std::mutex> lock(mtx);
        stopped = true;
        cv.notify_all();
    }

private:
    std::queue<T> q;
    std::mutex mtx;
    std::condition_variable cv;
    bool stopped = false;
};