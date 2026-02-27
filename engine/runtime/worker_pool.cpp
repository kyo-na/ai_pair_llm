#include <queue>
#include <thread>
#include <mutex>
#include <condition_variable>
#include <functional>

class WorkerPool {
    std::vector<std::thread> workers;
    std::queue<std::function<void()>> jobs;
    std::mutex mtx;
    std::condition_variable cv;
    bool stop=false;

public:
    WorkerPool(int n){
        for(int i=0;i<n;i++){
            workers.emplace_back([&]{
                while(true){
                    std::function<void()> job;
                    {
                        std::unique_lock lk(mtx);
                        cv.wait(lk,[&]{return stop||!jobs.empty();});
                        if(stop) return;
                        job = jobs.front();
                        jobs.pop();
                    }
                    job();
                }
            });
        }
    }

    void submit(std::function<void()> f){
        {
            std::lock_guard lk(mtx);
            jobs.push(f);
        }
        cv.notify_one();
    }
};