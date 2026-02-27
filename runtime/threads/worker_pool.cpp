#include <thread>
#include <vector>

struct WorkerPool {
    std::vector<std::thread> workers;

    template<typename Fn>
    void run(Fn fn){
        workers.emplace_back(fn);
    }

    void join_all(){
        for(auto& w : workers){
            if(w.joinable()) w.join();
        }
    }
};