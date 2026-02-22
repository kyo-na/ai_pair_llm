
#include "backend.h"
#include <iostream>
#include <vector>

extern "C" Backend* create_backend();

int main(){
    Backend* backend = create_backend();

    std::vector<int> prompt = { 'H','e','l','l','o','\n' };
    backend->reset();

    for(int t : prompt)
        backend->step(t); // prefill

    int cur = prompt.back();
    while(!backend->finished()){
        int next = backend->step(cur);
        if(next < 0) break;
        std::cout << char(next);
        cur = next;
    }
    std::cout << std::endl;
    return 0;
}
