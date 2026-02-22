
#include "backend.h"
#include <vector>
#include <cstdlib>

struct DummyModel {
    int state = 0;
    void reset(){ state = 0; }
    int forward(int t){
        state++;
        if(state > 20) return -1; // eos
        return (t + 1) % 128;
    }
    bool is_eos(int t){ return t < 0; }
};

struct CPUBackend : Backend {
    DummyModel model;
    bool done = false;

    void reset() override {
        model.reset();
        done = false;
    }

    int step(int token) override {
        int next = model.forward(token);
        if(model.is_eos(next)) done = true;
        return next;
    }

    bool finished() const override {
        return done;
    }
};

extern "C" Backend* create_backend(){
    return new CPUBackend();
}
