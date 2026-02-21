#include "model/mini_AI.h"

Tensor4D MiniAI::forward(const Tensor4D& x){
    block_inputs.clear();
    Tensor4D h = x;
    for(auto& b: blocks){
        block_inputs.push_back(h);
        h = b.forward(h);
    }
    return h;
}

Tensor4D MiniAI::backward(const Tensor4D& dout){
    Tensor4D dh = dout;
    for(int i=(int)blocks.size()-1;i>=0;i--){
        dh = blocks[i].backward(block_inputs[i], dh);
    }
    return dh;
}

void MiniAI::step(float lr){
    for(auto& b : blocks){
        b.step(lr);
    }
}