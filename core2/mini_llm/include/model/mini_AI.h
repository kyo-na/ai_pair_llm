#pragma once
#include <vector>
#include "blocks/transformer_block4d.h"

struct MiniAI {
    std::vector<TransformerBlock4D> blocks;
    std::vector<Tensor4D> block_inputs;

    MiniAI(int layers, int dim){
        for(int i=0;i<layers;i++)
            blocks.emplace_back(dim);
    }

    Tensor4D forward(const Tensor4D& x);
    Tensor4D backward(const Tensor4D& dout);
    void step(float lr);
};