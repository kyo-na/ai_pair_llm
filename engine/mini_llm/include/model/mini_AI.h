#pragma once
#include "tensor4d.h"
#include "blocks/transformer_block4d.h"
#include "layers/linear4d.h"
#include <vector>

class MiniAI {
public:
    MiniAI(int layers, int dim, int vocab_size);

    Tensor4D forward(const Tensor4D& x);
    void step(float lr);

private:
    std::vector<TransformerBlock4D> blocks;
    Linear4D vocab_head;   // ← 追加
    int dim;
};