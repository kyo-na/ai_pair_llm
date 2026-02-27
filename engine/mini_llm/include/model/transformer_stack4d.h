#pragma once
#include <vector>
#include "blocks/transformer_block4d.h"

class TransformerStack4D {
public:
    TransformerStack4D(
        int layers,
        int heads,
        int head_dim,
        int ffn_hidden);

    Tensor4D forward(const Tensor4D& x);

private:
    std::vector<TransformerBlock4D> blocks_;
};