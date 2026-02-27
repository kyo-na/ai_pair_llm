#include "model/transformer_stack4d.h"

TransformerStack4D::TransformerStack4D(
    int layers,
    int heads,
    int head_dim,
    int ffn_hidden)
{
    for(int i=0;i<layers;++i)
        blocks_.emplace_back(
            heads,
            head_dim,
            ffn_hidden);
}

Tensor4D TransformerStack4D::forward(
    const Tensor4D& x)
{
    Tensor4D out = x;

    for(auto& block : blocks_)
        out = block.forward(out);

    return out;
}