#include "blocks/transformer_block4d.h"

TransformerBlock4D::TransformerBlock4D(
    int heads,
    int head_dim,
    int ffn_hidden)
: attn_(heads, head_dim),
  norm1_(1, heads*head_dim),
  norm2_(1, heads*head_dim),
  ffn_(1, heads*head_dim, ffn_hidden)
{}

Tensor4D TransformerBlock4D::forward(
    const Tensor4D& input)
{
    Tensor4D x = input;

    // ---- Norm1 ----
    Tensor4D n1 = norm1_.forward(x);

    // ---- Attention ----
    Tensor4D attn_out = attn_.forward(n1);

    // ---- Residual ----
    for(int d=0; d<x.D; ++d)
        x.at(0,0,0,d) += attn_out.at(0,0,0,d);

    // ---- Norm2 ----
    Tensor4D n2 = norm2_.forward(x);

    // ---- FFN ----
    Tensor4D ffn_out = ffn_.forward(n2);

    // ---- Residual ----
    for(int d=0; d<x.D; ++d)
        x.at(0,0,0,d) += ffn_out.at(0,0,0,d);

    return x;
}