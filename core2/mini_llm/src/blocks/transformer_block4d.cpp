#include "blocks/transformer_block4d.h"

Tensor4D TransformerBlock4D::forward(const Tensor4D& x){
    last_x = x;

    // Attention block
    Tensor4D h1 = ln1.forward(x);
    Tensor4D a  = attn.forward(h1);

    for(size_t i=0;i<a.data.size();i++){
        a.data[i] += x.data[i];   // residual
    }

    // FFN block（Linearのみ）
    Tensor4D h2 = ln2.forward(a);
    Tensor4D y  = ffn.forward(h2);

    for(size_t i=0;i<y.data.size();i++){
        y.data[i] += a.data[i];   // residual
    }
    return y;
}

Tensor4D TransformerBlock4D::backward(const Tensor4D& x, const Tensor4D& dout){
    // FFN backward
    Tensor4D d_ffn = ffn.backward(dout);
    Tensor4D d_ln2 = ln2.backward(d_ffn);

    Tensor4D da(x.B, x.T, x.H, x.D);
    for(size_t i=0;i<da.data.size();i++){
        da.data[i] = dout.data[i] + d_ln2.data[i];
    }

    // Attention backward
    Tensor4D d_attn = attn.backward(da);
    Tensor4D d_ln1  = ln1.backward(d_attn);

    Tensor4D dx(x.B, x.T, x.H, x.D);
    for(size_t i=0;i<dx.data.size();i++){
        dx.data[i] = da.data[i] + d_ln1.data[i];
    }
    return dx;
}

void TransformerBlock4D::step(float lr){
    ln1.step(lr);
    attn.step(lr);
    ln2.step(lr);
    ffn.step(lr);
}