#include "layers/ffn4d.h"
#include <cmath>

static inline float gelu(float x){
    return 0.5f*x*(1.0f+std::tanh(0.79788456f*(x+0.044715f*x*x*x)));
}

static inline float gelu_grad(float x){
    float t=std::tanh(0.79788456f*(x+0.044715f*x*x*x));
    return 0.5f*(1+t)+0.5f*x*(1-t*t)*(0.79788456f*(1+3*0.044715f*x*x));
}

FFN4D::FFN4D(int H,int D)
    :fc1(H,D),fc2(H,D){}

Tensor4D FFN4D::forward(const Tensor4D& x){
    h=fc1.forward(x);
    for(auto& v:h.data) v=gelu(v);
    return fc2.forward(h);
}

Tensor4D FFN4D::backward(const Tensor4D& x,const Tensor4D& dout){
    Tensor4D dh=fc2.backward(h,dout);
    for(size_t i=0;i<h.data.size();i++)
        dh.data[i]*=gelu_grad(h.data[i]);
    return fc1.backward(x,dh);
}
void FFN4D::step(float lr){
    fc1.step(lr);
    fc2.step(lr);
}