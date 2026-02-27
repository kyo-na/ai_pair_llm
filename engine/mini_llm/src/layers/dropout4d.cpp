#include "layers/dropout4d.h"
#include <random>

Dropout4D::Dropout4D(float p) : p_(p) {}

Tensor4D Dropout4D::forward(const Tensor4D& x, bool train_mode)
{
    if (!train_mode) return x;

    last_mask_ = Tensor4D(x.B,x.T,x.H,x.D);
    Tensor4D out(x.B,x.T,x.H,x.D);

    std::mt19937 rng(42);
    std::uniform_real_distribution<float> dist(0.0f,1.0f);

    for(int b=0;b<x.B;++b)
    for(int t=0;t<x.T;++t)
    for(int h=0;h<x.H;++h)
    for(int d=0;d<x.D;++d)
    {
        float keep = (dist(rng) > p_) ? 1.0f : 0.0f;
        last_mask_.at(b,t,h,d) = keep;
        out.at(b,t,h,d) = x.at(b,t,h,d) * keep;
    }

    return out;
}

Tensor4D Dropout4D::backward(const Tensor4D& grad)
{
    Tensor4D out(grad.B,grad.T,grad.H,grad.D);

    for(int b=0;b<grad.B;++b)
    for(int t=0;t<grad.T;++t)
    for(int h=0;h<grad.H;++h)
    for(int d=0;d<grad.D;++d)
        out.at(b,t,h,d) = grad.at(b,t,h,d) * last_mask_.at(b,t,h,d);

    return out;
}