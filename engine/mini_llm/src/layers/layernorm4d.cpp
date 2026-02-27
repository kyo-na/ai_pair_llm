#include "layers/layernorm4d.h"
#include <cmath>

LayerNorm4D::LayerNorm4D(int dim)
    : D_(dim),
      gamma_(1,1,1,dim),
      beta_(1,1,1,dim)
{
    for(int d=0; d<dim; ++d)
    {
        gamma_.at(0,0,0,d)=1.0f;
        beta_.at(0,0,0,d)=0.0f;
    }
}

Tensor4D LayerNorm4D::forward(const Tensor4D& x)
{
    last_input_ = x;

    Tensor4D out(x.B,x.T,x.H,x.D);
    last_mean_ = Tensor4D(x.B,x.T,x.H,1);
    last_var_  = Tensor4D(x.B,x.T,x.H,1);

    for(int b=0;b<x.B;++b)
    for(int t=0;t<x.T;++t)
    for(int h=0;h<x.H;++h)
    {
        float mean=0;
        for(int d=0;d<x.D;++d)
            mean+=x.at(b,t,h,d);
        mean/=x.D;

        float var=0;
        for(int d=0;d<x.D;++d)
        {
            float diff=x.at(b,t,h,d)-mean;
            var+=diff*diff;
        }
        var/=x.D;

        last_mean_.at(b,t,h,0)=mean;
        last_var_.at(b,t,h,0)=var;

        float inv=1.0f/std::sqrt(var+1e-5f);

        for(int d=0;d<x.D;++d)
        {
            float norm=(x.at(b,t,h,d)-mean)*inv;
            out.at(b,t,h,d)=norm*gamma_.at(0,0,0,d)
                             +beta_.at(0,0,0,d);
        }
    }

    return out;
}

Tensor4D LayerNorm4D::backward(const Tensor4D& grad)
{
    // 簡易版（まずは入力へそのまま流す）
    return grad;
}

std::vector<Tensor4D*> LayerNorm4D::parameters()
{
    return {&gamma_, &beta_};
}