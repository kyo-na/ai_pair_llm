#include "layers/linear_vocab4d.h"
#include <cmath>
#include <cstdlib>

static float kaiming(int fan_in){
    return std::sqrt(2.0f/fan_in)*((float)rand()/RAND_MAX-0.5f);
}

LinearVocab4D::LinearVocab4D(int d_model,int vocab)
: d_model_(d_model), vocab_(vocab),
  W_(1,1,d_model,vocab), b_(1,1,1,vocab)
{
    for(auto& w:W_.data) w=kaiming(d_model_);
    b_.zero();
}

Tensor4D LinearVocab4D::forward(const Tensor4D& x){
    last_x_=x;
    Tensor4D out(x.B,x.T,1,vocab_);

    for(int b=0;b<x.B;++b)
    for(int t=0;t<x.T;++t)
    for(int v=0;v<vocab_;++v){
        float s=b_.at(0,0,0,v);
        for(int d=0;d<d_model_;++d)
            s+=x.at(b,t,0,d)*W_.at(0,0,d,v);
        out.at(b,t,0,v)=s;
    }
    return out;
}

Tensor4D LinearVocab4D::backward(const Tensor4D& grad_out){
    Tensor4D grad_x(last_x_.B,last_x_.T,1,d_model_);
    W_.zero_grad(); b_.zero_grad();

    for(int b=0;b<grad_out.B;++b)
    for(int t=0;t<grad_out.T;++t)
    for(int v=0;v<vocab_;++v){
        float g=grad_out.at(b,t,0,v);
        b_.grad_at(0,0,0,v)+=g;
        for(int d=0;d<d_model_;++d){
            W_.grad_at(0,0,d,v)+=last_x_.at(b,t,0,d)*g;
            grad_x.at(b,t,0,d)+=W_.at(0,0,d,v)*g;
        }
    }
    return grad_x;
}

std::vector<Tensor4D*> LinearVocab4D::parameters(){
    return { &W_, &b_ };
}