#include "layers/swiglu4d.h"
#include <cmath>
#include <cstdlib>

static float init_w(int fan_in){
    return std::sqrt(2.0f/fan_in) *
           ((float)rand()/RAND_MAX - 0.5f);
}

static inline float sigmoid(float x){
    return 1.0f / (1.0f + std::exp(-x));
}

SwiGLU4D::SwiGLU4D(int H,int D,int hidden)
: H_(H), D_(D), hidden_(hidden),
  W1_(1,1,D,hidden),
  W2_(1,1,D,hidden)
{
    for(auto& w:W1_.data) w=init_w(D);
    for(auto& w:W2_.data) w=init_w(D);
}

Tensor4D SwiGLU4D::forward(const Tensor4D& x)
{
    last_x_=x;

    int B=x.B;
    int T=x.T;

    A_=Tensor4D(B,T,H_,hidden_);
    B_=Tensor4D(B,T,H_,hidden_);
    silu_=Tensor4D(B,T,H_,hidden_);

    // xW1 , xW2
    for(int b=0;b<B;++b)
    for(int t=0;t<T;++t)
    for(int h=0;h<H_;++h)
    for(int j=0;j<hidden_;++j){
        float a=0;
        float bb=0;
        for(int i=0;i<D_;++i){
            float xi=x.at(b,t,h,i);
            a  += xi*W1_.at(0,0,i,j);
            bb += xi*W2_.at(0,0,i,j);
        }
        A_.at(b,t,h,j)=a;
        B_.at(b,t,h,j)=bb;

        float s=sigmoid(bb);
        silu_.at(b,t,h,j)=bb*s;
    }

    Tensor4D out(B,T,H_,hidden_);

    for(int b=0;b<B;++b)
    for(int t=0;t<T;++t)
    for(int h=0;h<H_;++h)
    for(int j=0;j<hidden_;++j)
        out.at(b,t,h,j)=
            A_.at(b,t,h,j)*silu_.at(b,t,h,j);

    return out;
}

Tensor4D SwiGLU4D::backward(const Tensor4D& grad_out)
{
    int B=last_x_.B;
    int T=last_x_.T;

    W1_.zero_grad();
    W2_.zero_grad();

    Tensor4D dA(B,T,H_,hidden_);
    Tensor4D dB(B,T,H_,hidden_);
    Tensor4D dx(B,T,H_,D_);

    // ===== y = A * silu(B) =====
    for(int b=0;b<B;++b)
    for(int t=0;t<T;++t)
    for(int h=0;h<H_;++h)
    for(int j=0;j<hidden_;++j){
        float go=grad_out.at(b,t,h,j);

        float silu_val=silu_.at(b,t,h,j);
        float Bval=B_.at(b,t,h,j);

        float sig=sigmoid(Bval);
        float silu_prime=
            sig + Bval*sig*(1.0f-sig);

        dA.at(b,t,h,j)=go*silu_val;
        dB.at(b,t,h,j)=go*A_.at(b,t,h,j)*silu_prime;
    }

    // ===== backward projection =====
    for(int b=0;b<B;++b)
    for(int t=0;t<T;++t)
    for(int h=0;h<H_;++h)
    for(int j=0;j<hidden_;++j)
    for(int i=0;i<D_;++i){

        float xi=last_x_.at(b,t,h,i);

        W1_.grad_at(0,0,i,j)+=
            xi*dA.at(b,t,h,j);

        W2_.grad_at(0,0,i,j)+=
            xi*dB.at(b,t,h,j);

        dx.at(b,t,h,i)+=
            W1_.at(0,0,i,j)*dA.at(b,t,h,j)
           +W2_.at(0,0,i,j)*dB.at(b,t,h,j);
    }

    return dx;
}

std::vector<Tensor4D*> SwiGLU4D::parameters()
{
    return { &W1_, &W2_ };
}