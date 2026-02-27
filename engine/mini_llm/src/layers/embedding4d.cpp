#include "layers/embedding4d.h"
#include <cstdlib>

Embedding4D::Embedding4D(int V,int H,int D)
    :V_(V),H_(H),D_(D),
     W_(V,1,H,D)
{
    for(auto& x:W_.data)
        x=0.02f*((float)rand()/RAND_MAX-0.5f);
}

Tensor4D Embedding4D::forward(const std::vector<int>& ids)
{
    int T=ids.size();
    Tensor4D out(1,T,H_,D_);

    for(int t=0;t<T;++t)
    {
        int tok=ids[t];
        for(int h=0;h<H_;++h)
        for(int d=0;d<D_;++d)
            out.at(0,t,h,d)=W_.at(tok,0,h,d);
    }
    return out;
}

void Embedding4D::backward(const std::vector<int>& ids,
                           const Tensor4D& dX)
{
    int T=ids.size();
    for(int t=0;t<T;++t)
    {
        int tok=ids[t];
        for(int h=0;h<H_;++h)
        for(int d=0;d<D_;++d)
            W_.grad_at(tok,0,h,d)+=
                dX.at(0,t,h,d);
    }
}

std::vector<Tensor4D*> Embedding4D::parameters()
{
    return {&W_};
}

void Embedding4D::zero_grad()
{
    W_.zero_grad();
}