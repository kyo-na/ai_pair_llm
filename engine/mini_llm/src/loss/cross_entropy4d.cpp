#include "loss/cross_entropy4d.h"
#include <cmath>

float CrossEntropy4D::forward(
    const Tensor4D& probs,
    const std::vector<int>& targets)
{
    last_probs_=probs;
    last_targets_=targets;

    float loss=0;
    int idx=0;

    for(int b=0;b<probs.B;++b)
    for(int t=0;t<probs.T;++t){
        int y=targets[idx++];
        float p=probs.at(b,t,0,y);
        loss-=std::log(p+1e-9f);
    }
    return loss/(probs.B*probs.T);
}

Tensor4D CrossEntropy4D::backward(){
    Tensor4D grad=last_probs_;
    int idx=0;

    for(int b=0;b<grad.B;++b)
    for(int t=0;t<grad.T;++t){
        int y=last_targets_[idx++];
        grad.at(b,t,0,y)-=1.0f;
    }
    return grad;
}