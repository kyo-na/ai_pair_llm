#include "loss/cross_entropy.h"
#include <cmath>

float softmax_cross_entropy(
    const Tensor4D& logits,
    int target,
    Tensor4D& dlogits
){
    const int V = logits.W;
    float maxv=-1e9;
    for(int i=0;i<V;i++)
        maxv = std::max(maxv, logits.data[i]);

    float sum=0;
    for(int i=0;i<V;i++){
        dlogits.data[i]=std::exp(logits.data[i]-maxv);
        sum+=dlogits.data[i];
    }

    for(int i=0;i<V;i++)
        dlogits.data[i]/=sum;

    float loss = -std::log(dlogits.data[target]+1e-9f);
    dlogits.data[target]-=1.f;
    return loss;
}