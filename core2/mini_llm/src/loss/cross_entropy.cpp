#include "loss/cross_entropy.h"
#include <cmath>
#include <algorithm>

float cross_entropy(
    const Tensor4D& logits,
    const std::vector<int>& target,
    Tensor4D& dlogits
){
    float loss = 0.0f;
    int T = logits.T;
    int D = logits.D;

    for(int t=0; t<T; t++){
        float maxv = -1e9f;
        for(int d=0; d<D; d++)
            maxv = std::max(maxv, logits.at(0,0,t,d));

        float sum = 0.0f;
        for(int d=0; d<D; d++)
            sum += std::exp(logits.at(0,0,t,d) - maxv);

        for(int d=0; d<D; d++){
            float p = std::exp(logits.at(0,0,t,d) - maxv) / sum;
            int y = target[t];
            loss += (d == y ? -std::log(p + 1e-9f) : 0.0f);
            dlogits.at(0,0,t,d) = p - (d == y);
        }
    }
    return loss / T;
}