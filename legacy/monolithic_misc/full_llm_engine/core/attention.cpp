
#include "tensor.h"
#include <vector>
#include <cmath>

Tensor attention(const Tensor& Q,const Tensor& K,const Tensor& V){
    int D=Q.cols;
    Tensor out(1,D);

    std::vector<float> score(K.rows);
    for(int i=0;i<K.rows;i++){
        float dot=0;
        for(int d=0;d<D;d++)
            dot+=Q(0,d)*K(i,d);
        score[i]=dot/std::sqrt(D);
    }

    float sum=0;
    for(auto& s:score){ s=exp(s); sum+=s; }
    for(auto& s:score) s/=sum;

    for(int d=0;d<D;d++){
        float v=0;
        for(int i=0;i<K.rows;i++)
            v+=score[i]*V(i,d);
        out(0,d)=v;
    }
    return out;
}
