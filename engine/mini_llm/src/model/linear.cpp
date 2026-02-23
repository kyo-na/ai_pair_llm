#include "model/linear.h"

Tensor4D Linear::forward(const Tensor4D& x){
    last_x = x;
    Tensor4D y(1, weight.N, 1, 1);

    for(int o=0;o<weight.N;o++){
        float sum=0;
        for(int i=0;i<weight.C;i++)
            sum += weight.at(o,i,0,0) * x.at(0,i,0,0);
        y.at(0,o,0,0)=sum;
    }
    return y;
}

Tensor4D Linear::backward(const Tensor4D& gy){
    Tensor4D gx(1, weight.C,1,1);

    for(int o=0;o<weight.N;o++){
        for(int i=0;i<weight.C;i++){
            weight.grad[(o*weight.C+i)] +=
                gy.at(0,o,0,0) * last_x.at(0,i,0,0);
            gx.at(0,i,0,0) +=
                gy.at(0,o,0,0) * weight.at(o,i,0,0);
        }
    }
    return gx;
}