// linear.cpp
#include "linear.h"

Tensor4D Linear::forward(const Tensor4D& x){
    Tensor4D y(1,1,1,W.D);
    for(int o=0;o<W.D;o++)
        for(int i=0;i<W.C;i++)
            y.at(0,0,0,o)+=x.data[i]*W.at(0,0,i,o);
    return y;
}

Tensor4D Linear::backward(const Tensor4D& gy){
    Tensor4D gx(1,1,1,W.C);
    for(int o=0;o<W.D;o++)
        for(int i=0;i<W.C;i++){
            W.g(0,0,i,o)+=gy.data[o];
            gx.data[i]+=gy.data[o]*W.at(0,0,i,o);
        }
    return gx;
}