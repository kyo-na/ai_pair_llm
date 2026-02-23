#pragma once
#include "../tensor4d.h"

struct Linear {
    Tensor4D W;

    Linear(int in,int out)
        : W(1,1,in,out) {}

    Tensor4D forward(const Tensor4D& x){
        Tensor4D y(1,1,1,W.D);
        for(int o=0;o<W.D;o++)
            for(int i=0;i<W.C;i++)
                y.at(0,0,0,o)+=x.at(0,0,0,i)*W.at(0,0,i,o);
        return y;
    }

    Tensor4D backward(const Tensor4D& dy){
        Tensor4D dx(1,1,1,W.C);
        for(int o=0;o<W.D;o++)
            for(int i=0;i<W.C;i++){
                W.grad[W.idx(0,0,i,o)] += dy.at(0,0,0,o);
                dx.at(0,0,0,i)+=dy.at(0,0,0,o)*W.at(0,0,i,o);
            }
        return dx;
    }
};