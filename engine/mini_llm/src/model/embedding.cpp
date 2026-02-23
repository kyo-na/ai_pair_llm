// embedding.cpp
#include "embedding.h"

Tensor4D Embedding::forward(int id){
    Tensor4D out(1,1,1,weight.D);
    for(int d=0;d<weight.D;d++)
        out.at(0,0,0,d)=weight.at(0,0,id,d);
    return out;
}

void Embedding::backward(int id,const Tensor4D& g){
    for(int d=0;d<weight.D;d++)
        weight.g(0,0,id,d)+=g.data[d];
}