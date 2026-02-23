// embedding.h
#pragma once
#include "tensor4d.h"

struct Embedding {
    Tensor4D weight;
    Embedding(int vocab,int dim):weight(1,1,vocab,dim){}
    Tensor4D forward(int id);
    void backward(int id,const Tensor4D& grad);
};