
#pragma once
#include "rmsnorm.h"
#include "ffn.h"
#include "cache/kv_cache.h"
struct TransformerBlock{
    RMSNorm n1,n2; FFN ffn;
    TransformerBlock(int d):n1(d),n2(d),ffn(d,2048,true){}
    void forward(float* x,KVCache&,int){
        n1.forward(x); n2.forward(x); ffn.forward(x);
    }
};
