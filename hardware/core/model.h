
#pragma once
#include <vector>
#include "cache/kv_cache.h"
struct Model{
    int vocab; TransformerBlock block;
    Model(int v):vocab(v),block(512){}
    std::vector<float> forward(int t,KVCache& kv);
};
