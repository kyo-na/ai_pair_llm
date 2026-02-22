
#include "model.h"
#include <vector>
std::vector<float> Model::forward(int t,KVCache& kv){
    float x[512]={0}; x[t%512]=1.0f;
    block.forward(x,kv,0);
    std::vector<float> out(vocab);
    for(int i=0;i<vocab;i++) out[i]=x[i%512];
    return out;
}
