#include "decode/sampling4d.h"

std::vector<int> argmax4d(const Tensor4D& probs){
    std::vector<int> out;
    for(int b=0;b<probs.B;++b)
    for(int t=0;t<probs.T;++t){
        float best=-1;
        int idx=0;
        for(int v=0;v<probs.D;++v){
            float p=probs.at(b,t,0,v);
            if(p>best){ best=p; idx=v; }
        }
        out.push_back(idx);
    }
    return out;
}