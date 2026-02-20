
#pragma once
#include <vector>
struct KVCache {
    int T,D;
    std::vector<float> K,V;
    int pos=0;
    KVCache(int T,int D):T(T),D(D){
        K.resize(T*D);
        V.resize(T*D);
    }
};
