
#pragma once
#include <vector>
#include "activations.h"
struct FFN{
    int D,Dff; bool gelu_on;
    std::vector<float>w1,w2;
    FFN(int d,int dff,bool g):D(d),Dff(dff),gelu_on(g),w1(d*dff,0.01f),w2(dff*d,0.01f){}
    void forward(float* x){
        std::vector<float> h(Dff);
        for(int j=0;j<Dff;j++){
            float s=0; for(int i=0;i<D;i++) s+=x[i]*w1[i*Dff+j];
            h[j]=gelu_on?gelu(s):silu(s);
        }
        for(int j=0;j<D;j++){
            float s=0; for(int i=0;i<Dff;i++) s+=h[i]*w2[i*D+j];
            x[j]+=s;
        }
    }
};
