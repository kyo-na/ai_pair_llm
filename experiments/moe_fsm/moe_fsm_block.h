
// moe_fsm_block.h
#pragma once
#include <vector>

struct Expert {
    Tensor W;
    Tensor b;
    Expert(int D): W({D,D},false), b({D},false) {}
    void apply(const float* x, float* y, int D) const {
        for(int j=0;j<D;++j){
            float v=b.data[j];
            for(int i=0;i<D;++i) v+=x[i]*W.data[i*D+j];
            y[j]=v;
        }
    }
};

Tensor moe_fsm_forward(Tensor& X, Tensor& W_router, std::vector<Expert>& experts){
    int B=X.shape[0], D=X.shape[1], E=W_router.shape[1];
    Tensor O({B,D},false);
    for(int b=0;b<B;++b){
        float best1=-1e30f,best2=-1e30f;
        int i1=-1,i2=-1;
        for(int e=0;e<E;++e){
            float s=0.f;
            for(int d=0;d<D;++d) s+=X.data[b*D+d]*W_router.data[d*E+e];
            if(s>best1){best2=best1;i2=i1;best1=s;i1=e;}
            else if(s>best2){best2=s;i2=e;}
        }
        std::vector<float> tmp(D,0.f), tmp2(D);
        if(i1>=0) experts[i1].apply(&X.data[b*D], tmp.data(), D);
        if(i2>=0){
            experts[i2].apply(&X.data[b*D], tmp2.data(), D);
            for(int d=0;d<D;++d) tmp[d]+=tmp2[d];
        }
        for(int d=0;d<D;++d) O.data[b*D+d]=0.5f*tmp[d];
    }
    return O;
}
