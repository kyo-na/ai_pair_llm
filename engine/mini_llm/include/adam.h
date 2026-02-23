#pragma once
#include "tensor4d.h"
#include <vector>
#include <cmath>

struct Adam {
    float lr=0.05f,b1=0.9f,b2=0.999f,eps=1e-8f;
    int step=0;
    std::vector<float> m,v;

    Adam(int n):m(n,0.0f),v(n,0.0f){}

    void update(Tensor4D& w){
        step++;
        for(int i=0;i<w.data.size();i++){
            float g=w.grad[i];
            m[i]=b1*m[i]+(1-b1)*g;
            v[i]=b2*v[i]+(1-b2)*g*g;
            float mh=m[i]/(1-std::pow(b1,step));
            float vh=v[i]/(1-std::pow(b2,step));
            w.data[i]-=lr*mh/(std::sqrt(vh)+eps);
        }
    }
};