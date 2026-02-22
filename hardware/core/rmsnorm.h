
#pragma once
#include <cmath>
struct RMSNorm{
    int d; float eps;
    RMSNorm(int d_, float e=1e-5f):d(d_),eps(e){}
    void forward(float* x)const{
        float s=0; for(int i=0;i<d;i++) s+=x[i]*x[i];
        s=1.0f/std::sqrt(s/d+eps);
        for(int i=0;i<d;i++) x[i]*=s;
    }
};
