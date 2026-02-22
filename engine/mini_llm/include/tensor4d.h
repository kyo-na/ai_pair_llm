#pragma once
#include <vector>
#include <algorithm>

struct Tensor4D {
    int B=0, T=0, H=0, D=0;
    std::vector<float> data;
    std::vector<float> grad;
    std::vector<float> m;
    std::vector<float> v;

    Tensor4D() {}

    Tensor4D(int b,int t,int h,int d)
        : B(b),T(t),H(h),D(d),
          data((size_t)b*t*h*d,0.f),
          grad((size_t)b*t*h*d,0.f),
          m((size_t)b*t*h*d,0.f),
          v((size_t)b*t*h*d,0.f) {}

    inline size_t idx(int b,int t,int h,int d) const {
        return (((size_t)b*T + t)*H + h)*D + d;
    }

    inline float& at(int b,int t,int h,int d){
        return data[idx(b,t,h,d)];
    }

    inline float at(int b,int t,int h,int d) const {
        return data[idx(b,t,h,d)];
    }

    inline void zero_grad(){
        std::fill(grad.begin(), grad.end(), 0.f);
    }
};