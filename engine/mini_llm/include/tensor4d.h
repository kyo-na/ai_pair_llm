#pragma once
#include <vector>
#include <algorithm>

class Tensor4D {
public:
    int B,T,H,D;

    std::vector<float> data;
    std::vector<float> grad;

    Tensor4D()
        : B(0),T(0),H(0),D(0)
    {}

    Tensor4D(int b,int t,int h,int d)
        : B(b),T(t),H(h),D(d),
          data(b*t*h*d, 0.0f),
          grad(b*t*h*d, 0.0f)
    {}

    inline int index(int b,int t,int h,int d) const
    {
        return ((b*T + t)*H + h)*D + d;
    }

    inline float& at(int b,int t,int h,int d)
    {
        return data[index(b,t,h,d)];
    }

    inline float at(int b,int t,int h,int d) const
    {
        return data[index(b,t,h,d)];
    }

    inline float& grad_at(int b,int t,int h,int d)
    {
        return grad[index(b,t,h,d)];
    }

    void zero()
    {
        std::fill(data.begin(), data.end(), 0.0f);
    }

    void zero_grad()
    {
        std::fill(grad.begin(), grad.end(), 0.0f);
    }
};