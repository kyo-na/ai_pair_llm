#pragma once
#include <vector>
#include <algorithm>

struct Tensor4D {
    int B,T,C,D;
    std::vector<float> data, grad;

    Tensor4D(int b=1,int t=1,int c=1,int d=1);

    inline int idx(int b,int t,int c,int d) const {
        return ((b*T+t)*C+c)*D+d;
    }

    inline float& at(int b,int t,int c,int d) {
        return data[idx(b,t,c,d)];
    }

    void zero_grad();
    void save(const char* path) const;
    void load(const char* path);
};