
#pragma once
#include <vector>
#include <cstdlib>

struct Tensor {
    int rows, cols;
    std::vector<float> data;

    Tensor(int r=0,int c=0):rows(r),cols(c),data(r*c){
        for(auto& v:data) v = (float)rand()/RAND_MAX;
    }

    float& operator()(int r,int c){
        return data[r*cols+c];
    }
};
