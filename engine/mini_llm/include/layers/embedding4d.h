#pragma once
#include <vector>
#include "tensor4d.h"

class Embedding4D
{
public:
    Embedding4D(int V,int H,int D);

    Tensor4D forward(const std::vector<int>& ids);
    void backward(const std::vector<int>& ids,
                  const Tensor4D& dX);

    std::vector<Tensor4D*> parameters();
    void zero_grad();

private:
    int V_,H_,D_;
    Tensor4D W_;
};