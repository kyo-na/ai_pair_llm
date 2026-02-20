
#include "tensor.h"
#include <cmath>

Tensor ffn(const Tensor& x){
    Tensor y=x;
    for(auto& v:y.data)
        v = v/(1+exp(-v)); // SiLU
    return y;
}
