
#include "tensor.h"
#include <cmath>

void rmsnorm(Tensor& x){
    float ss=0;
    for(float v:x.data) ss+=v*v;
    float inv=1.0f/std::sqrt(ss/x.data.size()+1e-6f);
    for(auto& v:x.data) v*=inv;
}
