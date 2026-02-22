
#pragma once
#include <cmath>
inline float silu(float x){ return x/(1.0f+std::exp(-x)); }
inline float gelu(float x){
    const float k=0.7978845608f;
    return 0.5f*x*(1.0f+std::tanh(k*(x+0.044715f*x*x*x)));
}
