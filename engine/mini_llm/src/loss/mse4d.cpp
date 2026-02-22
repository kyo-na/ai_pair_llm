#include "loss/mse4d.h"
#include <cassert>

float mse_forward(const Tensor4D& y, const Tensor4D& t){
    assert(y.B==t.B && y.T==t.T && y.H==t.H && y.D==t.D);

    double sum = 0.0;
    size_t n = y.data.size();
    for(size_t i=0;i<n;i++){
        double d = (double)y.data[i] - (double)t.data[i];
        sum += d*d;
    }
    return (float)(sum / (double)n);
}

void mse_backward(Tensor4D& y, const Tensor4D& t){
    assert(y.B==t.B && y.T==t.T && y.H==t.H && y.D==t.D);

    size_t n = y.data.size();
    float inv_n = 1.0f / (float)n;

    for(size_t i=0;i<n;i++){
        // d/dy ( (y - t)^2 / n ) = 2*(y - t)/n
        y.grad[i] = 2.0f * (y.data[i] - t.data[i]) * inv_n;
    }
}