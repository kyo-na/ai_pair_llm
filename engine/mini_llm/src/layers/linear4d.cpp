#include "linear4d.h"

Tensor4D Linear4D::forward(const Tensor4D& x) {
    Tensor4D y(x.N, outC, x.H, x.W);
    for (int n=0;n<x.N;n++)
    for (int oc=0;oc<outC;oc++)
    for (int h=0;h<x.H;h++)
    for (int w=0;w<x.W;w++) {
        float sum = 0;
        for (int ic=0;ic<inC;ic++)
            sum += x.at(n,ic,h,w) * W.at(0,oc,0,ic);
        y.at(n,oc,h,w) = sum;
    }
    return y;
}

Tensor4D Linear4D::backward(const Tensor4D& gy, const Tensor4D& x) {
    Tensor4D gx(x.N, inC, x.H, x.W);
    for (int n=0;n<x.N;n++)
    for (int oc=0;oc<outC;oc++)
    for (int h=0;h<x.H;h++)
    for (int w=0;w<x.W;w++) {
        float g = gy.at(n,oc,h,w);
        for (int ic=0;ic<inC;ic++) {
            W.g(0,oc,0,ic) += g * x.at(n,ic,h,w);
            gx.at(n,ic,h,w) += g * W.at(0,oc,0,ic);
        }
    }
    return gx;
}

void Linear4D::step(OptimizerContext& opt) {
    opt.step(W.data.data(), W.grad.data(), W.data.size());
}