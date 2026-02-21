#include "layers/layernorm4d.h"
#include "optimizer/adam.h"
#include <cmath>
#include <cassert>
#include <algorithm>

LayerNorm4D::LayerNorm4D(int d)
    : dim(d),
      gamma((size_t)d, 1.0f),
      beta((size_t)d, 0.0f),
      dgamma((size_t)d, 0.0f),
      dbeta((size_t)d, 0.0f),
      mg((size_t)d, 0.0f),
      vg((size_t)d, 0.0f),
      mb((size_t)d, 0.0f),
      vb((size_t)d, 0.0f)
{}

void LayerNorm4D::zero_grad(){
    std::fill(dgamma.begin(), dgamma.end(), 0.0f);
    std::fill(dbeta.begin(),  dbeta.end(),  0.0f);
}

Tensor4D LayerNorm4D::forward(const Tensor4D& x){
    assert(x.D == dim);

    const int N = x.B * x.T * x.H;
    mean.assign((size_t)N, 0.0f);
    var.assign((size_t)N, 0.0f);

    x_hat = Tensor4D(x.B, x.T, x.H, x.D);
    Tensor4D y(x.B, x.T, x.H, x.D);

    int idx = 0;
    for(int b=0;b<x.B;b++){
        for(int t=0;t<x.T;t++){
            for(int h=0;h<x.H;h++, idx++){
                // mean
                double m = 0.0;
                for(int d=0;d<dim;d++){
                    m += x.at(b,t,h,d);
                }
                m /= (double)dim;
                mean[idx] = (float)m;

                // variance
                double v = 0.0;
                for(int d=0;d<dim;d++){
                    double diff = (double)x.at(b,t,h,d) - m;
                    v += diff * diff;
                }
                v /= (double)dim;
                var[idx] = (float)v;

                float inv_std = 1.0f / std::sqrt((float)v + eps);

                // normalize
                for(int d=0;d<dim;d++){
                    float xh = (x.at(b,t,h,d) - (float)m) * inv_std;
                    x_hat.at(b,t,h,d) = xh;
                    y.at(b,t,h,d) = gamma[(size_t)d] * xh + beta[(size_t)d];
                }
            }
        }
    }
    return y;
}

Tensor4D LayerNorm4D::backward(const Tensor4D& dout){
    assert(dout.D == dim);

    Tensor4D dx(dout.B, dout.T, dout.H, dout.D);
    dx.zero_grad();

    const int N = dout.B * dout.T * dout.H;
    int idx = 0;

    for(int b=0;b<dout.B;b++){
        for(int t=0;t<dout.T;t++){
            for(int h=0;h<dout.H;h++, idx++){

                float mu  = mean[idx];
                float varv = var[idx];
                float inv_std = 1.0f / std::sqrt(varv + eps);

                // dγ, dβ
                for(int d=0;d<dim;d++){
                    float go = dout.at(b,t,h,d);
                    dgamma[(size_t)d] += go * x_hat.at(b,t,h,d);
                    dbeta[(size_t)d]  += go;
                }

                // dx
                double sum1 = 0.0;
                double sum2 = 0.0;
                for(int d=0;d<dim;d++){
                    float go = dout.at(b,t,h,d) * gamma[(size_t)d];
                    sum1 += go;
                    sum2 += go * (x_hat.at(b,t,h,d));
                }

                for(int d=0;d<dim;d++){
                    float go = dout.at(b,t,h,d) * gamma[(size_t)d];
                    float xh = x_hat.at(b,t,h,d);
                    float dxv = (go - (float)(sum1/dim) - xh*(float)(sum2/dim)) * inv_std;
                    dx.at(b,t,h,d) = dxv;
                }
            }
        }
    }
    return dx;
}

void LayerNorm4D::step(float lr){
    step_count += 1;
    adam_update_vec(gamma, dgamma, mg, vg, lr, step_count);
    adam_update_vec(beta,  dbeta,  mb, vb, lr, step_count);
    zero_grad();
}