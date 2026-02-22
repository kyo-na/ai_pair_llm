#include "layers/linear4d.h"
#include "utils.h"
#include "optimizer/adam.h"
#include <algorithm>
#include <cassert>

Linear4D::Linear4D(int in_d, int out_d)
    : in_dim(in_d), out_dim(out_d),
      W((size_t)in_d * out_d, 0.0f),
      dW((size_t)in_d * out_d, 0.0f),
      mW((size_t)in_d * out_d, 0.0f),
      vW((size_t)in_d * out_d, 0.0f),
      b((size_t)out_d, 0.0f),
      db((size_t)out_d, 0.0f),
      mb((size_t)out_d, 0.0f),
      vb((size_t)out_d, 0.0f),
      last_x()
{
    // small init
    for (auto& w : W) w = init_uniform();
    for (auto& bi : b) bi = 0.0f;
}

void Linear4D::zero_grad() {
    std::fill(dW.begin(), dW.end(), 0.0f);
    std::fill(db.begin(), db.end(), 0.0f);
}

Tensor4D Linear4D::forward(const Tensor4D& x) {
    assert(x.D == in_dim);
    last_x = x;

    Tensor4D y(x.B, x.T, x.H, out_dim);

    // y[b,t,h,o] = sum_i x[b,t,h,i] * W[i,o] + b[o]
    for(int bch=0; bch<x.B; bch++){
        for(int t=0; t<x.T; t++){
            for(int h=0; h<x.H; h++){
                for(int o=0; o<out_dim; o++){
                    double sum = (double)b[(size_t)o];
                    for(int i=0; i<in_dim; i++){
                        float xv = x.at(bch, t, h, i);
                        float wv = W[(size_t)i*out_dim + (size_t)o];
                        sum += (double)xv * (double)wv;
                    }
                    y.at(bch, t, h, o) = (float)sum;
                }
            }
        }
    }
    return y;
}

Tensor4D Linear4D::backward(const Tensor4D& dout) {
    // dout: (B,T,H,out_dim)
    assert(dout.D == out_dim);
    assert(last_x.D == in_dim);

    Tensor4D dx(last_x.B, last_x.T, last_x.H, in_dim);
    dx.zero_grad(); // not strictly needed

    // Accumulate dW, db, dx
    // dW[i,o] += sum_{b,t,h} x[b,t,h,i] * dout[b,t,h,o]
    // db[o]   += sum_{b,t,h} dout[b,t,h,o]
    // dx[b,t,h,i] = sum_o dout[b,t,h,o] * W[i,o]
    for(int bch=0; bch<last_x.B; bch++){
        for(int t=0; t<last_x.T; t++){
            for(int h=0; h<last_x.H; h++){

                // db and dW
                for(int o=0; o<out_dim; o++){
                    float go = dout.at(bch, t, h, o);
                    db[(size_t)o] += go;

                    for(int i=0; i<in_dim; i++){
                        float xv = last_x.at(bch, t, h, i);
                        dW[(size_t)i*out_dim + (size_t)o] += xv * go;
                    }
                }

                // dx
                for(int i=0; i<in_dim; i++){
                    double sum = 0.0;
                    for(int o=0; o<out_dim; o++){
                        float go = dout.at(bch, t, h, o);
                        float wv = W[(size_t)i*out_dim + (size_t)o];
                        sum += (double)go * (double)wv;
                    }
                    dx.at(bch, t, h, i) = (float)sum;
                }
            }
        }
    }

    return dx;
}

void Linear4D::step(float lr) {
    step_count += 1;

    // Adam update for W, b (vector-based)
    adam_update_vec(W, dW, mW, vW, lr, step_count);
    adam_update_vec(b, db, mb, vb, lr, step_count);

    // Important: clear grads after update (so accumulation doesn't explode)
    zero_grad();
}