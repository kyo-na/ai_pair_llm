#include "layers/attention4d.h"
#include <algorithm>
#include <cassert>

static inline float clampf(float x, float lo, float hi){
    return std::max(lo, std::min(hi, x));
}

Attention4D::Attention4D(int d)
    : dim(d),
      Wq(d,d),
      Wk(d,d),
      Wv(d,d)
{
}

Tensor4D Attention4D::forward(const Tensor4D& x){

    assert(x.D == dim);

    Q = Wq.forward(x);
    K = Wk.forward(x);
    V = Wv.forward(x);

    Tensor4D y(x.B, x.T, x.H, x.D);

    attn.assign((size_t)x.B * x.T * x.T, 0.0f);

    const float scale   = 1.0f / std::sqrt((float)dim);
    const float NEG_INF = -1e9f;

    for(int b=0;b<x.B;b++){
        for(int t=0;t<x.T;t++){

            std::vector<float> score(x.T);

            // ---- QK^T + Causal Mask ----
            for(int tj=0;tj<x.T;tj++){

                if(tj > t){
                    score[tj] = NEG_INF;  // 未来は見えない
                    continue;
                }

                double s = 0.0;
                for(int d=0; d<dim; d++){
                    s += (double)Q.at(b,t,0,d) *
                         (double)K.at(b,tj,0,d);
                }

                score[tj] = (float)(s * scale);
            }

            // ---- max subtraction (数値安定) ----
            float m = *std::max_element(score.begin(), score.end());
            for(auto& v : score){
                v = clampf(v - m, -50.0f, 50.0f);
            }

            // ---- softmax ----
            double sum = 0.0;
            for(auto v : score){
                sum += std::exp((double)v);
            }

            for(int tj=0;tj<x.T;tj++){
                float p = (float)(std::exp((double)score[tj]) / sum);
                attn[(size_t)b*x.T*x.T +
                     (size_t)t*x.T +
                     (size_t)tj] = p;
            }

            // ---- weighted sum ----
            for(int d=0; d<dim; d++){
                double out = 0.0;
                for(int tj=0;tj<x.T;tj++){
                    float p = attn[(size_t)b*x.T*x.T +
                                   (size_t)t*x.T +
                                   (size_t)tj];
                    out += (double)p *
                           (double)V.at(b,tj,0,d);
                }
                y.at(b,t,0,d) = (float)out;
            }
        }
    }

    return y;
}

Tensor4D Attention4D::backward(const Tensor4D& dout){

    Tensor4D dQ(Q.B, Q.T, Q.H, Q.D);
    Tensor4D dK(K.B, K.T, K.H, K.D);
    Tensor4D dV(V.B, V.T, V.H, V.D);

    const float scale = 1.0f / std::sqrt((float)dim);

    for(int b=0;b<Q.B;b++){
        for(int t=0;t<Q.T;t++){

            std::vector<float> p(Q.T);
            std::vector<float> g(Q.T);

            // g_j = dout · V_j
            for(int tj=0;tj<Q.T;tj++){

                p[tj] = attn[(size_t)b*Q.T*Q.T +
                             (size_t)t*Q.T +
                             (size_t)tj];

                double dot = 0.0;
                for(int d=0; d<dim; d++){
                    dot += (double)dout.at(b,t,0,d) *
                           (double)V.at(b,tj,0,d);
                }
                g[tj] = (float)dot;
            }

            // 完全 softmax backward
            double pg = 0.0;
            for(int j=0;j<Q.T;j++){
                pg += (double)p[j] * (double)g[j];
            }

            std::vector<float> dscore(Q.T);

            for(int j=0;j<Q.T;j++){
                dscore[j] = p[j] * (g[j] - (float)pg);
                dscore[j] = clampf(dscore[j], -5.0f, 5.0f);
            }

            // Q/K grad
            for(int tj=0;tj<Q.T;tj++){
                for(int d=0; d<dim; d++){
                    dQ.at(b,t,0,d)  +=
                        dscore[tj] * K.at(b,tj,0,d) * scale;

                    dK.at(b,tj,0,d) +=
                        dscore[tj] * Q.at(b,t,0,d)  * scale;
                }
            }

            // V grad
            for(int tj=0;tj<Q.T;tj++){
                for(int d=0; d<dim; d++){
                    dV.at(b,tj,0,d) +=
                        p[tj] * dout.at(b,t,0,d);
                }
            }
        }
    }

    Tensor4D dx  = Wq.backward(dQ);
    Tensor4D dk  = Wk.backward(dK);
    Tensor4D dv  = Wv.backward(dV);

    for(size_t i=0;i<dx.data.size();i++){
        dx.data[i] += dk.data[i] + dv.data[i];
    }

    return dx;
}

void Attention4D::step(float lr){
    Wq.step(lr);
    Wk.step(lr);
    Wv.step(lr);
}