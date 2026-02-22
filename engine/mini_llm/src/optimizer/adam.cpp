#include "optimizer/adam.h"
#include <cmath>

static constexpr float B1=0.9f;
static constexpr float B2=0.999f;
static constexpr float EPS=1e-8f;

void adam_update(Tensor4D& t, float lr, int step){
    float b1t = 1.f - std::pow(B1, step);
    float b2t = 1.f - std::pow(B2, step);

    for(size_t i=0;i<t.data.size();i++){
        float g = t.grad[i];
        t.m[i] = B1*t.m[i] + (1-B1)*g;
        t.v[i] = B2*t.v[i] + (1-B2)*g*g;

        float mh = t.m[i]/b1t;
        float vh = t.v[i]/b2t;
        t.data[i] -= lr * mh / (std::sqrt(vh)+EPS);
    }
}

// 追加：vector parameter 用
void adam_update_vec(
    std::vector<float>& w,
    const std::vector<float>& g,
    std::vector<float>& m,
    std::vector<float>& v,
    float lr,
    int step
){
    float b1t = 1.f - std::pow(B1, step);
    float b2t = 1.f - std::pow(B2, step);

    const size_t n = w.size();
    for(size_t i=0;i<n;i++){
        float gi = g[i];
        m[i] = B1*m[i] + (1-B1)*gi;
        v[i] = B2*v[i] + (1-B2)*gi*gi;

        float mh = m[i]/b1t;
        float vh = v[i]/b2t;

        w[i] -= lr * mh / (std::sqrt(vh)+EPS);
    }
}