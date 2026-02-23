#include "../../include/adam.h"
#include <cmath>

Adam::Adam(int n, float lr_)
    : lr(lr_), b1(0.9f), b2(0.999f), eps(1e-8f),
      step(0), m(n,0.0f), v(n,0.0f) {}

void Adam::update(Tensor4D& w) {
    step++;
    for (int i = 0; i < w.size(); i++) {
        float g = w.grad[i];
        m[i] = b1*m[i] + (1-b1)*g;
        v[i] = b2*v[i] + (1-b2)*g*g;

        float mh = m[i] / (1 - std::pow(b1, step));
        float vh = v[i] / (1 - std::pow(b2, step));

        w.data[i] -= lr * mh / (std::sqrt(vh) + eps);
    }
}