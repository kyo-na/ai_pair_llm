// engine/mini_llm/src/loss/cross_entropy4d.cpp
#include "loss/cross_entropy4d.h"
#include <cmath>

namespace mini_llm {

float cross_entropy4d(const Tensor4D& logits, int target_id) {
    float maxv = -1e9f;
    for (float v : logits.data)
        if (v > maxv) maxv = v;

    float sum = 0.f;
    for (float v : logits.data)
        sum += std::exp(v - maxv);

    float logp = (logits.data[target_id] - maxv) - std::log(sum);
    return -logp;
}

}