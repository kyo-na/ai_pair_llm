#pragma once
#include "tensor4d.h"
#include <vector>

class CrossEntropy4D {
public:
    float forward(const Tensor4D& probs,
                  const std::vector<int>& targets); // size=B*T

    Tensor4D backward();  // grad w.r.t probs

private:
    Tensor4D last_probs_;
    std::vector<int> last_targets_;
};