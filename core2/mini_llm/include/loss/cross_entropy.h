#pragma once
#include "tensor4d.h"
#include <vector>

float cross_entropy(
    const Tensor4D& logits,
    const std::vector<int>& target,
    Tensor4D& dlogits
);