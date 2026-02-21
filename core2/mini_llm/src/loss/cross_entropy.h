#pragma once
#include <vector>
#include "tensor4d.h"

float cross_entropy(
    const Tensor4D& logits,
    const std::vector<int>& target,
    Tensor4D& dlogits
);