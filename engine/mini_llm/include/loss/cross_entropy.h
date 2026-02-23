#pragma once
#include "tensor4d.h"

float softmax_cross_entropy(
    const Tensor4D& logits,
    int target,
    Tensor4D& dlogits
);