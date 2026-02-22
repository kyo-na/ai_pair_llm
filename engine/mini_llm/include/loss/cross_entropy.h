#pragma once
#include "tensor4d.h"

float cross_entropy(const Tensor4D& logits,
                    const std::vector<int>& targets,
                    int vocab_size);