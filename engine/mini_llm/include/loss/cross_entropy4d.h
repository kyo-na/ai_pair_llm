#pragma once
#include <vector>
#include <cstdint>
#include "tensor4d.h"

// logits: (B,T,1,V)
// target_ids: size B*T (next-token target)
float cross_entropy_forward(const Tensor4D& logits, const std::vector<int32_t>& target_ids);

// dlogits.grad を埋める（同shape）
void cross_entropy_backward(Tensor4D& dlogits, const std::vector<int32_t>& target_ids);