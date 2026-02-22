#pragma once
#include <vector>
#include "tensor4d.h"

// 既存：Tensor4D用（残してOK）
void adam_update(Tensor4D& t, float lr, int step);

// 追加：パラメータ配列用（W, b など）
void adam_update_vec(
    std::vector<float>& w,
    const std::vector<float>& g,
    std::vector<float>& m,
    std::vector<float>& v,
    float lr,
    int step
);