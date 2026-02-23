// ffn4d.h
#pragma once
#include "tensor4d.h"

struct FFN4D {
    Tensor4D forward(const Tensor4D& x) {
        return x; // 最小実装
    }
};