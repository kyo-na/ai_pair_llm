// attention4d.h
#pragma once
#include "tensor4d.h"
#include "softmax4d.h"

struct Attention4D {
    Tensor4D last_attn;

    Tensor4D forward(const Tensor4D& qk) {
        last_attn = softmax4d(qk);
        return last_attn;
    }

    Tensor4D backward(const Tensor4D& grad_out) {
        // ★ 簡易：softmax backward 省略（実験用）
        return grad_out;
    }
};