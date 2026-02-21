#pragma once
#include "tensor4d.h"

// Mean Squared Error (4D)
// loss = mean((y - t)^2)
float mse_forward(const Tensor4D& y, const Tensor4D& t);

// dL/dy を y.grad に書き込む
void mse_backward(Tensor4D& y, const Tensor4D& t);