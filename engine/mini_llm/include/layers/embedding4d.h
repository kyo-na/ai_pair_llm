#pragma once
#include <vector>
#include <cstdint>
#include "tensor4d.h"

class Embedding4D {
public:
    int vocab;
    int dim;

    // weight: [vocab * dim]
    std::vector<float> W, dW;
    std::vector<int32_t> last_ids; // for backward (optional)

    Embedding4D(int vocab_size, int dim_);

    // ids: size = B*T, output: (B,T,1,dim)
    Tensor4D forward_ids(int B, int T, const std::vector<int32_t>& ids);

    // dout: (B,T,1,dim)
    void backward(const Tensor4D& dout);

    void step(float lr);
};