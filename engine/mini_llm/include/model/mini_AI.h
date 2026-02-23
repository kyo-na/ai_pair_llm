#pragma once
#include <vector>
#include <cstdint>

#include "tensor4d.h"
#include "layers/embedding4d.h"
#include "blocks/transformer_block4d.h"

/**
 * MiniAI
 * - 現行 mini_llm API 完全一致
 * - forward / backward / step 対応
 */
class MiniAI {
public:
    MiniAI(int layers, int hidden_dim, int vocab_size);

    // forward
    Tensor4D forward_ids(
        int B,
        int T,
        const std::vector<int32_t>& token_ids
    );

    // backward
    void backward(const Tensor4D& dlogits);

    // optimizer step
    void step(float lr);

private:
    int layers_;
    int hidden_dim_;
    int vocab_size_;

    Embedding4D embedding_;
    TransformerBlock4D block_;

    // cache (for backward)
    Tensor4D embed_out_;
    Tensor4D block_out_;
};