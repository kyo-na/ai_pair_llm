#include "model/mini_AI.h"

MiniAI::MiniAI(int layers, int hidden_dim, int vocab_size)
    : layers_(layers),
      hidden_dim_(hidden_dim),
      vocab_size_(vocab_size),
      embedding_(vocab_size, hidden_dim),
      block_(hidden_dim),
      embed_out_(),
      block_out_() {}

Tensor4D MiniAI::forward_ids(
    int B,
    int T,
    const std::vector<int32_t>& token_ids
) {
    // Embedding
    embed_out_ = embedding_.forward_ids(B, T, token_ids);

    // Transformer block
    block_out_ = block_.forward(embed_out_);

    return block_out_;
}

void MiniAI::backward(const Tensor4D& dlogits) {
    // backward through transformer
    Tensor4D dembed = block_.backward(embed_out_, dlogits);

    // backward through embedding
    embedding_.backward(dembed);
}

void MiniAI::step(float lr) {
    embedding_.step(lr);
    block_.step(lr);
}