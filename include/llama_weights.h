#pragma once
#include <vector>

struct LlamaLayerWeights {
    void* attn_q; void* attn_k; void* attn_v; void* attn_output;
    void* ffn_gate; void* ffn_down; void* ffn_up;
    void* attn_norm; void* ffn_norm;
};

struct LlamaWeights {
    void* token_embd; void* output_norm; void* output_weight;
    std::vector<LlamaLayerWeights> layers;
    int n_layers=0, n_embd=0, n_head=0, n_head_kv=0, n_vocab=0;
};
