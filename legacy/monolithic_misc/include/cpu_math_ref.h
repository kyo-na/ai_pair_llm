#pragma once
#include <vector>

// CPU Reference Implementations for LLM Operations
void rmsnorm_cpu(float* out, const float* x, const float* weight, int size);
void matmul_cpu(float* out, const float* x, const float* w, int rows, int cols);
void rope_cpu(float* q, float* k, int dim, int pos);
void attention_cpu(float* out, float* q, float* k_cache, float* v_cache, int seq_len, int n_heads, int head_size);
void ffn_swiglu_cpu(float* out, float* x, float* w_up, float* w_gate, float* w_down, int D, int hidden_dim);
