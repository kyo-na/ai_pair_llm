#pragma once
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cstdint>

constexpr int Q4K_GROUP = 32;
struct Q4KBlock { uint8_t qs[Q4K_GROUP / 2]; float scale; };

__global__ void get_embedding_kernel(const half* embd, int token_id, half* out, int D);
__global__ void rmsnorm_kernel(const half* x, const float* w, half* y, int D, float eps);
__global__ void q4k_gemv_kernel(const Q4KBlock* W, const half* x, half* y, int rows, int cols);
__global__ void rope_kernel(half* qk, int Hd, int pos, float base);
__global__ void causal_attention_kernel(const half* q, const half* k_cache, const half* v_cache, half* out, int T, int Hd);
__global__ void silu_mul_kernel(const half* up, const half* gate, half* out, int D);
__global__ void residual_add_kernel(half* x, const half* res, int D);
