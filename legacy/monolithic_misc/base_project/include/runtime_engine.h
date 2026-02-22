#pragma once
#include "llama_weights.h"
#include "memory_arena.h"
#include "kv_cache.h"
#include <vector>

class LlamaRuntimeGPU {
public:
    LlamaWeights w;
    MemoryArena arena;
    KVPagePool kv_pool;
    cudaStream_t stream;

    int D=4096, H=32, Hd=128, L=32, V=32000, FF=11008;

    void init(int d, int h, int l, int v);
    void forward_token(int token, GPUKVPaged& kv, std::vector<float>& logits_out);
};
