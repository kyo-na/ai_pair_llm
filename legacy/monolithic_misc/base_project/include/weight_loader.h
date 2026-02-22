#pragma once
#include "llama_weights.h"
#include "memory_arena.h"
#include "gguf_parser.h"

class WeightLoader {
public:
    static void load(const GGUFFile& gguf, LlamaWeights& w, MemoryArena& arena, cudaStream_t stream);
};
