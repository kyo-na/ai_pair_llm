#pragma once
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <vector>
#include <atomic>

constexpr int KV_PAGE_TOKENS = 16;
struct KVPage { half* d_k; half* d_v; std::atomic<int> refcnt; };

class KVPagePool {
    std::vector<KVPage*> free_;
public:
    void init(int max_pages, int n_kv, int hd);
    KVPage* alloc();
    void decref(KVPage* p);
    void incref(KVPage* p);
};

struct GPUKVPaged {
    int cur_pos = 0;
    std::vector<std::vector<KVPage*>> pages;
    void append_kv(KVPagePool* pool, int layer, half* k, half* v, int n_kv, int hd, cudaStream_t s);
    void gather_to_linear(half* k_out, half* v_out, int layer, int n_kv, int hd, cudaStream_t s);
};
