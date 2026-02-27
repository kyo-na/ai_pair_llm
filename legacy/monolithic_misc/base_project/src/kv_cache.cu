#include "cache/kv_cache.h"
#include <stdexcept>

void KVPagePool::init(int max_pages, int n_kv, int hd) {
    size_t sz = KV_PAGE_TOKENS * n_kv * hd * sizeof(half);
    for(int i=0; i<max_pages; i++) {
        KVPage* p = new KVPage;
        cudaMalloc(&p->d_k, sz); cudaMalloc(&p->d_v, sz);
        p->refcnt.store(0); free_.push_back(p);
    }
}

KVPage* KVPagePool::alloc() {
    if(free_.empty()) throw std::runtime_error("KV Pool empty");
    KVPage* p = free_.back(); free_.pop_back(); p->refcnt.store(1); return p;
}

void KVPagePool::decref(KVPage* p) { if(p->refcnt.fetch_sub(1) == 1) free_.push_back(p); }
void KVPagePool::incref(KVPage* p) { p->refcnt.fetch_add(1); }

void GPUKVPaged::append_kv(KVPagePool* pool, int layer, half* k, half* v, int n_kv, int hd, cudaStream_t s) {
    int p_idx = cur_pos / KV_PAGE_TOKENS; int off = cur_pos % KV_PAGE_TOKENS;
    if(pages.size() <= layer) pages.resize(layer+1);
    if(p_idx >= pages[layer].size()) pages[layer].push_back(pool->alloc());
    
    KVPage* p = pages[layer][p_idx];
    if(p->refcnt.load() > 1) {
        KVPage* np = pool->alloc();
        size_t sz = KV_PAGE_TOKENS * n_kv * hd * sizeof(half);
        cudaMemcpyAsync(np->d_k, p->d_k, sz, cudaMemcpyDeviceToDevice, s);
        cudaMemcpyAsync(np->d_v, p->d_v, sz, cudaMemcpyDeviceToDevice, s);
        pool->decref(p); pages[layer][p_idx] = np; p = np;
    }
    cudaMemcpyAsync(p->d_k + off*n_kv*hd, k, n_kv*hd*sizeof(half), cudaMemcpyDeviceToDevice, s);
    cudaMemcpyAsync(p->d_v + off*n_kv*hd, v, n_kv*hd*sizeof(half), cudaMemcpyDeviceToDevice, s);
}

void GPUKVPaged::gather_to_linear(half* k_out, half* v_out, int layer, int n_kv, int hd, cudaStream_t s) {
    for(int i=0; i<cur_pos; i++) {
        int p_idx = i / KV_PAGE_TOKENS; int off = i % KV_PAGE_TOKENS;
        KVPage* p = pages[layer][p_idx];
        cudaMemcpyAsync(k_out + i*n_kv*hd, p->d_k + off*n_kv*hd, n_kv*hd*sizeof(half), cudaMemcpyDeviceToDevice, s);
        cudaMemcpyAsync(v_out + i*n_kv*hd, p->d_v + off*n_kv*hd, n_kv*hd*sizeof(half), cudaMemcpyDeviceToDevice, s);
    }
}
