#include "runtime_engine.h"
#include "gpu_kernels.cuh"
#include <algorithm>

void LlamaRuntimeGPU::init(int d, int h, int l, int v) {
    D=d; H=h; Hd=d/h; L=l; V=v; FF=D*4;
    cudaStreamCreate(&stream);
    arena.allocate_pool(8ULL * 1024 * 1024 * 1024);
    kv_pool.init(4096, H, Hd);
}

void LlamaRuntimeGPU::forward_token(int token, GPUKVPaged& kv, std::vector<float>& logits_out) {
    size_t start_offset = arena.get_current_offset();
    
    half* d_x = arena.allocate<half>(D);
    half* d_q = arena.allocate<half>(D); half* d_k = arena.allocate<half>(H*Hd); half* d_v = arena.allocate<half>(H*Hd);
    half* d_attn = arena.allocate<half>(D);
    half* d_up = arena.allocate<half>(FF); half* d_gate = arena.allocate<half>(FF);
    half* d_logits = arena.allocate<half>(V);

    get_embedding_kernel<<<D/256+1, 256, 0, stream>>>((half*)w.token_embd, token, d_x, D);

    for(int l=0; l<w.n_layers; l++) {
        rmsnorm_kernel<<<1, D, 0, stream>>>(d_x, (float*)w.layers[l].attn_norm, d_attn, D, 1e-6f);
        q4k_gemv_kernel<<<D, 1, 0, stream>>>((Q4KBlock*)w.layers[l].attn_q, d_attn, d_q, D, D);
        q4k_gemv_kernel<<<H*Hd, 1, 0, stream>>>((Q4KBlock*)w.layers[l].attn_k, d_attn, d_k, H*Hd, D);
        q4k_gemv_kernel<<<H*Hd, 1, 0, stream>>>((Q4KBlock*)w.layers[l].attn_v, d_attn, d_v, H*Hd, D);

        rope_kernel<<<1, Hd, 0, stream>>>(d_q, Hd, kv.cur_pos, 10000.f);
        rope_kernel<<<1, Hd, 0, stream>>>(d_k, Hd, kv.cur_pos, 10000.f);

        kv.append_kv(&kv_pool, l, d_k, d_v, H, Hd, stream);

        half* linear_k = arena.allocate<half>((kv.cur_pos+1)*H*Hd);
        half* linear_v = arena.allocate<half>((kv.cur_pos+1)*H*Hd);
        kv.gather_to_linear(linear_k, linear_v, l, H, Hd, stream);
        
        causal_attention_kernel<<<H, std::max(kv.cur_pos+1, Hd), 0, stream>>>(d_q, linear_k, linear_v, d_q, kv.cur_pos+1, Hd);
        q4k_gemv_kernel<<<D, 1, 0, stream>>>((Q4KBlock*)w.layers[l].attn_output, d_q, d_attn, D, D);
        residual_add_kernel<<<D/256+1, 256, 0, stream>>>(d_x, d_attn, D);

        rmsnorm_kernel<<<1, D, 0, stream>>>(d_x, (float*)w.layers[l].ffn_norm, d_attn, D, 1e-6f);
        q4k_gemv_kernel<<<FF, 1, 0, stream>>>((Q4KBlock*)w.layers[l].ffn_up, d_attn, d_up, FF, D);
        q4k_gemv_kernel<<<FF, 1, 0, stream>>>((Q4KBlock*)w.layers[l].ffn_gate, d_attn, d_gate, FF, D);
        silu_mul_kernel<<<FF/256+1, 256, 0, stream>>>(d_up, d_gate, d_up, FF);
        q4k_gemv_kernel<<<D, 1, 0, stream>>>((Q4KBlock*)w.layers[l].ffn_down, d_up, d_attn, D, FF);
        residual_add_kernel<<<D/256+1, 256, 0, stream>>>(d_x, d_attn, D);
    }

    rmsnorm_kernel<<<1, D, 0, stream>>>(d_x, (float*)w.output_norm, d_x, D, 1e-6f);
    q4k_gemv_kernel<<<V, 1, 0, stream>>>((Q4KBlock*)w.output_weight, d_x, d_logits, V, D);

    std::vector<half> h_logits(V);
    cudaMemcpyAsync(h_logits.data(), d_logits, V*sizeof(half), cudaMemcpyDeviceToHost, stream);
    cudaStreamSynchronize(stream);

    logits_out.resize(V);
    for(int i=0; i<V; i++) logits_out[i] = __half2float(h_logits[i]);

    kv.cur_pos++;
    arena.reset_to(start_offset);
}
