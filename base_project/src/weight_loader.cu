#include "weight_loader.h"
#include <iostream>

void WeightLoader::load(const GGUFFile& gguf, LlamaWeights& w, MemoryArena& arena, cudaStream_t stream) {
    auto it = gguf.tensors.find("token_embd.weight");
    if(it == gguf.tensors.end()) throw std::runtime_error("token_embd missing");
    w.n_vocab = it->second.dims[1]; w.n_embd = it->second.dims[0];
    
    w.n_layers = 0;
    while(gguf.tensors.find("blk." + std::to_string(w.n_layers) + ".attn_q.weight") != gguf.tensors.end()) w.n_layers++;
    w.layers.resize(w.n_layers);
    
    auto load_tensor = [&](const std::string& name, void*& ptr) {
        auto t_it = gguf.tensors.find(name);
        if(t_it != gguf.tensors.end()) {
            size_t elements = 1; for(auto d : t_it->second.dims) elements *= d;
            size_t bytes = 0;
            if(t_it->second.type == 0) bytes = elements * 4;
            else if(t_it->second.type == 1) bytes = elements * 2;
            else if(t_it->second.type == 12) bytes = (elements / 256) * 144;
            
            ptr = arena.allocate<uint8_t>(bytes);
            cudaMemcpyAsync(ptr, gguf.data + gguf.data_offset + t_it->second.offset, bytes, cudaMemcpyHostToDevice, stream);
        } else { ptr = nullptr; }
    };

    load_tensor("token_embd.weight", w.token_embd);
    for(int i=0; i<w.n_layers; i++) {
        std::string p = "blk." + std::to_string(i) + ".";
        load_tensor(p+"attn_q.weight", w.layers[i].attn_q);
        load_tensor(p+"attn_k.weight", w.layers[i].attn_k);
        load_tensor(p+"attn_v.weight", w.layers[i].attn_v);
        load_tensor(p+"attn_output.weight", w.layers[i].attn_output);
        load_tensor(p+"ffn_gate.weight", w.layers[i].ffn_gate);
        load_tensor(p+"ffn_down.weight", w.layers[i].ffn_down);
        load_tensor(p+"ffn_up.weight", w.layers[i].ffn_up);
        load_tensor(p+"attn_norm.weight", w.layers[i].attn_norm);
        load_tensor(p+"ffn_norm.weight", w.layers[i].ffn_norm);
    }
    load_tensor("output_norm.weight", w.output_norm);
    load_tensor("output.weight", w.output_weight);
    cudaStreamSynchronize(stream);
}
