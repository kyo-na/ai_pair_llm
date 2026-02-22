#pragma once
#include <cuda_runtime.h>
#include <cstdint>
#include <stdexcept>
class MemoryArena {
    uint8_t* base_ptr_ = nullptr;
    size_t total_size_ = 0;
    size_t current_offset_ = 0;
    static constexpr size_t ALIGNMENT = 256;
    size_t align_up(size_t s) const { return (s + ALIGNMENT - 1) & ~(ALIGNMENT - 1); }
public:
    ~MemoryArena(){ if(base_ptr_) cudaFree(base_ptr_); }
    void allocate_pool(size_t bytes){
        if(base_ptr_) cudaFree(base_ptr_);
        if(cudaMalloc(&base_ptr_, bytes)!=cudaSuccess) throw std::runtime_error("cudaMalloc failed");
        total_size_=bytes; current_offset_=0;
    }
    template<typename T> T* allocate(size_t n){
        size_t b=align_up(n*sizeof(T));
        if(current_offset_+b>total_size_) throw std::runtime_error("OOM");
        T* p=(T*)(base_ptr_+current_offset_);
        current_offset_+=b; return p;
    }
    void reset_to(size_t o){ current_offset_=o; }
    size_t get_current_offset() const { return current_offset_; }
};
