#include "memory_arena.h"
#include <iostream>

size_t MemoryArena::align_up(size_t size) const {
    return (size + ALIGNMENT - 1) & ~(ALIGNMENT - 1);
}

MemoryArena::~MemoryArena() {
    if (base_ptr_) cudaFree(base_ptr_);
}

void MemoryArena::allocate_pool(size_t size_in_bytes) {
    if (base_ptr_) cudaFree(base_ptr_);
    cudaError_t err = cudaMalloc(reinterpret_cast<void**>(&base_ptr_), size_in_bytes);
    if (err != cudaSuccess) throw std::runtime_error("cudaMalloc failed");
    total_size_ = size_in_bytes;
    current_offset_ = 0;
    std::cout << "[MemoryArena] Allocated " << (size_in_bytes / (1024*1024)) << " MB.\n";
}

void MemoryArena::reset_to(size_t offset) {
    if (offset > total_size_) throw std::out_of_range("Invalid offset");
    current_offset_ = offset;
}

size_t MemoryArena::get_current_offset() const {
    return current_offset_;
}
