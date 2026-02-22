cmake_minimum_required(VERSION 3.20)
project(ai_pair_llm CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_ARCHITECTURES 80 86 89 90)

add_executable(ai_pair_llm
    src/main.cpp
    src/memory_arena.cu
    src/gguf_parser.cpp
    src/sentencepiece.cpp
    src/weight_loader.cu
    src/gpu_kernels.cu
    src/kv_cache.cu
    src/cognitive_layer.cpp
    src/runtime_engine.cu
    src/cpu_math_ref.cpp
)

target_include_directories(ai_pair_llm PRIVATE include)
target_link_libraries(ai_pair_llm PRIVATE cudart)
