cmake_minimum_required(VERSION 3.20)
project(attention LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)

add_executable(attention
    main.cpp
    cpu/attention_decode_ref.cpp
    tests/numerical_test.cpp
    cuda/attention_decode_scalar.cu
)

target_include_directories(attention PRIVATE
    common
    kv
    gqa_mqa
    pipeline
)

set_target_properties(attention PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
)
