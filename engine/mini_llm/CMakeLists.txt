cmake_minimum_required(VERSION 3.20)
project(mini_llm LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

if (MSVC)
    add_compile_options(/utf-8)
endif()

include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include)

# OpenMP（任意）
find_package(OpenMP)

add_library(mini_llm_core STATIC
    # blocks
    src/blocks/transformer_block4d.cpp

    # layers
    src/layers/embedding4d.cpp
    src/layers/linear4d.cpp
    src/layers/attention4d.cpp
    src/layers/attention4d_backward.cpp
    src/layers/attention4d_infer_fused.cpp
    src/layers/ffn4d.cpp
    src/layers/layernorm4d.cpp
    src/layers/dropout4d.cpp
    src/layers/linear_vocab4d.cpp
    src/layers/softmax4d.cpp
    src/layers/gelu4d.cpp
    src/layers/rmsnorm4d.cpp
    src/layers/swiglu_ffn4d.cpp
    src/layers/moe_ffn4d.cpp
    src/layers/linear_int8.cpp

    # mask
    src/mask/causal_mask4d.cpp

    # position
    src/position/rope4d.cpp

    # tokenizer
    src/tokenizer/vocab.cpp

    # model
    src/model/transformer_stack4d.cpp
    src/model/transformer_stack_infer4d.cpp

    # loss
    src/loss/cross_entropy4d.cpp

    # optimizer
    src/optimizer/adam.cpp
    src/optimizer/adamw.cpp

    # decode
    src/decode/sampling4d.cpp
    src/decode/repetition_penalty4d.cpp
    src/decode/infer_sampling.cpp

    # train utils
    src/train/label_shift4d.cpp
    src/train/optim/lr_scheduler.cpp
    src/train/optim/grad_clip.cpp

    # cache
    src/cache/kv_cache4d.cpp
    src/cache/kv_cache_ring4d.cpp

    # runtime/memory
    src/memory/memory_arena.cpp
    src/runtime/thread_affinity_win.cpp
    #src/train/trainer4d.cpp
    src/data/dataloader4d.cpp
    src/metrics/ppl4d.cpp
)

target_include_directories(mini_llm_core PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)

if(OpenMP_CXX_FOUND)
    target_link_libraries(mini_llm_core PUBLIC OpenMP::OpenMP_CXX)
endif()

add_executable(train_llm src/train.cpp)
target_link_libraries(train_llm PRIVATE mini_llm_core)

add_executable(infer_llm src/infer.cpp)
target_link_libraries(infer_llm PRIVATE mini_llm_core)