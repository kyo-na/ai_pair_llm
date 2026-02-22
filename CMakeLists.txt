cmake_minimum_required(VERSION 3.20)
project(ai_pair_llm)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

include_directories(${PROJECT_SOURCE_DIR})

add_library(engine
    engine/cpu_engine/cpu_engine.cpp
    engine/mini_llm/src/mini_llm.cpp
)

add_library(runtime
    runtime/threads/worker_pool.cpp
)

add_executable(ai_pair_llm
    src/main.cpp
)

target_link_libraries(ai_pair_llm engine runtime)