// runtime/resource/resource_snapshot.h
#pragma once
#include <windows.h>
#include <cstdint>

struct ResourceSnapshot {
    float cpu_usage;      // 0.0 - 1.0
    uint32_t cpu_cores;

    float gpu_usage;      // stub
    size_t vram_free_mb;  // stub

    static ResourceSnapshot capture();
};