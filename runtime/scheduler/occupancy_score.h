// runtime/scheduler/occupancy_score.h
#pragma once

struct DeviceScore {
    float cpu_score;
    float gpu_score;
};

inline DeviceScore compute_device_score(
    float cpu_usage,      // 0.0 - 1.0
    int   cpu_queue_len,
    float gpu_usage,      // 0.0 - 1.0
    int   gpu_queue_len,
    size_t vram_free_mb
) {
    DeviceScore s{};

    // CPU: 使用率 + キュー長
    s.cpu_score =
        cpu_usage * 0.6f +
        (cpu_queue_len * 0.1f);

    // GPU: 使用率 + キュー長 + VRAM圧迫
    float vram_penalty = (vram_free_mb < 512) ? 1.0f : 0.0f;

    s.gpu_score =
        gpu_usage * 0.6f +
        (gpu_queue_len * 0.2f) +
        vram_penalty;

    return s;
}