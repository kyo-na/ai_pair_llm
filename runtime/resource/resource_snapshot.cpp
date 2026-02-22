// runtime/resource/resource_snapshot.cpp

#include "resource_snapshot.h"

static ULONGLONG filetime_to_uint64(const FILETIME& ft) {
    return (static_cast<ULONGLONG>(ft.dwHighDateTime) << 32)
         | ft.dwLowDateTime;
}

ResourceSnapshot ResourceSnapshot::capture() {
    ResourceSnapshot snap{};

    SYSTEM_INFO si;
    GetSystemInfo(&si);
    snap.cpu_cores = si.dwNumberOfProcessors;

    static ULONGLONG prev_idle = 0;
    static ULONGLONG prev_kernel = 0;
    static ULONGLONG prev_user = 0;

    FILETIME idle_time, kernel_time, user_time;
    GetSystemTimes(&idle_time, &kernel_time, &user_time);

    ULONGLONG idle = filetime_to_uint64(idle_time);
    ULONGLONG kernel = filetime_to_uint64(kernel_time);
    ULONGLONG user = filetime_to_uint64(user_time);

    ULONGLONG delta_idle   = idle   - prev_idle;
    ULONGLONG delta_kernel = kernel - prev_kernel;
    ULONGLONG delta_user   = user   - prev_user;

    ULONGLONG delta_total = delta_kernel + delta_user;

    if (delta_total > 0) {
        snap.cpu_usage =
            1.0f - (float)delta_idle / (float)delta_total;
    } else {
        snap.cpu_usage = 0.0f;
    }

    prev_idle   = idle;
    prev_kernel = kernel;
    prev_user   = user;

    // GPUは今はstub
    snap.gpu_usage = 0.0f;
    snap.vram_free_mb = 4096;

    return snap;
}