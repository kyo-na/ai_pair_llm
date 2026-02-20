
#include "pipeline.h"

bool detect_nvidia_gpu() {
    return true; // replace with cudaGetDeviceCount()
}

Backend select_backend() {
    if (detect_nvidia_gpu())
        return Backend::CUDA_GPU;
    return Backend::CUDA_FPGA;
}
