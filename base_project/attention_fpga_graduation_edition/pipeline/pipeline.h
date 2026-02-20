
#pragma once

enum class Backend {
    CUDA_GPU,
    CUDA_FPGA
};

Backend select_backend();
