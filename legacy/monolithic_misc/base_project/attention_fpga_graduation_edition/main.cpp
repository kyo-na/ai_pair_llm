
#include <iostream>
#include "pipeline/pipeline.h"

int main() {
    Backend backend = select_backend();

    if (backend == Backend::CUDA_GPU)
        std::cout << "Running CUDA GPU backend" << std::endl;
    else
        std::cout << "Running FPGA backend" << std::endl;

    return 0;
}
