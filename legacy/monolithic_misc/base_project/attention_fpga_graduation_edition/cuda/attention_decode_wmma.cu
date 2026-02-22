
#include <mma.h>
using namespace nvcuda::wmma;

// WMMA-ready skeleton
// QK / QV compute sections replace scalar dot with mma_sync
__global__ void attention_decode_wmma_kernel(...) {
    // load fragments
    // mma_sync
    // online softmax remains scalar
}
