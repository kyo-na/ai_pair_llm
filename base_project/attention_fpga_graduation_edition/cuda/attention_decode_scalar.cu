
#include <cuda.h>
#include <math.h>
#include "attention_types.h"

__global__ void attention_decode_scalar_kernel(
    const float* Q,
    const float* K,
    const float* V,
    float* Out,
    int d,
    int T
) {
    int head = blockIdx.x;
    int tid = threadIdx.x;

    __shared__ float row_max;
    __shared__ float row_sum;

    if (tid == 0) {
        row_max = -1e30f;
        row_sum = 0.0f;
    }
    __syncthreads();

    float acc = 0.0f;

    for (int t = 0; t < T; ++t) {
        float score = 0.0f;
        for (int i = tid; i < d; i += blockDim.x)
            score += Q[head*d+i] * K[(head*T+t)*d+i];

        __shared__ float buf[256];
        buf[tid] = score;
        __syncthreads();

        for (int s = blockDim.x/2; s > 0; s >>= 1) {
            if (tid < s) buf[tid] += buf[tid+s];
            __syncthreads();
        }

        float s = buf[0];
        float new_max = fmaxf(row_max, s);
        float exp_old = expf(row_max - new_max);
        float exp_new = expf(s - new_max);

        row_sum = row_sum * exp_old + exp_new;
        acc     = acc     * exp_old + exp_new * V[(head*T+t)*d+tid];
        row_max = new_max;
        __syncthreads();
    }

    Out[head*d+tid] = acc / row_sum;
}
