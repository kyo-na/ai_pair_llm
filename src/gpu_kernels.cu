#include "gpu_kernels.cuh"
#include <math_constants.h>

__global__ void get_embedding_kernel(const half* embd, int token_id, half* out, int D) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if(tid < D) out[tid] = embd[token_id * D + tid];
}

__global__ void rmsnorm_kernel(const half* x, const float* w, half* y, int D, float eps) {
    int tid = threadIdx.x;
    float val = tid < D ? __half2float(x[tid]) : 0.f;
    float sq = val * val;
    for(int offset=16; offset>0; offset/=2) sq += __shfl_down_sync(0xffffffff, sq, offset);
    __shared__ float ssum; if(tid==0) ssum = rsqrtf(sq/D + eps);
    __syncthreads();
    if(tid < D) y[tid] = __float2half(val * ssum * (w ? w[tid] : 1.f));
}

__global__ void q4k_gemv_kernel(const Q4KBlock* W, const half* x, half* y, int rows, int cols) {
    int r = blockIdx.x; if(r >= rows) return;
    float acc = 0.f; int b_per_row = cols / Q4K_GROUP;
    for(int b=0; b<b_per_row; b++) {
        const Q4KBlock& blk = W[r * b_per_row + b]; float s = blk.scale;
        for(int i=0; i<Q4K_GROUP/2; i++) {
            uint8_t q = blk.qs[i];
            int8_t q0 = (q & 0x0F) - 8; int8_t q1 = ((q >> 4) & 0x0F) - 8;
            int idx = b * Q4K_GROUP + i*2;
            acc += q0 * s * __half2float(x[idx]);
            acc += q1 * s * __half2float(x[idx+1]);
        }
    }
    if(threadIdx.x == 0) y[r] = __float2half(acc);
}

__global__ void rope_kernel(half* qk, int Hd, int pos, float base) {
    int i = threadIdx.x;
    if(i < Hd/2) {
        float inv = powf(base, -2.f*i/Hd); float th = pos * inv;
        float c = cosf(th), s = sinf(th);
        float a = __half2float(qk[2*i]), b = __half2float(qk[2*i+1]);
        qk[2*i] = __float2half(a*c - b*s); qk[2*i+1] = __float2half(a*s + b*c);
    }
}

__global__ void causal_attention_kernel(const half* q, const half* k_cache, const half* v_cache, half* out, int T, int Hd) {
    int h = blockIdx.x; int tid = threadIdx.x;
    __shared__ float scores[4096]; __shared__ float m, sum_exp;
    if(tid < T) {
        float acc = 0.f;
        for(int d=0; d<Hd; d++) acc += __half2float(q[h*Hd+d]) * __half2float(k_cache[(h*T + tid)*Hd + d]);
        scores[tid] = acc / sqrtf((float)Hd);
    }
    __syncthreads();
    if(tid == 0) {
        m = -CUDART_INF_F; for(int i=0; i<T; i++) m = fmaxf(m, scores[i]);
        sum_exp = 0.f; for(int i=0; i<T; i++) sum_exp += expf(scores[i] - m);
    }
    __syncthreads();
    if(tid < Hd) {
        float acc = 0.f;
        for(int t=0; t<T; t++) {
            float w = expf(scores[t] - m) / sum_exp;
            acc += w * __half2float(v_cache[(h*T + t)*Hd + tid]);
        }
        out[h*Hd + tid] = __float2half(acc);
    }
}

__global__ void silu_mul_kernel(const half* up, const half* gate, half* out, int D) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if(tid < D) {
        float g = __half2float(gate[tid]);
        float silu = g / (1.0f + expf(-g));
        out[tid] = __float2half(silu * __half2float(up[tid]));
    }
}

__global__ void residual_add_kernel(half* x, const half* res, int D) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if(tid < D) x[tid] = __float2half(__half2float(x[tid]) + __half2float(res[tid]));
}
