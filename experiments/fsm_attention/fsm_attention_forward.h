
// fsm_attention_forward.h
#pragma once
#include <vector>
#include <limits>
#include <cmath>

Tensor fsm_attention_forward(Tensor& Q, Tensor& K, Tensor& V) {
    int B = Q.shape[0];
    int D = Q.shape[1];
    Tensor O({B, D}, false);
    float scale = 1.0f / std::sqrt((float)D);

    for (int i = 0; i < B; ++i) {
        float best1 = -1e30f, best2 = -1e30f;
        int idx1=-1, idx2=-1;
        for (int j = 0; j < B; ++j) {
            float s = 0.f;
            for (int d = 0; d < D; ++d)
                s += Q.data[i*D+d]*K.data[j*D+d];
            s *= scale;
            if (s > best1) { best2=best1; idx2=idx1; best1=s; idx1=j; }
            else if (s > best2) { best2=s; idx2=j; }
        }
        for (int d = 0; d < D; ++d) {
            float v = 0.f;
            if (idx1>=0) v += V.data[idx1*D+d];
            if (idx2>=0) v += V.data[idx2*D+d];
            O.data[i*D+d] = 0.5f*v;
        }
    }
    return O;
}
