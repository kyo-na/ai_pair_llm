
#include <cmath>
#include "attention_types.h"

void attention_decode_cpu_ref(
    const double* Q,
    const double* K,
    const double* V,
    double* Out,
    const AttentionParams& p
) {
    for (int h = 0; h < p.num_heads; ++h) {
        double row_max = -1e30;
        double row_sum = 0.0;

        for (int i = 0; i < p.d; ++i)
            Out[h*p.d+i] = 0.0;

        for (int t = 0; t < p.T; ++t) {
            double score = 0.0;
            for (int i = 0; i < p.d; ++i)
                score += Q[h*p.d+i] * K[(h*p.T+t)*p.d+i];

            double new_max = std::max(row_max, score);
            double exp_old = std::exp(row_max - new_max);
            double exp_new = std::exp(score - new_max);

            row_sum = row_sum * exp_old + exp_new;

            for (int i = 0; i < p.d; ++i)
                Out[h*p.d+i] =
                    Out[h*p.d+i] * exp_old +
                    exp_new * V[(h*p.T+t)*p.d+i];

            row_max = new_max;
        }

        for (int i = 0; i < p.d; ++i)
            Out[h*p.d+i] /= row_sum;
    }
}
