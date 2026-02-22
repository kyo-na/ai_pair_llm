#include "loss/cross_entropy.h"
#include <cmath>

float cross_entropy(const Tensor4D& logits,
                    const std::vector<int>& targets,
                    int vocab_size)
{
    float loss = 0.0f;

    for (int t = 0; t < targets.size(); ++t) {
        int base = t * vocab_size;

        float maxv = logits.data[base];
        for (int i = 1; i < vocab_size; ++i)
            maxv = std::max(maxv, logits.data[base+i]);

        float sum = 0.0f;
        for (int i = 0; i < vocab_size; ++i)
            sum += std::exp(logits.data[base+i] - maxv);

        float logprob = logits.data[base + targets[t]] - maxv - std::log(sum);
        loss -= logprob;
    }

    return loss / targets.size();
}