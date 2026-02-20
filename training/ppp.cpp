// ppo.cpp
// Minimal PPO / RLJF for chat models

#include <vector>
#include <cmath>
#include <algorithm>

struct PPOBuffer {
    std::vector<float> logp_old;
    std::vector<float> logp_new;
    std::vector<float> reward;
    std::vector<float> advantage;
};

// KL divergence (scalar)
float kl_div(float logp_old, float logp_new){
    return std::exp(logp_old) * (logp_old - logp_new);
}

// PPO loss (clipped)
float ppo_loss(
    const PPOBuffer& buf,
    float clip = 0.2f,
    float kl_coef = 0.01f
){
    float loss = 0.f;
    int N = buf.reward.size();

    for(int i=0;i<N;i++){
        float ratio = std::exp(buf.logp_new[i] - buf.logp_old[i]);
        float clipped = std::clamp(ratio, 1.f-clip, 1.f+clip);

        float pg = -std::min(
            ratio * buf.advantage[i],
            clipped * buf.advantage[i]
        );

        float kl = kl_div(buf.logp_old[i], buf.logp_new[i]);

        loss += pg + kl_coef * kl;
    }
    return loss / N;
}
