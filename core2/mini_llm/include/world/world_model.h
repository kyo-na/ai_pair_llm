#pragma once
#include "world/world_state.h"
#include "blocks/transformer_block4d.h"

struct WorldModel {

    TransformerBlock4D transition;
    WorldState current;

    WorldModel(int B, int T, int H, int D);

    void init();
    void inject_observation(const Tensor4D& obs);
    void step_forward();

    // World loss からの勾配
    void backward(const Tensor4D& dloss);

    void update(float lr);
};