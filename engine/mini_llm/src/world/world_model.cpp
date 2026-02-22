#include "world/world_model.h"
#include <cassert>

// -----------------------------
// constructor
// -----------------------------
WorldModel::WorldModel(int B, int T, int H, int D)
    : transition(D),
      current(B, T, H, D)
{
}

// -----------------------------
// init
// -----------------------------
void WorldModel::init()
{
    current.zero();
}

// -----------------------------
// inject observation
// -----------------------------
void WorldModel::inject_observation(const Tensor4D& obs)
{
    assert(obs.B == current.latent.B);
    assert(obs.T == current.latent.T);
    assert(obs.H == current.latent.H);
    assert(obs.D == current.latent.D);

    current.latent = obs;
}

// -----------------------------
// step forward (world transition)
// -----------------------------
void WorldModel::step_forward()
{
    current.latent = transition.forward(current.latent);
}

// -----------------------------
// backward
// ⚠ TransformerBlock4D は
// backward(x, dout) なので
// current.latent を必ず渡す
// -----------------------------
void WorldModel::backward(const Tensor4D& dloss)
{
    transition.backward(current.latent, dloss);
}

// -----------------------------
// update
// -----------------------------
void WorldModel::update(float lr)
{
    transition.step(lr);
}