#pragma once
#include "engine/mini_llm/api/mini_llm_api.h"

struct OccupancyScore {
    float load;
    float latency;
    bool available;
};

class EngineBase {
public:
    virtual ~EngineBase() = default;
    virtual OccupancyScore score() const = 0;
    virtual void submit(const MiniLLMTask& task) = 0;
    virtual bool try_fetch(MiniLLMResult& result) = 0;
};