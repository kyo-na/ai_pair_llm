#pragma once

#include <string>
#include "engine/mini_llm/api/mini_llm_api.h"

class CpuEngine {
public:
    CpuEngine();
    std::string infer(const std::string& text);

private:
    MiniLLM llm_;
};