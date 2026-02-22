#pragma once
#include <string>
#include <vector>

struct WorldState {
    std::string summary;
    std::vector<std::string> facts;
    std::vector<std::string> constraints;
};

class WorldModel {
public:
    WorldState st;
    void init_from_prompt(const std::string& prompt);
    std::string build_conditioning_prefix() const;
    void update(const std::string& query, const std::string& answer);
};

struct CriticScore {
    float coverage = 0.f; float clarity = 0.f; float consistency = 0.f; float safety = 0.f;
    float total() const { return 0.25f*(coverage+clarity+consistency+safety); }
};

class CriticEngine {
public:
    CriticScore evaluate(const std::string& prompt, const std::string& answer);
    std::string generate_feedback(const CriticScore& s);
};

class ReviseEngine {
public:
    std::string revise(const std::string& prompt, const std::string& draft, const CriticScore& score);
};
