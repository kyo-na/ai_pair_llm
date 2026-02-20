#include "cognitive_layer.h"

void WorldModel::init_from_prompt(const std::string& prompt) {
    st.summary = prompt;
    st.facts.push_back("System initialized.");
    st.constraints.push_back("Maintain logical consistency. Output in Japanese unless specified.");
}

std::string WorldModel::build_conditioning_prefix() const {
    std::string p = "[WORLD_STATE]\nSUMMARY: " + st.summary + "\nCONSTRAINTS:\n";
    for(auto& c : st.constraints) p += "- " + c + "\n";
    return p + "[/WORLD_STATE]\n\n";
}

void WorldModel::update(const std::string& query, const std::string& answer) {
    if(st.facts.size()>10) st.facts.erase(st.facts.begin());
    st.facts.push_back("Context updated.");
}

CriticScore CriticEngine::evaluate(const std::string& prompt, const std::string& answer) {
    CriticScore s;
    s.coverage = (answer.length() > 10) ? 1.0f : 0.4f;
    s.clarity = (answer.find("不明") == std::string::npos) ? 0.9f : 0.3f;
    s.consistency = 1.0f; s.safety = 1.0f;
    return s;
}

std::string CriticEngine::generate_feedback(const CriticScore& s) {
    if(s.total() < 0.8f) return "回答が短すぎるか、曖昧な表現が含まれています。詳細かつ明確に修正してください。";
    return "Good.";
}

std::string ReviseEngine::revise(const std::string& prompt, const std::string& draft, const CriticScore& score) {
    std::string p = prompt + "\n[DRAFT]:\n" + draft + "\n[FEEDBACK]:\n";
    if(score.coverage < 1.f) p += "- ユーザーの要求に完全に答えていません。\n";
    if(score.clarity < 1.f) p += "- 不確実な表現を避け、論理的に記述してください。\n";
    return p + "\n[REVISED]:\n";
}
