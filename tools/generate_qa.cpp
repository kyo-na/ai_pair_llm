#include <iostream>
#include <fstream>
#include <string>
#include <sstream>

// ----------------------------
// 構造体定義
// ----------------------------
struct LifeEvent {
    std::string thought;
    std::string decision;
    std::string reflection;
};

struct QA {
    std::string q;
    std::string a;
    std::string tone;
    std::string source;
};

// ----------------------------
// 超簡易 JSON 抽出（key:"value" 前提）
// ※ structured.jsonl はこの形式で作っているため成立
// ----------------------------
std::string extract(const std::string& line, const std::string& key) {
    std::string pattern = "\"" + key + "\":\"";
    auto pos = line.find(pattern);
    if (pos == std::string::npos) return "";
    pos += pattern.size();
    auto end = line.find("\"", pos);
    return line.substr(pos, end - pos);
}

// ----------------------------
// QA生成ロジック（あなたの定義そのまま）
// ----------------------------
QA generateQA(const LifeEvent& e) {
    QA qa;
    qa.q = "なぜ" + e.decision + "のですか？";
    qa.a =
        "当時は「" + e.thought + "」と感じていました。"
        "今振り返ると「" + e.reflection + "」と理解しています。";
    qa.tone = "reflective";
    qa.source = "structured:auto";
    return qa;
}

// ----------------------------
// メイン
// ----------------------------
int main() {
    std::ifstream in("C:/Users/spenc/Downloads/ai_pair_llm/dataset/structured/kyosuke_life_events.jsonl");
    if (!in) {
        std::cerr << "structured file not found\n";
        return 1;
    }

    std::ofstream out("C:/Users/spenc/Downloads/ai_pair_llm/tools/kyosuke_identity_qa.jsonl");
    if (!out) {
        std::cerr << "qa output file not found\n";
        return 1;
    }

    std::string line;
    while (std::getline(in, line)) {
        LifeEvent e;
        e.thought     = extract(line, "thought");
        e.decision    = extract(line, "decision");
        e.reflection  = extract(line, "reflection_after_recovery");

        if (e.thought.empty() || e.decision.empty() || e.reflection.empty())
            continue;

        QA qa = generateQA(e);

        out
            << "{"
            << "\"q\":\"" << qa.q << "\","
            << "\"a\":\"" << qa.a << "\","
            << "\"tone\":\"" << qa.tone << "\","
            << "\"source\":\"" << qa.source << "\""
            << "}\n";
    }

    std::cout << "QA generation completed.\n";
    return 0;
}