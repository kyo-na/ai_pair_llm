#pragma once
#include <string>
#include <vector>
#include <unordered_map>

class SentencePiece {
    struct SPToken { std::string piece; int id; float score; };
    std::vector<SPToken> vocab_;
    std::unordered_map<std::string, int> piece_to_id_;
    int bos_ = 1, eos_ = 2, unk_ = 0;
public:
    bool load(const std::string& path);
    std::vector<int> encode(const std::string& text) const;
    std::string decode(const std::vector<int>& ids) const;
    int bos_id() const { return bos_; }
    int eos_id() const { return eos_; }
};
