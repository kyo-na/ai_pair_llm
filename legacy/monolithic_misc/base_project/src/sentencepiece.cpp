#include "sentencepiece.h"
#include <fstream>
#include <algorithm>

bool SentencePiece::load(const std::string& path) {
    std::ifstream f(path, std::ios::binary);
    if(!f) return false;
    uint32_t magic; f.read((char*)&magic, 4);
    if(magic != 0x0a0d0d0a) f.seekg(0);
    uint32_t v_size; f.read((char*)&v_size, 4);
    vocab_.reserve(v_size);
    for(uint32_t i=0; i<v_size; i++) {
        uint32_t len; f.read((char*)&len, 4);
        std::string p(len, '\0'); f.read(&p[0], len);
        float score; f.read((char*)&score, 4);
        uint32_t type; f.read((char*)&type, 4);
        vocab_.push_back({p, (int)i, score});
        piece_to_id_[p] = i;
        if(p == "<unk>") unk_ = i;
        if(p == "<s>") bos_ = i;
        if(p == "</s>") eos_ = i;
    }
    return true;
}

std::vector<int> SentencePiece::encode(const std::string& text) const {
    std::vector<std::string> chars;
    for(size_t i=0; i<text.size();) {
        unsigned char c = text[i]; size_t len = 1;
        if((c&0xE0)==0xC0) len=2; else if((c&0xF0)==0xE0) len=3; else if((c&0xF8)==0xF0) len=4;
        chars.push_back(text.substr(i, len)); i += len;
    }
    int N = chars.size();
    std::vector<float> dp(N+1, 1e9f); std::vector<int> prev(N+1, -1), pid(N+1, -1);
    dp[0] = 0.f;
    for(int i=0; i<N; i++) {
        if(dp[i] >= 1e9f) continue;
        std::string s;
        for(int j=i; j<N; j++) {
            s += chars[j];
            auto it = piece_to_id_.find(s);
            if(it != piece_to_id_.end()) {
                float sc = -vocab_[it->second].score;
                if(dp[j+1] > dp[i] + sc) { dp[j+1] = dp[i] + sc; prev[j+1] = i; pid[j+1] = it->second; }
            }
        }
    }
    std::vector<int> ids;
    for(int cur=N; cur>0;) {
        if(prev[cur]<0) { ids.push_back(unk_); cur--; }
        else { ids.push_back(pid[cur]); cur = prev[cur]; }
    }
    std::reverse(ids.begin(), ids.end());
    ids.insert(ids.begin(), bos_);
    return ids;
}

std::string SentencePiece::decode(const std::vector<int>& ids) const {
    std::string out;
    for(int id : ids) {
        if(id == bos_ || id == eos_ || id < 0 || id >= vocab_.size()) continue;
        out += vocab_[id].piece;
    }
    return out;
}
