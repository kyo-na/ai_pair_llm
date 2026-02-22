#include "utf8.h"

namespace mini_llm {

std::vector<char32_t> utf8_to_codepoints(const std::string& s) {
    std::vector<char32_t> out;
    for (size_t i=0;i<s.size();) {
        unsigned char c=s[i];
        if((c&0x80)==0){ out.push_back(c); i++; }
        else if((c&0xE0)==0xC0){
            out.push_back(((c&0x1F)<<6)|(s[i+1]&0x3F)); i+=2;
        } else if((c&0xF0)==0xE0){
            out.push_back(((c&0x0F)<<12)|((s[i+1]&0x3F)<<6)|(s[i+2]&0x3F)); i+=3;
        } else {
            out.push_back(((c&0x07)<<18)|((s[i+1]&0x3F)<<12)|((s[i+2]&0x3F)<<6)|(s[i+3]&0x3F)); i+=4;
        }
    }
    return out;
}

std::string codepoint_to_utf8(char32_t cp){
    std::string s;
    if(cp<=0x7F) s.push_back((char)cp);
    else if(cp<=0x7FF){
        s.push_back(0xC0|((cp>>6)&0x1F));
        s.push_back(0x80|(cp&0x3F));
    } else if(cp<=0xFFFF){
        s.push_back(0xE0|((cp>>12)&0x0F));
        s.push_back(0x80|((cp>>6)&0x3F));
        s.push_back(0x80|(cp&0x3F));
    } else {
        s.push_back(0xF0|((cp>>18)&0x07));
        s.push_back(0x80|((cp>>12)&0x3F));
        s.push_back(0x80|((cp>>6)&0x3F));
        s.push_back(0x80|(cp&0x3F));
    }
    return s;
}

}