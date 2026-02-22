#include "tokenizer.h"
#include "utf8.h"

namespace mini_llm {
std::vector<char32_t> Tokenizer::encode(const std::string& s) {
    return utf8_to_codepoints(s);
}
}