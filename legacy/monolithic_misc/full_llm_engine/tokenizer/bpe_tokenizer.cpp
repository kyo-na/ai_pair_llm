
#include <string>
#include <vector>

std::vector<int> tokenize(const std::string& s){
    std::vector<int> t;
    for(char c:s) t.push_back((int)c);
    return t;
}

std::string detokenize(const std::vector<int>& t){
    std::string s;
    for(int i:t) s.push_back((char)i);
    return s;
}
