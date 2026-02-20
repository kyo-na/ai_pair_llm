// loss_mask.cpp
// assistant-only loss mask for chat training

#include <vector>
#include <string>

// 0 = ignore, 1 = apply loss
std::vector<int> build_assistant_loss_mask(
    const std::vector<int>& tokens,
    const std::string& decoded
){
    std::vector<int> mask(tokens.size(), 0);

    bool in_assistant = false;
    for(size_t i = 0; i < decoded.size(); ++i){
        // enter assistant
        if(decoded.substr(i, 11) == "<assistant>"){
            in_assistant = true;
        }
        // exit assistant
        if(decoded.substr(i, 12) == "</assistant>"){
            in_assistant = false;
        }
        if(in_assistant){
            mask[i] = 1;
        }
    }
    return mask;
}
