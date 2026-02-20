#include "runtime_engine.h"
#include "weight_loader.h"
#include "sentencepiece.h"
#include "cognitive_layer.h"
#include "cpu_math_ref.h" // Include reference math
#include <iostream>
#include <algorithm>

int sample_argmax(const std::vector<float>& logits) {
    return std::distance(logits.begin(), std::max_element(logits.begin(), logits.end()));
}

int main(int argc, char** argv) {
    if(argc < 2) { std::cerr << "Usage: ./ai_pair_llm <model.gguf>\n"; return 1; }

    std::cout << "[Boot] Initializing ai_pair_llm Megalith Engine...\n";
    
    MemoryArena weight_arena;
    weight_arena.allocate_pool(16ULL * 1024 * 1024 * 1024);
    
    GGUFFile gguf;
    gguf.load(argv[1]);
    
    LlamaRuntimeGPU target_model;
    target_model.init(4096, 32, 32, 32000);
    WeightLoader::load(gguf, target_model.w, weight_arena, target_model.stream);

    SentencePiece tokenizer;
    
    WorldModel world;
    CriticEngine critic;
    ReviseEngine revise;

    std::string input;
    while(true) {
        std::cout << "\nUser > ";
        std::getline(std::cin, input);
        if(input == "exit" || input.empty()) break;

        if(world.st.summary.empty()) world.init_from_prompt(input);
        std::string full_prompt = world.build_conditioning_prefix() + "User: " + input + "\nAssistant:";
        
        std::vector<int> tokens = tokenizer.encode(full_prompt);
        if(tokens.empty()) tokens = {1};

        GPUKVPaged draft_kv;
        std::cout << "Thinking (Draft)...\n";
        
        std::string draft_text = "";
        for(int step=0; step<10; step++) {
            std::vector<float> logits;
            target_model.forward_token(tokens.back(), draft_kv, logits);
            int next_tok = sample_argmax(logits);
            tokens.push_back(next_tok);
            draft_text += " token_" + std::to_string(next_tok);
        }

        CriticScore score = critic.evaluate(full_prompt, draft_text);
        if(score.total() < 0.8f) {
            std::cout << "Evaluating (Critic low score " << score.total() << "). Revising...\n";
            std::string feedback = critic.generate_feedback(score);
            std::string revise_prompt = revise.revise(full_prompt, draft_text, score);
            
            GPUKVPaged revise_kv = draft_kv; 
            std::vector<int> r_tokens = tokenizer.encode(revise_prompt);
            if(r_tokens.empty()) r_tokens = {1};
            
            std::string revised_text = "";
            for(int step=0; step<10; step++) {
                std::vector<float> logits;
                target_model.forward_token(r_tokens.back(), revise_kv, logits);
                int next_tok = sample_argmax(logits);
                r_tokens.push_back(next_tok);
                revised_text += " revised_" + std::to_string(next_tok);
            }
            std::cout << "\n[Assistant]:" << revised_text << std::endl;
            world.update(input, revised_text);
        } else {
            std::cout << "\n[Assistant]:" << draft_text << std::endl;
            world.update(input, draft_text);
        }
    }
    return 0;
}
