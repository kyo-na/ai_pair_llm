#include "../spm/spm_model.h"
#include "../spm/spm_tokenizer.h"
#include <iostream>
int main(){SPMModel m;m.load("tokenizer.model");SPMTokenizer tok(m);auto ids=tok.encode("こんにちは");for(int i:ids)std::cout<<i<<" ";std::cout<<"\n";}
