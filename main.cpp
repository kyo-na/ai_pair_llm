#include <iostream>
#include "mini_llm.h"

int main() {
    MiniLLM model(1000, 64);   // ← 引数は2つ
    model.set_train(false);

    int token = 1;
    for (int i = 0; i < 5; ++i) {
        token = model.infer_next(token);
        std::cout << "token: " << token << std::endl;
    }
    return 0;
}