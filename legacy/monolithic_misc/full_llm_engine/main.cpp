
#include <iostream>
#include "core/tensor.h"

extern Tensor transformer(Tensor,int);
extern void train_bpe();
extern void pretrain();
extern void flash_attention();

int main(){

    train_bpe();
    pretrain();
    flash_attention();

    Tensor x(1,8);
    Tensor y=transformer(x,0);

    std::cout<<"Output: ";
    for(float v:y.data) std::cout<<v<<" ";
    std::cout<<"\n";

    return 0;
}
