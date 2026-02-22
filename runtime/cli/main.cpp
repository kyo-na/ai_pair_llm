
#include <iostream>
#include "../core/model.h"
int main(){
    Model m(256); KVCache kv;
    auto l=m.forward('a',kv);
    std::cout<<"OK logit[a]="<<l['a']<<"\n";
}
