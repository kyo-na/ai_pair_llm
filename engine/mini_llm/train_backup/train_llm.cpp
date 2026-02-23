#include <iostream>
#include <fstream>
#include <string>
#include "../include/model/embedding.h"
#include "../include/model/linear.h"
#include "../include/loss/mse4d.h"
#include "../include/optimizer/adam.h"

int main(){
    Embedding emb(256,64);
    Linear proj(64,64);

    std::ifstream fin("../data/shard_000.txt");
    if(!fin){ std::cerr<<"no data\n"; return 1; }

    int step=1;
    std::string line;
    while(std::getline(fin,line)){
        for(size_t i=0;i+1<line.size();i++){
            int a=(unsigned char)line[i];
            int b=(unsigned char)line[i+1];

            auto x=emb.forward(a);
            auto y=proj.forward(x);
            auto t=emb.forward(b);

            auto gy=mse_grad(y,t);
            auto gx=proj.backward(gy);
            emb.backward(a,gx);

            adam_update(emb.weight,1e-3f,step);
            adam_update(proj.weight,1e-3f,step);

            if(step%10000==0)
                std::cout<<"step "<<step<<"\n";
            if(step++>500000) return 0;
        }
    }
}