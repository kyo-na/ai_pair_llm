
#include "tensor.h"

extern Tensor matmul(const Tensor&,const Tensor&);
extern void rmsnorm(Tensor&);
extern void rope(Tensor&,int);
extern Tensor attention(const Tensor&,const Tensor&,const Tensor&);
extern Tensor ffn(const Tensor&);

Tensor transformer(Tensor x,int pos){
    rmsnorm(x);

    Tensor q=x,k=x,v=x;
    rope(q,pos); rope(k,pos);

    Tensor a=attention(q,k,v);
    x=a;

    rmsnorm(x);
    x=ffn(x);

    return x;
}
