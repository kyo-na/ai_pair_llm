
#include "tensor.h"
#include <cmath>

void rope(Tensor& q,int pos){
    for(int i=0;i<q.cols;i+=2){
        float angle=pos*0.01f*i;
        float x=q(0,i), y=q(0,i+1);
        q(0,i)=x*cos(angle)-y*sin(angle);
        q(0,i+1)=x*sin(angle)+y*cos(angle);
    }
}
