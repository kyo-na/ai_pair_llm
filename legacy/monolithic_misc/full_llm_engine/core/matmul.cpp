
#include "tensor.h"

Tensor matmul(const Tensor& A,const Tensor& B){
    Tensor C(A.rows,B.cols);
    for(int i=0;i<A.rows;i++)
        for(int j=0;j<B.cols;j++){
            float s=0;
            for(int k=0;k<A.cols;k++)
                s+=A.data[i*A.cols+k]*B.data[k*B.cols+j];
            C(i,j)=s;
        }
    return C;
}
