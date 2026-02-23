#include "../include/tensor4d.h"
#include <fstream>

Tensor4D::Tensor4D(int b,int t,int c,int d)
    : B(b),T(t),C(c),D(d),
      data(b*t*c*d,0.0f),
      grad(b*t*c*d,0.0f) {}

void Tensor4D::zero_grad(){
    std::fill(grad.begin(),grad.end(),0.0f);
}

void Tensor4D::save(const char* path) const{
    std::ofstream o(path,std::ios::binary);
    o.write((char*)&B,4);
    o.write((char*)&T,4);
    o.write((char*)&C,4);
    o.write((char*)&D,4);
    o.write((char*)data.data(),data.size()*4);
}

void Tensor4D::load(const char* path){
    std::ifstream i(path,std::ios::binary);
    i.read((char*)&B,4);
    i.read((char*)&T,4);
    i.read((char*)&C,4);
    i.read((char*)&D,4);
    data.resize(B*T*C*D);
    grad.resize(B*T*C*D);
    i.read((char*)data.data(),data.size()*4);
}