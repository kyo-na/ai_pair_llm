#include "proto_reader.h"
#include <stdexcept>
ProtoReader::ProtoReader(const uint8_t* d,size_t sz):p(d),end(d+sz){}
uint64_t ProtoReader::read_varint(){uint64_t v=0;int s=0;while(p<end){uint8_t b=*p++;v|=uint64_t(b&0x7F)<<s;if(!(b&0x80))return v;s+=7;}throw std::runtime_error("varint overflow");}
std::string ProtoReader::read_string(){uint64_t l=read_varint();std::string s((const char*)p,l);p+=l;return s;}
void ProtoReader::skip_field(uint32_t wt){if(wt==0)read_varint();else if(wt==2){uint64_t l=read_varint();p+=l;}}
