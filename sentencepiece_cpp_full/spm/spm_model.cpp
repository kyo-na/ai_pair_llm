#include "spm_model.h"
#include "../proto/proto_reader.h"
#include <fstream>
#include <iterator>
#include <cstring>
void SPMModel::load(const char* path){std::ifstream f(path,std::ios::binary);std::vector<uint8_t> buf((std::istreambuf_iterator<char>(f)),{});ProtoReader r(buf.data(),buf.size());while(r.p<r.end){uint64_t key=r.read_varint();uint32_t field=key>>3,wt=key&7;if(field==1&&wt==2){uint64_t len=r.read_varint();const uint8_t* e=r.p+len;SentencePiece sp;ProtoReader sr(r.p,len);while(sr.p<sr.end){uint64_t k=sr.read_varint();uint32_t f2=k>>3,w=k&7;if(f2==1&&w==2)sp.piece=sr.read_string();else if(f2==2&&w==5){std::memcpy(&sp.score,sr.p,4);sr.p+=4;}else if(f2==3&&w==0)sp.type=(PieceType)sr.read_varint();else sr.skip_field(w);}if(sp.type==UNKNOWN)unk_id=pieces.size();pieces.push_back(sp);r.p=e;}else r.skip_field(wt);}}
