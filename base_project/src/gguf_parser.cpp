#include "gguf_parser.h"
#include <iostream>
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>
#include <cstring>
#include <stdexcept>

void GGUFFile::load(const std::string& path) {
    fd = open(path.c_str(), O_RDONLY);
    if(fd < 0) throw std::runtime_error("Failed to open GGUF");
    struct stat st; fstat(fd, &st); size = st.st_size;
    data = (uint8_t*)mmap(nullptr, size, PROT_READ, MAP_PRIVATE, fd, 0);
    
    uint8_t* p = data;
    if(memcmp(p, "GGUF", 4)!=0) throw std::runtime_error("Not GGUF");
    p += 8;
    
    uint64_t n_tensors, n_kv;
    memcpy(&n_tensors, p, 8); p += 8;
    memcpy(&n_kv, p, 8); p += 8;
    
    auto read_str = [&]() {
        uint64_t len; memcpy(&len, p, 8); p += 8;
        std::string s((char*)p, len); p += len; return s;
    };
    
    for(uint64_t i=0; i<n_kv; i++) {
        read_str();
        uint32_t type; memcpy(&type, p, 4); p += 4;
        if(type==8) { read_str(); }
        else if(type==9) {
            uint32_t et; memcpy(&et, p, 4); p+=4;
            uint64_t n; memcpy(&n, p, 8); p+=8;
            if(et==8){ for(uint64_t j=0; j<n; j++) read_str(); } else p += n*4;
        }
        else if(type==10) p+=8;
        else if(type==6 || type==4 || type==5) p+=4;
        else p+=1;
    }
    
    for(uint64_t i=0; i<n_tensors; i++) {
        std::string name = read_str();
        uint32_t nd; memcpy(&nd, p, 4); p+=4;
        std::vector<uint64_t> dims(nd);
        for(uint32_t j=0; j<nd; j++) { memcpy(&dims[j], p, 8); p+=8; }
        uint32_t type; memcpy(&type, p, 4); p+=4;
        uint64_t offset; memcpy(&offset, p, 8); p+=8;
        tensors[name] = {dims, type, offset};
    }
    size_t header_size = p - data;
    data_offset = (header_size + 31) & ~31;
}

GGUFFile::~GGUFFile() {
    if(data) munmap(data, size);
    if(fd >= 0) close(fd);
}
