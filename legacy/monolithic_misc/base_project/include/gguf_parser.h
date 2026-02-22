#pragma once
#include <string>
#include <vector>
#include <unordered_map>
#include <cstdint>

struct GGUFTensor {
    std::vector<uint64_t> dims;
    uint32_t type;
    uint64_t offset;
};

class GGUFFile {
public:
    uint8_t* data = nullptr;
    size_t size = 0;
    int fd = -1;
    size_t data_offset = 0;
    std::unordered_map<std::string, GGUFTensor> tensors;

    void load(const std::string& path);
    ~GGUFFile();
};
