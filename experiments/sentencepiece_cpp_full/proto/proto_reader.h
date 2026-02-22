#pragma once
#include <cstdint>
#include <string>
struct ProtoReader{const uint8_t* p;const uint8_t* end;ProtoReader(const uint8_t*,size_t);uint64_t read_varint();std::string read_string();void skip_field(uint32_t);};
