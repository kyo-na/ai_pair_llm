
#pragma once
#include <vector>
struct KVLayer{ std::vector<float> k,v; };
struct KVCache{ std::vector<KVLayer> layers; };
inline KVCache fork_kv(const KVCache& s){ return s; }
