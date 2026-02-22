#pragma once
#include "spm_model.h"
#include <vector>
#include <string>
struct SPMTokenizer{const SPMModel& model;SPMTokenizer(const SPMModel& m):model(m){}std::vector<int> encode(const std::string&) const;std::string decode(const std::vector<int>&) const;};
