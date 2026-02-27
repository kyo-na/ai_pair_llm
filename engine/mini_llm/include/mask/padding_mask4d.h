#pragma once
#include "tensor4d.h"
#include <vector>

class PaddingMask4D
{
public:
    // pad_id はトークンID
    static void apply(
        Tensor4D& scores,
        const std::vector<std::vector<int>>& input_ids,
        int pad_id);
};