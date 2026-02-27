#include "mask/padding_mask4d.h"

void PaddingMask4D::apply(
    Tensor4D& scores,
    const std::vector<std::vector<int>>& input_ids,
    int pad_id)
{
    int B = scores.B;
    int H = scores.H;
    int T = scores.T;
    int Tk = scores.D;

    for (int b = 0; b < B; ++b)
    {
        for (int k = 0; k < Tk; ++k)
        {
            if (input_ids[b][k] == pad_id)
            {
                for (int h = 0; h < H; ++h)
                for (int t = 0; t < T; ++t)
                {
                    scores.at(b,h,t,k) += -1e9f;
                }
            }
        }
    }
}