#include "problem.h"
#include <algorithm>

extern bool verify(const std::vector<char>&,
                   const std::vector<Constraint>&);

std::vector<char> solve(const Problem& p) {
    std::vector<char> v = p.items;
    std::sort(v.begin(), v.end());
    do {
        if (verify(v, p.constraints)) return v;
    } while (std::next_permutation(v.begin(), v.end()));
    return {};
}
