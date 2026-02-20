#include "problem.h"
#include <cmath>

bool verify(const std::vector<char>& order,
            const std::vector<Constraint>& cons) {
    int pos[256];
    for (int i = 0; i < order.size(); i++)
        pos[(int)order[i]] = i;

    for (auto& c : cons) {
        if (c.type == LEFT_OF && !(pos[c.a] < pos[c.b])) return false;
        if (c.type == ADJACENT && std::abs(pos[c.a] - pos[c.b]) != 1) return false;
        if (c.type == NOT_EDGE) {
            int p = pos[c.a];
            if (p == 0 || p == order.size() - 1) return false;
        }
    }
    return true;
}
