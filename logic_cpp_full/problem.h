#pragma once
#include <vector>

enum ConstraintType { LEFT_OF, ADJACENT, NOT_EDGE };

struct Constraint {
    ConstraintType type;
    char a;
    char b;
};

struct Problem {
    std::vector<char> items;
    std::vector<Constraint> constraints;
};
