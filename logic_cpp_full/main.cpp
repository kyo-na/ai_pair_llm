#include "problem.h"
#include <iostream>

Problem parse_constraints(const std::string&);
std::vector<char> solve(const Problem&);

int main() {
    std::string input =
        "ITEMS: A B C D E\n"
        "LEFT_OF A B\n"
        "ADJACENT C D\n"
        "NOT_EDGE E\n";

    Problem p = parse_constraints(input);
    auto res = solve(p);

    if (res.empty()) {
        std::cout << "No solution\n";
    } else {
        std::cout << "Solution: ";
        for (char c : res) std::cout << c << " ";
        std::cout << "\n";
    }
}
