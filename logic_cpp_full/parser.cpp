#include "problem.h"
#include <sstream>
#include <string>

Problem parse_constraints(const std::string& text) {
    Problem p;
    std::istringstream iss(text);
    std::string line;
    while (std::getline(iss, line)) {
        if (line.find("ITEMS:") == 0) {
            for (char c : line)
                if (c >= 'A' && c <= 'Z')
                    p.items.push_back(c);
        }
        else if (line.find("LEFT_OF") == 0)
            p.constraints.push_back({LEFT_OF, line[8], line[10]});
        else if (line.find("ADJACENT") == 0)
            p.constraints.push_back({ADJACENT, line[9], line[11]});
        else if (line.find("NOT_EDGE") == 0)
            p.constraints.push_back({NOT_EDGE, line[9], 0});
    }
    return p;
}
