
#include <vector>
#include <algorithm>

enum ConstraintType{LEFT_OF,ADJACENT,NOT_EDGE};

struct Constraint{
    ConstraintType type;
    char a,b;
};

extern bool verify(const std::vector<char>&,
                   const std::vector<Constraint>&);

std::vector<char> solve(std::vector<char> items,
                        const std::vector<Constraint>& cons){
    std::sort(items.begin(),items.end());
    do{
        if(verify(items,cons))
            return items;
    }while(std::next_permutation(items.begin(),items.end()));
    return {};
}
