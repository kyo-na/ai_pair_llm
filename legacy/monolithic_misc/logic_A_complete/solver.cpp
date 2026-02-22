
#include <iostream>
#include <vector>
#include <string>
#include <algorithm>
#include <cmath>

using namespace std;

struct Constraint {
    string type;
    string a;
    string b;
};

bool check_left_of(const vector<string>& order, string a, string b) {
    int pa=-1,pb=-1;
    for(int i=0;i<order.size();++i){
        if(order[i]==a) pa=i;
        if(order[i]==b) pb=i;
    }
    return pa < pb;
}

bool check_adjacent(const vector<string>& order, string a, string b) {
    int pa=-1,pb=-1;
    for(int i=0;i<order.size();++i){
        if(order[i]==a) pa=i;
        if(order[i]==b) pb=i;
    }
    return abs(pa-pb)==1;
}

bool check_not_edge(const vector<string>& order, string a) {
    int pa=-1;
    for(int i=0;i<order.size();++i)
        if(order[i]==a) pa=i;
    return pa!=0 && pa!=order.size()-1;
}

bool verify(const vector<string>& order, const vector<Constraint>& cons){
    for(auto& c: cons){
        if(c.type=="left_of" && !check_left_of(order,c.a,c.b)) return false;
        if(c.type=="adjacent" && !check_adjacent(order,c.a,c.b)) return false;
        if(c.type=="not_edge" && !check_not_edge(order,c.a)) return false;
    }
    return true;
}

vector<string> solve(vector<string> items, vector<Constraint> cons){
    sort(items.begin(), items.end());
    do {
        if(verify(items,cons)) return items;
    } while(next_permutation(items.begin(), items.end()));
    return {};
}

int main(){
    vector<string> items = {"A","B","C","D","E"};
    vector<Constraint> cons = {
        {"left_of","A","B"},
        {"adjacent","C","D"},
        {"not_edge","E",""}
    };

    auto result = solve(items,cons);
    if(result.empty()){
        cout<<"No solution\n";
    } else {
        cout<<"Solution: ";
        for(auto&s:result) cout<<s<<" ";
        cout<<endl;
    }
}
