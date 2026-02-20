
#include <vector>
#include <cmath>
using namespace std;

void attention_cpu(
    const vector<float>& Q,
    const vector<float>& K,
    const vector<float>& V,
    vector<float>& O,
    int T, int D
){
    for(int t=0;t<T;t++){
        float sum=0;
        for(int j=0;j<T;j++){
            float s=0;
            for(int d=0;d<D;d++)
                s+=Q[t*D+d]*K[j*D+d];
            sum+=exp(s);
        }
        for(int d=0;d<D;d++){
            O[t*D+d]=0;
            for(int j=0;j<T;j++)
                O[t*D+d]+=V[j*D+d]/sum;
        }
    }
}
