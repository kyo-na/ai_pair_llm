<<<<<<< HEAD

/*
 * training.cpp
 * Minimal scratch-built training engine for MoE-FSM / Transformer
 * No PyTorch, no external ML libs.
 */
=======
// training.cpp
// Minimal scratch-built training engine with loss mask + PPO
>>>>>>> d8e6bad (Initial clean project structure)

#include <iostream>
#include <vector>
#include <cmath>
<<<<<<< HEAD
#include <cstdlib>
#include <cstdint>

// --- Minimal Tensor ---
struct Tensor {
    std::vector<int> shape;
    std::vector<float> data;
    std::vector<float> grad;
    bool requires_grad;

    Tensor(std::vector<int> s, bool req=false)
        : shape(s), requires_grad(req) {
        int n = 1;
        for (int v : s) n *= v;
        data.assign(n, 0.f);
        grad.assign(n, 0.f);
=======
#include <cstdint>
#include <string>

// ---- include local modules ----
#include "loss_mask.cpp"
#include "ppo.cpp"

// --- Tensor ---
struct Tensor {
    std::vector<int> shape;
    std::vector<float> data, grad;
    bool requires_grad;
    Tensor(std::vector<int> s, bool req=false):shape(s),requires_grad(req){
        int n=1; for(int v:s) n*=v;
        data.assign(n,0.f);
        grad.assign(n,0.f);
>>>>>>> d8e6bad (Initial clean project structure)
    }
    int numel() const { return data.size(); }
};

<<<<<<< HEAD
// --- Simple SGD ---
struct SGD {
    float lr;
    std::vector<Tensor*> params;
    SGD(float lr_) : lr(lr_) {}
    void add(Tensor& t) { params.push_back(&t); }
    void step() {
        for (auto* p : params)
            for (int i = 0; i < p->numel(); ++i)
                p->data[i] -= lr * p->grad[i];
    }
    void zero_grad() {
        for (auto* p : params)
            std::fill(p->grad.begin(), p->grad.end(), 0.f);
    }
};

// --- Copy task batch ---
struct Batch {
    std::vector<int> x, y;
};

Batch make_copy_batch(int B, int T, int vocab, uint32_t& rng) {
    auto rnd = [&]() {
        rng = 1664525u * rng + 1013904223u;
        return int(rng % vocab);
    };
    Batch b;
    b.x.resize(B*T);
    b.y.resize(B*T);
    for (int i=0;i<B*T;++i) {
        int t = rnd();
        b.x[i]=t;
        b.y[i]=t;
    }
    return b;
}

// --- Dummy linear model (placeholder for MoE-FSM block) ---
struct Linear {
    int D, V;
    Tensor W;
    Linear(int d,int v):D(d),V(v),W({d,v},true){
        for(int i=0;i<W.numel();++i) W.data[i]=0.01f;
=======
// --- SGD ---
struct SGD {
    float lr;
    std::vector<Tensor*> params;
    SGD(float lr_):lr(lr_){}
    void add(Tensor& t){ params.push_back(&t); }
    void zero_grad(){
        for(auto*p:params)
            std::fill(p->grad.begin(),p->grad.end(),0.f);
    }
    void step(){
        for(auto*p:params)
            for(int i=0;i<p->numel();++i)
                p->data[i]-=lr*p->grad[i];
    }
};

// --- Dummy Linear (replace with MoE-FSM / Attention later) ---
struct Linear {
    int D,V;
    Tensor W;
    Linear(int d,int v):D(d),V(v),W({d,v},true){
        for(float&x:W.data) x=0.01f;
>>>>>>> d8e6bad (Initial clean project structure)
    }
    Tensor forward(Tensor& X){
        int B=X.shape[0];
        Tensor O({B,V},true);
<<<<<<< HEAD
        for(int b=0;b<B;++b)
            for(int v=0;v<V;++v){
                float s=0.f;
                for(int d=0;d<D;++d)
=======
        for(int b=0;b<B;b++)
            for(int v=0;v<V;v++){
                float s=0;
                for(int d=0;d<D;d++)
>>>>>>> d8e6bad (Initial clean project structure)
                    s+=X.data[b*D+d]*W.data[d*V+v];
                O.data[b*V+v]=s;
            }
        return O;
    }
};

<<<<<<< HEAD
float cross_entropy(Tensor& logits, const std::vector<int>& y) {
    int B=logits.shape[0], V=logits.shape[1];
    float loss=0.f;
    for(int b=0;b<B;++b){
        float m=-1e9f;
        for(int v=0;v<V;++v) m=std::max(m,logits.data[b*V+v]);
        float s=0.f;
        for(int v=0;v<V;++v) s+=std::exp(logits.data[b*V+v]-m);
        loss += -(logits.data[b*V+y[b]]-m-std::log(s));
    }
    return loss/B;
}

int main(){
    const int B=8, T=8, D=16, vocab=32;
    uint32_t rng=1;

    // fake embedding output
    Tensor X({B*T,D}, true);
    for(int i=0;i<X.numel();++i) X.data[i]=0.01f;

    Linear out(D,vocab);
    SGD opt(0.1f);
    opt.add(out.W);

    for(int epoch=0;epoch<200;++epoch){
        Batch batch = make_copy_batch(B,T,vocab,rng);
        Tensor logits = out.forward(X);
        float loss = cross_entropy(logits, batch.y);
        opt.zero_grad();
        // (grad omitted: placeholder to show structure)
        opt.step();
        if(epoch%20==0)
            std::cout<<"epoch "<<epoch<<" loss="<<loss<<"\n";
    }
    return 0;
=======
float cross_entropy(
    Tensor& logits,
    const std::vector<int>& y,
    const std::vector<int>& mask
){
    int B=logits.shape[0], V=logits.shape[1];
    float loss=0; int cnt=0;
    for(int b=0;b<B;b++){
        if(!mask[b]) continue;
        float m=-1e9;
        for(int v=0;v<V;v++) m=std::max(m,logits.data[b*V+v]);
        float s=0;
        for(int v=0;v<V;v++) s+=std::exp(logits.data[b*V+v]-m);
        loss+=-(logits.data[b*V+y[b]]-m-std::log(s));
        cnt++;
    }
    return cnt?loss/cnt:0.f;
}

int main(){
    const int B=8, D=16, V=32;
    Tensor X({B,D},true);
    for(float&x:X.data) x=0.01f;

    Linear out(D,V);
    SGD opt(0.1f);
    opt.add(out.W);

    PPOBuffer ppo;

    for(int epoch=0;epoch<200;epoch++){
        // fake decoded text
        std::string decoded =
            "<user>hi</user><assistant>hello</assistant>";

        auto mask = build_assistant_loss_mask(decoded);

        std::vector<int> target(B,1);
        Tensor logits = out.forward(X);

        opt.zero_grad();
        float ce = cross_entropy(logits,target,mask);

        // ---- PPO (dummy values for now) ----
        ppo.logp_old.push_back(-1.f);
        ppo.logp_new.push_back(-0.5f);
        ppo.reward.push_back(1.f);
        ppo.advantage.push_back(1.f);

        float rl = ppo_loss(ppo);

        opt.step();

        if(epoch%20==0)
            std::cout<<"epoch "<<epoch
                     <<" CE="<<ce
                     <<" PPO="<<rl<<"\n";
    }
>>>>>>> d8e6bad (Initial clean project structure)
}
