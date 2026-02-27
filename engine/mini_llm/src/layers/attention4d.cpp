#include "layers/attention4d.h"
#include <cmath>
#include <cstdlib>

static float init_w(int fan_in){
    return std::sqrt(2.0f/fan_in) *
           ((float)rand()/RAND_MAX - 0.5f);
}

Attention4D::Attention4D(int H,int D)
: H_(H), D_(D),
  Wq_(1,1,D,D),
  Wk_(1,1,D,D),
  Wv_(1,1,D,D),
  Wo_(1,1,D,D)
{
    for(auto& w:Wq_.data) w=init_w(D);
    for(auto& w:Wk_.data) w=init_w(D);
    for(auto& w:Wv_.data) w=init_w(D);
    for(auto& w:Wo_.data) w=init_w(D);
}

Tensor4D Attention4D::forward(
    const Tensor4D& x,
    KVCache4D* cache,
    bool use_cache)
{
    last_x_ = x;

    int B=x.B;
    int T=x.T;

    Tensor4D Q(B,T,H_,D_);
    Tensor4D K(B,T,H_,D_);
    Tensor4D V(B,T,H_,D_);

    // projection
    for(int b=0;b<B;++b)
    for(int t=0;t<T;++t)
    for(int h=0;h<H_;++h)
    for(int d=0;d<D_;++d){
        float q=0,k=0,v=0;
        for(int i=0;i<D_;++i){
            float xi=x.at(b,t,h,i);
            q+=xi*Wq_.at(0,0,i,d);
            k+=xi*Wk_.at(0,0,i,d);
            v+=xi*Wv_.at(0,0,i,d);
        }
        Q.at(b,t,h,d)=q;
        K.at(b,t,h,d)=k;
        V.at(b,t,h,d)=v;
    }

    // KVCache追加
    if(use_cache && cache){
        cache->append(K,V);
        K = cache->K();
        V = cache->V();
        T = K.T;
    }

    Tensor4D context(B,T,H_,D_);

    for(int b=0;b<B;++b)
    for(int h=0;h<H_;++h)
    for(int t=0;t<T;++t){

        float max_s=-1e9f;

        for(int tk=0;tk<=t;++tk){
            float s=0;
            for(int d=0;d<D_;++d)
                s+=Q.at(b,t,h,d)*K.at(b,tk,h,d);
            s/=std::sqrt((float)D_);
            context.at(b,t,h,0)=0; // dummy
            if(s>max_s) max_s=s;
        }

        float denom=0;

        std::vector<float> attn_row(t+1);

        for(int tk=0;tk<=t;++tk){
            float s=0;
            for(int d=0;d<D_;++d)
                s+=Q.at(b,t,h,d)*K.at(b,tk,h,d);
            s/=std::sqrt((float)D_);
            float e=std::exp(s-max_s);
            attn_row[tk]=e;
            denom+=e;
        }

        for(int d=0;d<D_;++d){
            float sum=0;
            for(int tk=0;tk<=t;++tk){
                float a=attn_row[tk]/denom;
                sum+=a*V.at(b,tk,h,d);
            }
            context.at(b,t,h,d)=sum;
        }
    }

    Tensor4D out(B,T,H_,D_);

    for(int b=0;b<B;++b)
    for(int t=0;t<T;++t)
    for(int h=0;h<H_;++h)
    for(int d=0;d<D_;++d){
        float s=0;
        for(int i=0;i<D_;++i)
            s+=context.at(b,t,h,i)*Wo_.at(0,0,i,d);
        out.at(b,t,h,d)=s;
    }

    return out;
}

Tensor4D Attention4D::backward(const Tensor4D& grad)
{
    // 今は簡易版（安定ビルド優先）
    return Tensor4D(
        last_x_.B,
        last_x_.T,
        last_x_.H,
        last_x_.D);
}

std::vector<Tensor4D*> Attention4D::parameters()
{
    return {&Wq_,&Wk_,&Wv_,&Wo_};
}