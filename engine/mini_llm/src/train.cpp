#include <iostream>
#include <vector>
#include <cmath>

#include "layers/embedding4d.h"
#include "layers/linear_vocab4d.h"
#include "layers/softmax4d.h"
#include "loss/cross_entropy4d.h"
#include "optimizer/adamw.h"
#include "train/optim/grad_clip.h"

int main()
{
    std::cout << "train start\n";

    int vocab = 256;
    int H = 1;
    int D = 64;

    Embedding4D embedding(vocab, H, D);
    LinearVocab4D proj(D, vocab);
    Softmax4D softmax;
    CrossEntropy4D loss_fn;
    AdamW optim(0.001f);

    std::vector<int> data =
    {
        1,2,3,4,5,6,7,8,
        2,3,4,5,6,7,8,9,
        3,4,5,6,7,8,9,10
    };

    int seq_len = 8;

    for(int epoch=0; epoch<50; ++epoch)
    {
        float total_loss=0;
        int steps=0;

        for(size_t i=0;i+seq_len+1<data.size();i+=seq_len)
        {
            std::vector<int> input(
                data.begin()+i,
                data.begin()+i+seq_len);

            std::vector<int> target(
                data.begin()+i+1,
                data.begin()+i+seq_len+1);

            Tensor4D x      = embedding.forward(input);
            Tensor4D logits = proj.forward(x);
            Tensor4D probs  = softmax.forward(logits);

            float loss = loss_fn.forward(probs, target);

            Tensor4D dprobs  = loss_fn.backward();
            Tensor4D dlogits = softmax.backward(dprobs);
            Tensor4D dx      = proj.backward(dlogits);

            embedding.backward(input, dx);

            std::vector<Tensor4D*> params;

            auto p1 = embedding.parameters();
            auto p2 = proj.parameters();

            params.insert(params.end(), p1.begin(), p1.end());
            params.insert(params.end(), p2.begin(), p2.end());

            train::optim::clip_grad_norm(params, 1.0f);
            optim.step(params);

            total_loss+=loss;
            steps++;
        }

        std::cout<<"epoch "<<epoch
                 <<" loss="<<total_loss/steps<<"\n";
    }

    std::cout<<"train done\n";
}