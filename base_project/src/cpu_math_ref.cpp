#include "cpu_math_ref.h"
#include <cmath>
#include <vector>
#include <iostream>

// ============================================================================
// 1. RMSNorm (二乗平均平方根正規化)
// AIのテンション（数値の爆発）を抑えるための数学操作
// ============================================================================
void rmsnorm_cpu(float* out, const float* x, const float* weight, int size) {
    float ss = 0.0f;
    // 1. すべての要素の2乗和を計算
    for (int j = 0; j < size; j++) {
        ss += x[j] * x[j];
    }
    ss /= size;        // 平均をとる
    ss += 1e-5f;       // ゼロ除算回避 (epsilon)
    ss = 1.0f / sqrtf(ss); // 平方根の逆数

    // 2. 正規化して学習済み重み(weight)を掛ける
    for (int j = 0; j < size; j++) {
        out[j] = weight[j] * (ss * x[j]);
    }
}

// ============================================================================
// 2. MatMul (行列ベクトル積: GEMV)
// 最も計算に時間がかかる部分。入力ベクトル x に重み行列 W を掛ける
// ============================================================================
void matmul_cpu(float* out, const float* x, const float* w, int rows, int cols) {
    // W のサイズは [rows, cols], x のサイズは [cols]
    for (int i = 0; i < rows; i++) {
        float val = 0.0f;
        for (int j = 0; j < cols; j++) {
            val += w[i * cols + j] * x[j];
        }
        out[i] = val;
    }
}

// ============================================================================
// 3. RoPE (Rotary Positional Embedding)
// 単語の位置（順番）を「角度（サイン・コサイン）」に変換して埋め込む処理
// ============================================================================
void rope_cpu(float* q, float* k, int dim, int pos) {
    // 2次元ずつペアにして回転させる
    for (int i = 0; i < dim; i += 2) {
        // 周波数を計算 (10000^(2i/d))
        float freq = 1.0f / powf(10000.0f, (float)i / (float)dim);
        float val = (float)pos * freq;
        float fcr = cosf(val);
        float fci = sinf(val);
        
        // Q（Query）を回転
        float q0 = q[i];
        float q1 = q[i+1];
        q[i]   = q0 * fcr - q1 * fci;
        q[i+1] = q0 * fci + q1 * fcr;
        
        // K（Key）も回転
        float k0 = k[i];
        float k1 = k[i+1];
        k[i]   = k0 * fcr - k1 * fci;
        k[i+1] = k0 * fci + k1 * fcr;
    }
}

// ============================================================================
// 4. Causal Attention (注意機構)
// 「どの過去の単語に注目すべきか」を計算する、LLMの魂。
// ============================================================================
void attention_cpu(float* out, float* q, float* k_cache, float* v_cache, 
                   int seq_len, int n_heads, int head_size) {
    
    // 各ヘッド（マルチヘッドアテンション）ごとに独立して計算
    for (int h = 0; h < n_heads; h++) {
        float* q_head = q + h * head_size;
        std::vector<float> att(seq_len); // スコア保存用
        
        // --- Step A: Q * K^T ---
        // 現在の単語(Q)と、過去のすべての単語(K)の類似度(内積)を計算
        for (int t = 0; t < seq_len; t++) {
            float* k_head = k_cache + t * (n_heads * head_size) + h * head_size;
            float score = 0.0f;
            for (int i = 0; i < head_size; i++) {
                score += q_head[i] * k_head[i];
            }
            att[t] = score / sqrtf((float)head_size); // スケール調整
        }
        
        // --- Step B: Softmax ---
        // スコアを確率（合計が1.0になる数値）に変換
        float max_val = att[0];
        for (int t = 1; t < seq_len; t++) {
            if (att[t] > max_val) max_val = att[t]; // オーバーフロー対策
        }
        
        float sum = 0.0f;
        for (int t = 0; t < seq_len; t++) {
            att[t] = expf(att[t] - max_val);
            sum += att[t];
        }
        for (int t = 0; t < seq_len; t++) {
            att[t] /= sum;
        }
        
        // --- Step C: Score * V ---
        // 計算した確率(att)を、過去の記憶(V)に掛けて合算する
        for (int i = 0; i < head_size; i++) {
            out[h * head_size + i] = 0.0f;
        }
        for (int t = 0; t < seq_len; t++) {
            float* v_head = v_cache + t * (n_heads * head_size) + h * head_size;
            for (int i = 0; i < head_size; i++) {
                out[h * head_size + i] += att[t] * v_head[i];
            }
        }
    }
}

// ============================================================================
// 5. SwiGLU FFN (フィードフォワード・ネットワーク)
// 単語の特徴を変換する非線形活性化関数
// ============================================================================
void ffn_swiglu_cpu(float* out, float* x, float* w_up, float* w_gate, float* w_down, int D, int hidden_dim) {
    std::vector<float> up(hidden_dim);
    std::vector<float> gate(hidden_dim);
    
    // 1. アッププロジェクション & ゲートプロジェクション
    matmul_cpu(up.data(), x, w_up, hidden_dim, D);
    matmul_cpu(gate.data(), x, w_gate, hidden_dim, D);
    
    // 2. SiLU 活性化関数: x * sigmoid(x)
    for (int i = 0; i < hidden_dim; i++) {
        float val = gate[i];
        float silu = val / (1.0f + expf(-val));
        gate[i] = silu * up[i]; // 要素ごとの掛け算
    }
    
    // 3. ダウンプロジェクション
    matmul_cpu(out, gate.data(), w_down, D, hidden_dim);
}
