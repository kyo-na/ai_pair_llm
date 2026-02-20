# 開発過程・学習の記録

> **English version follows below / 英語版は下部に記載**

---

## 本・AI・実装を往復しながらの理解

本ディレクトリの実装に至るまで、Transformer / Attention / 量子化 / ハードウェア実装について、書籍・論文・既存実装・AI ツールを行き来しながら理解を積み重ねるという、非常に試行錯誤の多いプロセスを経ました。

特に以下の点は、**「読む → 分かったつもり → 実装 → 壊れる → 調べ直す」** を何度も繰り返しています。

- Attention が数式上どうなっているか
- それを逐次・並列・パイプラインでどう分解できるか
- GPU 的実装と FPGA 的実装の発想の違い
- 量子化（Ternary / BitNet）が演算器レベルで何を意味するか

---

## AI を「答え」ではなく「思考補助」として使う

利用した AI ツールは、完成コードを吐かせるためのものとしてではなく、以下の用途で徹底的に活用しました。

- 数式や処理フローの言語化
- 「この構造はハードウェア的に成立するか？」という壁打ち
- 実装が破綻した理由の切り分け
- FPGA 観点での設計妥当性の確認

AI の出す回答をそのまま信じることはほぼなく、必ず本・既存実装・波形・コードと突き合わせながら「自分で納得できる形」に落とし込んでいます。

---

## 理解に最も苦労した点

特に理解と設計に時間を要したのは以下です。

- Attention の時間方向依存性（Causal）
- KV Cache をメモリ構造としてどう扱うか
- Softmax を近似・ROM・パイプラインでどう成立させるか
- GPU 的な「巨大並列」と FPGA 的な「細粒度パイプライン」の違い

これらは AI に聞けば一瞬で説明は返ってくるものの、**「実装として成立させる」** ためには何度も手を動かして壊し、考え直す必要がありました。

---

## このディレクトリの位置づけ

このコードは「AI に聞いた結果を並べたもの」ではなく、

> **AI を使い切るくらい調べ、考え、理解するための途中結果**

として位置づけています。そのため、

- 完全最適解ではない
- 冗長な構成も含まれる
- 実験的な設計も多い

ですが、Transformer をハードウェア視点で理解しようとした痕跡はすべてここに残しています。

---

## ⚠️ モデルウェイトについて

本リポジトリにはモデルウェイトは含まれていません。GGUF 等のモデルファイルは、各ライセンスに従い Hugging Face 等から別途取得の上、`models/` ディレクトリ（`.gitignore` 対象）に配置してください。

---
---

# Development Notes

> This repository contains source code only for a custom inference runtime. **No model weights are included.**

---

## Learning by Iteration: Books, AI, and Implementation

This implementation was developed through extensive iteration between books, papers, existing implementations, and AI tools.

Understanding was not achieved by reading alone. The process repeatedly involved:

**read → think I understand → implement → fail → re-study**

especially for the following areas:

- Mathematical structure of Attention
- Pipeline decomposition for hardware implementation
- Differences between GPU-oriented and FPGA-oriented design approaches
- Meaning of quantization (Ternary / BitNet) at the arithmetic unit level

---

## Using AI as a Thinking Partner, Not an Answer Generator

AI tools were not used to generate final code, but rather as thinking and discussion partners for:

- Verbalizing mathematical formulas and processing flows
- Architectural reasoning (e.g., "Is this structure feasible in hardware?")
- Identifying the root causes of implementation failures
- Validating design decisions from an FPGA-oriented perspective

AI-generated responses were never accepted at face value. All outputs were cross-checked against books, existing implementations, waveforms, and actual code behavior until the design could be fully understood and justified.

---

## What Was Most Difficult

The most challenging aspects of this work were:

- Temporal dependency in causal attention
- Structuring the KV cache as a hardware-friendly memory system
- Implementing softmax using approximation, ROM-based lookup, or pipelining
- Reconciling GPU-style massive parallelism with FPGA-style fine-grained pipelines

While AI tools can explain these concepts almost instantly, **making them work as real, functioning hardware logic required deep iteration and repeated redesign.**

---

## Positioning of This Directory

This directory is not a collection of AI-generated answers. Instead, it represents:

> **The result of pushing AI-assisted learning to its limits in order to truly understand the system.**

The code is not necessarily optimal or production-ready. However, it preserves the full trace of a hardware-first reasoning process for understanding Transformer inference at a structural level.

---

## ⚠️ Model Weights

Model weights are **NOT** included in this repository. Users must place GGUF or other model files under the `models/` directory (excluded by `.gitignore`) after obtaining them separately from Hugging Face under the respective licenses.

Evaluation reports (CSV/JSON) contain only aggregated metrics and do not include model weights or raw model outputs.

---

## Disclaimer

This project is an independent personal research implementation. It is not affiliated with, endorsed by, or officially associated with OpenAI, Google, or any other organization.
