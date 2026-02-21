## GPU / CUDA について（重要）

本リポジトリの実装は、**CPU 上で完結する Transformer Attention の参照実装**です。
すべての forward / backward は CPU のみで動作し、GPU は必須ではありません。

本リポジトリは CPU 実装を主とした参照実装です。GPU / CUDA による高速化は任意であり、詳細は `cuda/` 以下に記載しています。

CUDA / GPU による高速化については、
- 実験的
- 環境依存が強い
- ビルド・実行条件がユーザーごとに異なる

という理由から、本リポジトリでは **GPU 実行を必須とはしていません**。

GPU を用いた実行・検証・最適化を行う場合は、
**ご自身の GPU 環境にて、自己責任で実施してください。**

※ GPU 実装は、本 CPU 実装と同一の数式・メモリレイアウトを前提とした
オプションの最適化として位置付けています。


## GPU / CUDA Notes (Important)

This repository provides a **reference CPU implementation** of Transformer
self-attention with complete forward and backward propagation.

This repository provides a CPU-first reference implementation. GPU / CUDA acceleration is optional and documented separately under `cuda/`.

All core functionalities work entirely on CPU.
**GPU / CUDA is NOT required.**

CUDA-based acceleration is intentionally treated as:
- experimental
- environment-dependent
- optional optimization

Due to differences in GPU hardware, drivers, and CUDA versions,
**users are expected to experiment with GPU execution on their own systems,
at their own responsibility.**

GPU implementations (if any) are designed as optional optimizations
based on the same mathematical formulation and memory layout
as the CPU reference implementation.
