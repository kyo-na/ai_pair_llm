#pragma once

/*
========================================================
 GPU 実行エンジン（将来用・未実装）
========================================================

このファイルは **現時点では一切使用しない**。

目的：
- CPU/GPU を同一インターフェースで扱うための設計メモ
- runtime から見たときの「GPUとは何か」を固定する

想定責務：
- CUDA Stream 管理
- GPUメモリ常駐（重み / KV cache）
- 非同期 kernel launch
- score() による GPU 使用率の報告

実装しない理由（今）：
- CPUランタイムの設計を先に固めるため
- GPUを入れるとデバッグ軸が増えるため

将来やること：
class CUDAEngine : public EngineBase {
    OccupancyScore score() const override;
    void submit(const MiniLLMTask& task) override;
    bool try_fetch(MiniLLMResult& result) override;
};

========================================================
*/