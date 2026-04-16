# Blackwell Kernel Wiki — 调研与建设方案草案

## 一、项目目标

在本地仓库中建立一个完整的 Blackwell (SM100) 和 Hopper (SM90) GPU Kernel 优化知识库，用于指导 LLM Agent 编写 B200 机器上的高性能 kernel。

知识库需要：
- 保存大量原始信息（PR、比赛数据、博客、文档）
- 通过多层文档作为入口，方便 LLM 查询
- 支持按问题、技巧、硬件特性、kernel 类型、语言等多维度交叉索引

## 二、调研范围

### 代码仓库 PR（2025-01 至今）
- **NVIDIA/cutlass** — CUTLASS 4.x Blackwell 支持，CuTe SM100 atoms
- **sgl-project/sglang** — SGLang Blackwell kernel 集成
- **vllm-project/vllm** — vLLM Blackwell 支持
- **flashinfer-ai/flashinfer** — FlashInfer Blackwell attention/MoE kernel
- **pytorch/pytorch** — PyTorch/TorchInductor Blackwell 后端

### 比赛
- **GPU Mode NVFP4 Blackwell Hackathon**（NVIDIA + GPU Mode，2025 Nov - 2026 Feb）
  - Problem 1: NVFP4 Batched GEMV
  - Problem 2: NVFP4 GEMM
  - Problem 3: Gated Dual GEMM
  - Problem 4: Grouped GEMM
- **FlashInfer AI Kernel Generation Contest**（MLSys 2026）
  - Track A: Fused MoE (FP8)
  - Track B: DeepSeek V3.2 Sparse Attention
  - Track C: Gated Delta Net (Qwen3-Next)

### 团队优化工作
- **DeepSeek** — MoE kernel、sparse attention、FP8 训练/推理
- **Qwen** — Gated Delta Net、Blackwell 适配

### 搜索关键词
blackwell, sm100, sm_100, sm_100a, tcgen05, tmem, cuda 13, B200, B100,
nvfp4, fp4, fp8, cutile, umma, clc, 2sm, warp specialization, persistent kernel,
fp8_moe, moe, deepseek, sparse attention, gated delta net, qwen,
grouped gemm, tma, block scale, microscaling, triton, tilelang, cute

## 三、知识库架构（参考 Karpathy LLM Wiki）

采用三层架构：原始数据层 → LLM 维护的知识层 → 交叉索引查询层。

```
blackwell-kernel-wiki/
├── CLAUDE.md                  # Schema：LLM 操作指南、约定、工作流
├── index.md                   # 主入口：按类别组织的所有页面索引
├── log.md                     # 时间线：所有 ingest 操作记录
│
├── sources/                   # 第一层：原始信息（不可修改）
│   ├── prs/                   # PR 原始数据
│   │   ├── cutlass/           # 每个 PR 一个文件
│   │   ├── sglang/
│   │   ├── vllm/
│   │   ├── flashinfer/
│   │   └── pytorch/
│   ├── contests/              # 比赛信息
│   │   ├── gpu-mode-nvfp4/
│   │   └── flashinfer-mlsys26/
│   ├── docs/                  # 官方文档摘要
│   └── blogs/                 # 社区博客/教程
│
├── wiki/                      # 第二层：LLM 维护的知识页面
│   ├── hardware/              # 硬件特性（TMEM, tcgen05, CLC, TMA...）
│   ├── techniques/            # 优化技巧（warp specialization, pipelining...）
│   ├── patterns/              # 问题→解决方案映射（SM利用率低→...）
│   ├── kernels/               # 具体 kernel 案例分析
│   └── languages/             # 语言/DSL 指南（CuTe, Triton, Tilelang...）
│
└── queries/                   # 第三层：交叉索引入口
    ├── by-problem.md          # 按问题类型查询
    ├── by-technique.md        # 按优化技巧查询
    ├── by-hardware-feature.md # 按硬件特性查询
    ├── by-repo.md             # 按来源仓库查询
    ├── by-kernel-type.md      # 按 kernel 类型查询
    └── by-language.md         # 按编程语言查询
```

### 导航流程
1. `index.md` → 按类别找到页面
2. `queries/by-problem.md` → 有具体性能问题时定位方案
3. `queries/by-technique.md` → 了解某个技巧的所有示例
4. 深入 `wiki/` 页面 → 详细解释和代码示例
5. 跟随 `[source]` 链接 → `sources/` 原始数据和代码

### Source 页面格式
```markdown
---
repo: NVIDIA/cutlass
pr: 1234
title: "PR title"
author: username
date: 2025-06-15
url: https://github.com/NVIDIA/cutlass/pull/1234
tags: [sm100, tcgen05, gemm, fp8, warp-specialization]
techniques: [warp-specialization, tmem-double-buffering]
hardware_features: [tcgen05, tmem, clc]
language: cute-dsl
---
## Summary
...
## Problem
...
## Solution / Techniques
...
## Key Code
...
## Performance
...
```

### Wiki 页面格式
```markdown
---
title: "Page Title"
tags: [tag1, tag2]
related: [link1.md, link2.md]
sources: [PR-123.md, PR-456.md]
---
## Overview
...
## How It Works
...
## When To Use
...
## Examples
...
## Related
- [Link](path) — description
```

### 标签体系
- **硬件特性**: sm100, sm90, tcgen05, tmem, tma, clc, 2sm-cooperative, pdl, gdc, nvfp4, fp8, block-scale, wgmma, cluster
- **优化技巧**: warp-specialization, persistent-kernel, swizzling, pipeline-stages, double-buffering, register-reuse, shared-memory-optimization, tma-multicast, epilogue-fusion, tile-scheduling, communication-overlap
- **Kernel 类型**: gemm, attention, moe, sparse-attention, gemv, grouped-gemm, gated-delta-net, fused-kernel, decode, prefill, quantization
- **编程语言**: cuda-cpp, cute-dsl, triton, tilelang, cutile, ptx, python

## 四、已收集的关键架构信息

### Blackwell SM100 核心变化（vs Hopper SM90）

| 方面 | Hopper SM90 | Blackwell SM100 |
|---|---|---|
| MMA 指令 | wgmma.mma_async（warp group 128 线程） | tcgen05.mma（单线程发射） |
| 累加器 | 寄存器 | TMEM（专用 256KB） |
| 最大 MMA 形状 | m64×n256×k16 | m128×n256×k16 (1SM), m256×n256×k16 (2SM) |
| 吞吐量 | 基线 | BF16 2×, FP4 4× |
| Tile 调度 | 静态/手动 | CLC 硬件动态调度 |
| Shared Memory | 228 KB | 228 KB |
| L2 Cache | 50 MB (H100) | 126 MB (B200) |
| TMEM | 无 | 256 KB/SM (128 rows × 512 cols) |

### 关键新特性
1. **tcgen05.mma** — 7 种变体，支持 TF32/FP16/INT8/FP8/FP6/FP4/NVFP4
2. **Tensor Memory (TMEM)** — 专用累加器内存，消除寄存器压力
3. **2-SM Cooperative** — 两个 SM 协作执行 256×256 MMA
4. **CLC** — 硬件级动态 tile 调度
5. **NVFP4** — 原生 4-bit 浮点，block scale
6. **PDL 默认开启** — kernel 间依赖执行重叠

### 性能优化路径（tcgen05 tutorial 数据）
```
Naive (17%) → Swizzling (46%) → Pipelining (62%) → Warp Specialization (80%)
→ 2-SM MMA (86%) → Persistent/CLC (98% of cuBLAS)
```

### 比赛信息摘要

**GPU Mode NVFP4 Hackathon**:
- 4 个问题，奖品 RTX 5080/5090，Grand prize GB300
- Problem 1 (NVFP4 Batched GEMV) 参赛者从 2000μs 优化到 22.3μs
- 关键技巧：memory coalescing, FP4/FP8 decode intrinsics, PTX assembly, ILP

**FlashInfer MLSys 2026 Contest**:
- 3 个 track 均在 B200 上评测
- 支持 CuTe DSL, CUDA, Tilelang, Triton, cuTile
- 允许人写、AI 生成、或混合提交
- 2026-04-24 截止提交

## 五、调研执行策略

### 并行方式
7 个独立研究任务可完全并行：
1. CUTLASS PRs → `sources/prs/cutlass/`
2. SGLang PRs → `sources/prs/sglang/`
3. vLLM PRs → `sources/prs/vllm/`
4. FlashInfer PRs → `sources/prs/flashinfer/`
5. PyTorch PRs → `sources/prs/pytorch/`
6. 比赛数据 → `sources/contests/`
7. DeepSeek/Qwen 优化 + 社区博客 → `sources/blogs/`

### 每个 PR 的提取模板
- PR number, title, author, date, URL
- 解决什么问题
- 使用了什么技巧
- 利用了哪些硬件特性
- 编程语言
- 关键代码片段
- 性能提升数据

### 后处理
调研完成后：
1. 从 sources 合成 wiki 页面
2. 建立交叉索引
3. 验证链接完整性
4. 确保每个技巧/问题都有足够的案例支撑

## 六、待讨论的设计决策

1. **PR 粒度** — 每个 PR 一个文件 vs. 按主题合并？重要 PR 保存多少代码？
2. **Hopper 内容范围** — 全面收录 vs. 只保留对 Blackwell 有启发的？
3. **语言** — 知识库正文用英文还是中文？
4. **索引粒度** — 一个 kernel 解决多个问题时，在所有相关索引中都出现？
5. **更新机制** — ingest 工作流的具体步骤？
6. **是否需要额外的可视化**（如架构图、性能对比图）？

## 七、参考资料

- [Karpathy LLM Wiki](https://gist.github.com/karpathy/442a6bf555914893e9891c11519de94f)
- [NVIDIA Blackwell Tuning Guide](https://docs.nvidia.com/cuda/blackwell-tuning-guide/)
- [tcgen05 for dummies (Gau Nernst)](https://gau-nernst.github.io/tcgen05/)
- [Colfax CUTLASS Blackwell Tutorial](https://research.colfax-intl.com/cutlass-tutorial-writing-gemm-kernels-using-tensor-memory-for-nvidia-blackwell-gpus/)
- [CUDA 13.0 Blog](https://developer.nvidia.com/blog/whats-new-and-important-in-cuda-toolkit-13-0/)
- [CUDA 13.1 Blog](https://developer.nvidia.com/blog/nvidia-cuda-13-1-powers-next-gen-gpu-programming-with-nvidia-cuda-tile-and-performance-gains/)
- [GPU Mode NVFP4 Hackathon](https://forums.developer.nvidia.com/t/join-us-for-the-blackwell-nvfp4-kernel-hackathon-with-nvidia-and-gpu-mode/350092)
- [FlashInfer MLSys 2026 Contest](https://mlsys26.flashinfer.ai/)
- [Blackwell NVFP4 Hackathon Journey (Yue Zhang)](https://yue-zhang-2025.github.io/2025/12/02/blackwell-nvfp4-kernel-hackathon-journey.html)
- [Modular: Matrix Multiplication on Blackwell](https://www.modular.com/blog/matrix-multiplication-on-nvidias-blackwell-part-1-introduction)
