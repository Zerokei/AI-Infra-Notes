---
aliases: [索引, 目录]
created: 2026-05-08
updated: 2026-06-04
---

# index

AI-Infra-Notes 全部 wiki 页面的目录，按主题分组。每条一句话点出本页内容，深入请点链接。

## 基础架构

底层 Transformer 系列——所有 LLM / 视觉 / 多模态讨论的共同前提。

- [[Transformer]] —— LLM 与多模态的共同骨架；attention 机制、encoder-decoder vs decoder-only、推理视角下的算力 / 显存特征。
- [[ViT]] —— 把 Transformer 直接搬到图像（patch 当 token），现代多模态视觉编码器的默认选择。
- [[DiT]] —— 文生图 / 文生视频的去噪主干，把扩散模型里的 U-Net 换成 Transformer；FLUX / SD3 / Sora 的共同基底。

## LLM 推理

围绕自回归生成的算力 / 显存 / 网络瓶颈展开的优化体系。

- [[LLM Inference Optimization]] —— 推理优化的总览页：Decode memory bound、Prefill flops bound，三类优化手段（缓存、减次数、调度）的导航。
- [[nanoGPT Inference]] —— 用 nanoGPT 的最小 GPT 实现拆解自回归推理：forward、矩阵 shape、采样循环，以及无 KV Cache 的教学实现边界。
- [[GPT-3 Inference]] —— 以 GPT-3-style decoder-only Transformer 为例，拆解 prefill / decode 与 KV Cache 如何接入自回归推理。
- [[KV Cache]] —— 自回归生成的核心优化：缓存历史 token 的 K、V，把总复杂度从 $O(n^3)$ 降到 $O(n^2)$。
- [[PagedAttention]] —— vLLM 的 KV Cache 显存管理机制，借鉴 OS 虚拟内存的分页思路，把浪费率从 60–80% 降至 4% 以下。

## 推理引擎

生产级 LLM 推理服务系统。

- [[vLLM]] —— UC Berkeley 开源、社区事实标准；PagedAttention + continuous batching + prefix caching + tensor parallelism 一整套。
- [[SGLang]] —— LMSYS 开源；后端 RadixAttention 做跨请求细粒度前缀复用，前端附带描述多步 LLM 程序的 DSL。

## 多模态

理解 + 生成的多模态家族——VLM / T2I / UFM 各自的架构与 infra 差异。

- [[Multimodal Models]] —— 多模态的导航页：VLM / ALM / Video-LLM / VLA / Omni 的子方向地图与代表模型。
- [[VLM Architecture]] —— LLaVA 之后开源 VLM 的事实标准三段式：vision encoder + connector + LLM。
- [[UFM]] —— 同时具备「理解 + 生成」的统一多模态模型；按理解与生成怎么连分三种范式（外部 API / 模块化联合 / 端到端原生）。
- [[Multimodal Inference Considerations]] —— 多模态推理 infra 与纯 LLM 的差异：VLM 是放大版 LLM，T2I 完全不同的 profile，UFM 混合负载。
- [[Qwen-Image-Edit-2511]] —— Qwen 系图像编辑模型 2025-11 版；架构同上代 2509，改进全在权重；本页重点是 diffusers 单卡参考流水线 + vllm-omni 三层并行（TP / CFG / SP）服务化部署。
