---
aliases: [多模态推理考量, MM Infra]
created: 2026-05-06
updated: 2026-05-06
---

# Multimodal Inference Considerations

多模态模型在推理 infra 上有几个与纯 LLM 显著不同的特征。本页按子方向（VLM / T2I / UFM）速览关键差异。

## VLM 推理：放大版的 LLM

VLM 仍是 transformer 标准的 prefill + decode 两阶段，但**视觉 token 显著放大 prefill 压力**：

> [!warning] VLM 与纯 LLM 的关键差异
> (1) **Prefill 重**：单图典型 500-4000 token（原生分辨率模型可达 16384），prefill 时间显著延长；
> (2) **KV cache 翻倍**：图像 token 全程驻留 KV cache，多图对话场景容量压力大；
> (3) **Batching 难**：动态分辨率模型（[[Kimi-VL]]、Qwen2.5-VL）让 batch padding 浪费严重；
> (4) **优化路径**：[[SGLang]] / [[vLLM]] 都为 VLM 做了专门的调度路径。

## T2I 推理：完全不同的 profile

T2I（Text-to-Image，文生图）的推理与 LLM 完全不同：

> [!warning] T2I 与 LLM 的 infra 差异
> (1) **不是 autoregressive，而是固定 N 步去噪**（典型 20-50 步）；
> (2) **无 KV cache**：每步是 full-image forward pass，无法增量计算；
> (3) **算力 profile 反过来**——LLM decode 是 memory-bound，T2I 是 compute-bound（每步都是 full image attention）；
> (4) **Batching 友好**：单请求延迟数秒-数十秒级，但多请求 batch 能线性提升 GPU 利用率；
> (5) **步数压缩**是活跃方向：LCM、SDXL Turbo、distillation、rectified flow 都把 50 步降到 1-4 步。

## UFM 推理：组合压力

UFM 根据范式不同（详见 [[UFM]]）有不同 infra 压力：

- **AR 生图**（[[GPT-4o]] image gen）：继承 LLM autoregressive 推理，但 1k+ 视觉 token 串行生成 → 单图延迟数十秒级
- **模块化联合**（[[Qwen-Image-Edit]]）："VLM 推理 + diffusion 推理"两段式 pipeline，部署比纯 diffusion 复杂
- **AR-Diffusion 混合**（Transfusion、JanusFlow）：单模型既要 AR decode 又要 N 步去噪，调度复杂度叠加

## Trade-offs

不论哪个子方向，多模态推理普遍有**更大的内存占用 + 更高的延迟 + 更复杂的 batching/scheduling 需求**。这也是 [[SGLang]] / [[vLLM]] 等推理引擎单独处理多模态模型的原因之一。

> [!quote] Karpathy 2025 年终回顾
> Gemini Nano Banana 这种"图文联合生成"代表了 LLM 与人交互的新方向——人偏好视觉而非纯文本输出，未来 LLM 应用可能更多用图像、infographic、白板等格式"说话"[^1]。
>
> Infra 含义：未来 LLM 推理的"输出"会越来越多包含非文本 modality，对 infra 的压力还会持续增长。

## Related

- [[Multimodal Models]] —— 多模态全景导航
- [[VLM Architecture]] —— VLM 三段式架构
- [[UFM]] —— UFM 三范式
- [[KV Cache]] —— VLM 多图场景的容量压力
- [[SGLang]] / [[vLLM]] —— 多模态推理引擎

[^1]: Andrej Karpathy (2025-12). *2025 LLM Year in Review*. [[Sources/Clippings/2025 LLM Year in Review]]
