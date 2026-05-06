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
- **模块化联合**（[[Qwen-Image-Edit]] / [[Qwen-Image-2.0]]）："VLM 推理 + diffusion 推理"两段式 pipeline，部署比纯 diffusion 复杂。v2.0 后 MMDiT 主干从 20B 砍到 7B、统一了 gen+edit 单模型，**单步算力降到约 1/3，但原生分辨率上调到 2K（视觉 token 数 ×4 → attention O(n²) ×16）部分抵消了瘦身收益**
- **AR-Diffusion 混合**（Transfusion、JanusFlow、[[BAGEL]]）：单模型既要 AR decode 又要 N 步去噪，调度复杂度叠加。BAGEL 这种"边推理边生图"在单请求内反复在 AR / diffusion 模式间切换，**SGLang / vLLM 现有调度都为纯 AR 设计，没有现成引擎能开箱即用**

## Omni 推理：延迟预算才是瓶颈，不是算力

[[Qwen2.5-Omni]] / [[Qwen3-Omni]] / [[Qwen3.5-Omni]] 这类对话型 omni 模型的瓶颈与生成型完全不同——**核心指标是 TTFT（首响应时间），不是吞吐**：

> [!warning] Omni 流式的核心 infra 痛点
> (1) **TTFT 预算紧**：speech 端到端首响应 232–320 ms（Qwen2.5-Omni 报告），意味着 vision/audio encode → Thinker prefill → Talker 启动整条链路必须在该预算内完成，**任何环节超时整个对话感都会崩**；
> (2) **长上下文 prefill 爆炸**：Qwen3.5-Omni 支持 256k 上下文 + 10h+ audio / 400s+ 视频输入，**prefill 阶段视觉 / 音频 token 数量级达数十万**，是真正算力大头；
> (3) **Audio 输出也是 AR**：Talker 流式生成 audio token，和 LLM decode 一样是 memory-bound（HBM 带宽限制权重加载），不是 T2I 那种 compute-bound；
> (4) **从 DiT 换到 RVQ 是关键加速**：Qwen3-Omni 的 Talker 用 DiT (Diffusion Transformer) 做语音合成——N 步去噪让首音频延迟难压；Qwen3.5-Omni 改成 **RVQ (Residual Vector Quantization) 离散 token 自回归生成**，每帧 1 步出，自然适配流式[^2]；
> (5) **多档尺寸是延迟优化的硬手段**：Qwen3.5-Omni 的 Plus / Flash / Light 三档大概率就是给不同延迟敏感场景留的——Light 推测为 realtime API 优化的小模型。

优化路径与 LLM/T2I 完全不同——**借鉴 ASR / TTS 流式工程经验**（如 chunk-wise streaming、并行 encoder、speculative decoding 在 audio token 上的应用）远比借鉴 LLM batching 优化更直接。

## 视频生成推理：步数压缩 + 联合音视频是关键

视频生成（Wan 2.5 / Wan 2.7 / [[Happy Horse 1.0]] / HunyuanVideo / CogVideoX）是 T2I profile 的"放大版"——**帧数 × 去噪步数 × 注意力序列长度三重压力同时存在**：

> [!warning] 视频生成 infra 的关键约束
> (1) **算力是 T2I 的 N×F 倍**（F = 帧数，常 100-500 帧；N = 步数）。一个 5 秒视频 @ 24 fps × 50 步 = 6000 次 full-frame forward；
> (2) **步数压缩是延迟生死线**：Happy Horse 1.0 用 **DMD-2 蒸馏到 8 步**才能 1080p 在 ~38 秒出片[^3]；不蒸馏的 base 模型相同分辨率延迟在分钟级；
> (3) **联合音视频生成 vs 拼接 TTS**：Happy Horse 1.0 的 15B 统一 Transformer 在单序列内同时处理文/图/视频/音频 token，原生联合合成（对话/环境音/Foley 一起出）。对比之下，旧 pipeline 是"先生成视频 → 再 TTS 配音 → lipsync 对齐"——**多一道对齐成本，且 lip-sync 误差累积**；
> (4) **超分而非原生 1080p**：多数视频模型 latent 空间生成低分辨率（如 256×256 latent），最后用专用 super-resolution 模块上采样到 1080p——**这部分独立于主 diffusion 模型，可以单独优化甚至跨请求复用**；
> (5) **Thinking Mode 加一道前置开销**：Wan 2.7 引入的"先理解 prompt 再生成"路线把推理分两阶段（reason + generate）——先做一次 LLM-style 长 CoT，再做 diffusion 生成。**好处是生成质量提升，代价是 TTFT 延后数秒**[^4]。

视频生成是当前**最 compute-heavy 的多模态推理负载**，专用推理引擎（如 vLLM-Omni、ComfyUI 集成路径）才刚起步——这是 ML infra 的下一个开放问题。

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
- [[Qwen3.5-Omni]] —— Omni 流式典型
- [[Happy Horse 1.0]] / [[Wan 2.7]] —— 视频生成代表
- [[KV Cache]] —— VLM 多图场景的容量压力
- [[SGLang]] / [[vLLM]] —— 多模态推理引擎

[^1]: Andrej Karpathy (2025-12). *2025 LLM Year in Review*. [[Sources/Clippings/2025 LLM Year in Review]]
[^2]: Xu et al., Qwen Team, Alibaba (2026-03). *Qwen3.5-Omni Technical Report*. [[Sources/Papers/2604.15804v2.pdf]]
[^3]: Alibaba (2026-04-09). *Happy Horse 1.0* (官方页). [https://happy-horse.art/](https://happy-horse.art/)
[^4]: Alibaba Tongyi Lab (2026-04). *Wan 2.7: Breakthrough AI Image & Video Generation Model with Thinking Mode*. [https://www.cliprise.app/news/wan-2-7-video-release](https://www.cliprise.app/news/wan-2-7-video-release)
