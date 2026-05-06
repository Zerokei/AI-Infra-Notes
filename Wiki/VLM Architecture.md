---
aliases: [VLM 架构, 视觉语言模型架构]
created: 2026-05-06
updated: 2026-05-06
---

# VLM Architecture

VLM（Vision-Language Model）的标准架构是 [[LLaVA]] 之后开源 VLM 的事实标准——**vision encoder + connector + LLM** 三段式[^1]：

> 图像 → vision encoder → connector → 与文本 token 拼接 → LLM → 输出文本

## Mechanism

一个具体例子，[[LLaVA]] 处理 "这张图里有几只猫？"[^2]：

1. 图像切成 24×24 = 576 个 patch（每个 14×14 像素），过 CLIP 视觉 encoder 后变成 576 个 768 维向量
2. 一个 2 层 MLP（这就是 connector）把 768 维投到 LLM 需要的 4096 维
3. 这 576 个图像 token 拼到文本 token "这张图里有几只猫？" 之前
4. LLM 一气呵成 next-token 生成 "图里有 2 只猫"

三段的常见选型[^1]：

- **视觉 encoder**：通常是预训练的 [[ViT]]（Vision Transformer），常见是 [[CLIP]]-ViT 或 SigLIP 的视觉部分
- **Connector**（也叫 projector / adapter，三个名字一回事）：linear 或小 MLP，做维度对齐
- **LLM**：直接复用预训练 LLM。训练时通常先冻住 LLM 只训 connector；instruction-tuning 阶段再解冻——LLaVA 的关键贡献 **visual instruction tuning** 就是用 GPT-4 合成图文指令数据让 LLM 学会"看图回答指令"[^2]

> [!note] 视觉 token 的接入位置有两种
> 多数 VLM（LLaVA、Molmo、Qwen2-VL）把视觉 token 拼在文本 token 前作为输入序列。少数例外（Llama 3.2 11B/90B、Flamingo）通过 **cross-attention** 注入 LLM 内部某些层——好处是视觉信息不占 KV cache、LLM 主体可冻结当作纯文本模型用，代价是 LLM 架构要改[^1]。

## Related

- [[Multimodal Models]] —— 多模态全景导航
- [[LLaVA]] —— 三段式开山作
- [[ViT]] —— 视觉 encoder 的主流选择
- [[CLIP]] —— CLIP-ViT 的来源
- [[Multimodal Inference Considerations]] —— 三段式对推理 infra 的影响

[^1]: Sebastian Raschka (2024-11). [[Sources/Clippings/Understanding Multimodal LLMs]]
[^2]: Liu et al. (2023). *Visual Instruction Tuning*. [[Sources/Papers/2304.08485v2.pdf]]
