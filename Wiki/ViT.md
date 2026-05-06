---
aliases: [Vision Transformer, 视觉 Transformer]
created: 2026-05-06
updated: 2026-05-06
---

# ViT

ViT (Vision Transformer) 把 NLP 里的 Transformer 直接搬到图像上——一张图被切成 N 个 patch，每个 patch 当作"一个词"，标准 self-attention 处理。Google 2020 年提出[^1]，证明只要数据够大，**Transformer 在视觉任务上能稳定打过 CNN**，从此成为现代多模态的视觉编码器默认选择。

## Mechanism

把一张 224×224 的图，按 16×16 切片：

1. 切出 14×14 = 196 个 patch，每个 patch flatten 成 768 维向量（16×16×3）
2. 线性投影到 embedding 维度（ViT-B 是 768 维）
3. 在序列前面加一个可学习的 `[CLS]` token
4. 加位置编码（原版是可学习的，后续多用 2D sinusoidal 或 RoPE）
5. 标准 Transformer encoder：multi-head self-attention + FFN，重复 L 层
6. 输出 197 个 token 的 embedding 序列

```text
图像 224×224 → 切 196 patch (16×16) → 196 个 patch token
                                              +
                                         1 个 [CLS] token
                                         + 位置编码
                                              ↓
                                     L 层 Transformer
                                              ↓
                                     197 个 768 维向量
```

Classification 任务用 `[CLS]` token；下游 multimodal 任务（[[CLIP]] / [[LLaVA]]）通常**丢掉 `[CLS]`，用全部 patch token**——拼接到 LLM 输入序列前。

## 关键变种与尺寸

ViT 论文给了 3 档基础尺寸，后续社区在不同方向各自演化：

| 尺寸 | 层数 L | 维度 | 头数 | 参数量 |
|---|---|---|---|---|
| ViT-Base (ViT-B) | 12 | 768 | 12 | ~86M |
| ViT-Large (ViT-L) | 24 | 1024 | 16 | ~307M |
| ViT-Huge (ViT-H) | 32 | 1280 | 16 | ~632M |

主流功能性变种：

- **CLIP-ViT**（OpenAI 2021）：用文本对齐训练，让 ViT 输出和文本 embedding 同空间——是开源 VLM 视觉编码器的事实标准[^2]
- **SigLIP**（Google 2023）：CLIP 的损失函数从 softmax 换成 sigmoid，**训练更稳、小 batch 也能做对比学习**——[[Qwen3-VL]]、[[InternVL3]] 主要用 SigLIP
- **DINOv2**（Meta 2023）：自监督训练，不需要文本配对——纯视觉表示更强，常用于 dense prediction
- **DeiT**（Meta 2021）：数据高效训练（用 ImageNet 而非 JFT-300M），蒸馏 + 改进训练策略

## ViT vs CNN

| 维度 | CNN（ResNet 等） | ViT |
|---|---|---|
| 感受野 | 局部，靠堆层数扩 | **全局**（任意 patch 互相 attention） |
| 归纳偏置 | 平移不变性 + 局部性强 | 弱——主要靠数据学 |
| 数据需求 | 中等（ImageNet 1M 即可） | **高**（需 100M+ 才能优于 CNN） |
| Scaling | 边际递减早 | **持续 scale**——和 LLM 一样 |
| 推理 | 卷积优化成熟 (cuDNN) | attention 是瓶颈（FlashAttention 改善） |

> [!note] 为什么要换掉 CNN
> 关键是 **scaling law**——CNN 在 1B+ 参数 / 100M+ 图像规模下增益乏力；ViT 没有这个瓶颈。2021 年 [[CLIP]] 用 4 亿对图文 + ViT 训练后，整个行业（特别是多模态）转向 ViT 路线[^2]。

## 在多模态中的角色

ViT 是 [[VLM Architecture]] 三段式里的"vision encoder"——把图像编码成 LLM 能消费的 embedding 序列。一个完整 VLM forward：

1. 图像 → ViT → 196 个 768 维 patch embedding
2. **Connector**（一层小 MLP）→ 升维到 LLM 隐层维度（如 4096）
3. 这 196 个图像 token 拼到文本 token 序列前
4. LLM 标准 next-token 预测

**几个工程细节按需 tune**：

- **高分辨率支持**：原 ViT 固定输入尺寸，但 [[Qwen3-VL]] / [[InternVL3]] 通过"动态切片 + 2D RoPE"支持任意分辨率
- **视频**：把每帧都过 ViT，再用时间维度的 attention 聚合
- **推理瓶颈**：单图 ViT 是一次 forward，相对 LLM decode 不算贵；但**多图 / 长视频场景下 prefill 阶段视觉 token 数能爆到数万**——这是 [[Multimodal Inference Considerations]] 里"VLM 推理是放大版 LLM"的根源

## Related

- [[Multimodal Models]] —— 多模态全景导航
- [[VLM Architecture]] —— ViT 在 VLM 三段式里的位置
- [[CLIP]] —— CLIP-ViT 的训练方法
- [[DiT]] —— Transformer 在生成端的对应物
- [[Qwen3-VL]] / [[InternVL3]] / [[Kimi-VL]] —— 当前主流 VLM 的视觉 backbone

[^1]: Dosovitskiy et al., Google (2020). *An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale*. [[Sources/Papers/2010.11929v2.pdf]]
[^2]: Radford et al., OpenAI (2021). *Learning Transferable Visual Models From Natural Language Supervision*. [[Sources/Papers/2103.00020v1.pdf]]
