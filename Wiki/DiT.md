---
aliases: [Diffusion Transformer, 扩散 Transformer]
created: 2026-05-06
updated: 2026-05-06
---

# DiT

DiT (Diffusion Transformer) 把扩散模型里的去噪网络从 U-Net 换成 Transformer——William Peebles & Saining Xie 2022 年提出[^1]。**论文证明 Transformer 不仅能替代 U-Net，scaling 行为还更好**（同算力下越大越好），由此成为现代 T2I / T2V 主干的事实标准——FLUX、Stable Diffusion 3、[[Qwen-Image-2.0]]、Sora 都基于 DiT 思路。

## DiT vs 传统 Diffusion

DiT 不是新的 diffusion 范式——**diffusion 过程本身完全没变**，只是把每步去噪那个神经网络从 **U-Net** 换成了 **Transformer**。

| 维度 | 传统 Diffusion (SD 1.x/2.x, DALL-E 2) | DiT-based (SD3, FLUX, Qwen-Image) |
|---|---|---|
| **去噪主干** | **U-Net** (CNN + 跳跃连接) | **Transformer** |
| Conditioning 注入 | cross-attention | AdaLN-Zero / 同序列 self-attn (MMDiT) |
| Scaling | <1B 表现良好，>1B 增益乏力 | **持续 scale 到 20B+** |
| 长 latent 序列 | O(n) 卷积友好 | O(n²) attention 瓶颈 |
| 推理优化 | 卷积成熟 (cuDNN) | attention 优化 (FlashAttention) |
| 用户视角 | 同样的 prompt → 出图接口 | 完全相同——只是质量更好 |

> [!note] 不是 DiT 必然带来的进步
> 同期还发生了：VAE 升级（Wan2.1-VAE）、text encoder 升级（CLIP → T5 → LLM/VLM）、训练目标演进（ε → v → rectified flow）。这些**理论上 U-Net 也能用**，但因为时间上和 DiT 一起出现，常被误认为是 DiT 自带特性。**DiT 的核心贡献只是"换个主干"**——和 [[ViT]] 替换 CNN 是同一个 scaling 故事。

## Mechanism

DiT 不直接处理像素——先用 VAE 把图像压到 latent 空间（典型 8× 压缩：1024×1024 → 128×128），DiT 在 latent 上做 N 步去噪，最后 VAE decoder 还原回像素。

```text
像素图 1024×1024
    ↓ VAE encoder (8×)
latent 128×128 (含噪声)
    ↓ patchify (2×2 patch)
4096 个 latent token
    ↓ ↻ ↻ ↻ N 步去噪 (典型 20-50 步)
    │  每步 Transformer:
    │    - self-attention over 4096 latent token
    │    - cross-attention with text condition
    │    - 加 timestep embedding (告诉模型现在是第几步)
    │    - 输出预测的 noise (或 v-prediction / 矫正流目标)
    ↓
clean latent
    ↓ VAE decoder
像素图 1024×1024
```

> [!note] AdaLN-Zero：DiT 的 conditioning 巧思
> 原 DiT 的关键贡献之一是 **AdaLN-Zero**——把 timestep + class label 通过自适应 LayerNorm 注入每一层，且初始化为零（让网络从 identity 开始训）。比 cross-attention 更高效，至今仍被 SD3 / FLUX 沿用[^1]。

## 关键变种：MMDiT

DiT 原版的"text 通过 cross-attention 注入"路线，2024 年被 **MMDiT** 取代：

| 路线 | 文本和图像怎么融合 | 代表 |
|---|---|---|
| 原版 DiT | 图像 token self-attn；text 通过 cross-attn 单向注入 | DiT (2022)、PixArt-α |
| **MMDiT** (Multi-Modal DiT) | 文本 token 和图像 token **拼到同一个 self-attn 序列里互相看**，平等融合 | Stable Diffusion 3、FLUX、[[Qwen-Image]] / [[Qwen-Image-2.0]] |

MMDiT 的提升来自"双向"——文本可以根据当前图像状态调整，图像也能根据文本细节微调。

代表模型尺寸：

| 模型 | DiT 主干参数 | text encoder |
|---|---|---|
| Stable Diffusion 3 | 2-8B 多档 | CLIP + T5-XXL |
| FLUX.1 dev | ~12B | CLIP + T5-XXL |
| FLUX.2 dev | ~12B (推测) | (升级版 text encoder) |
| Qwen-Image (v1) | 20B MMDiT | frozen Qwen2.5-VL |
| **[[Qwen-Image-2.0]]** | **7B MMDiT** | frozen Qwen3-VL 8B |

## 推理特性与加速路径

DiT 的推理 profile 和 LLM 完全不同。理解这部分有助于把握为什么 [[Multimodal Inference Considerations]] 里把 T2I / UFM 划成独立 profile。

### 为什么 LLM 风格的 KV Cache 不适用

| 维度 | LLM autoregressive decode | DiT 去噪 |
|---|---|---|
| 序列长度 | 1 → 2 → 3 → ...（增长） | **固定**（如 4096 token，N 步内不变） |
| 每步只算 | 新 token 的 K/V | **所有 token** 的 K/V 都重新算 |
| 过去 K/V 是否变 | 不变 → 可缓存 | 输入是含噪 latent，**每步噪声水平不同 → K/V 全刷** |
| 算力特性 | memory-bound | **compute-bound** |

所以"把上一步的 image-token K/V 留着这一步用"——逻辑上行不通。

### DiT 自己的几种加速路径

**(1) Conditioning 路径的 K/V 跨步复用** ✅（**这是真 KV cache，可用**）

文本 prompt 编码（CLIP/T5/Qwen-VL）输出在 N 步去噪里**永远是同一个**——所以：

- Text encoder 只 forward 一次
- Text token 的 K/V（被图像 token 在 cross-attn 或 MMDiT 联合 attn 里查询时用到）**算一次缓存 N 步用**——这是真正意义的 KV cache

SD3 / FLUX / MMDiT 实现里都有。但**它优化的是 conditioning 路径，不是图像 token 自己**。

**(2) 步间特征缓存** ✅（"伪 KV cache"，效果显著）

观察：相邻去噪步之间，**深层特征变化很小**。代表方法：

| 方法 | 思路 | 效果 |
|---|---|---|
| **DeepCache** (CVPR 2024) | 高层跨多步复用，只更新浅层 | 2-3× 加速 |
| **TeaCache** | 自适应判断"这步要不要重算" | 1.5-2× |
| **Faster Diffusion** | cross-attn K/V 跨步复用 | 1.5× |

形式上像缓存，但是**步级**而非 token 级。

**(3) 步数压缩** ✅（**主路径**）

直接砍 N。从 50 步压到 4-8 步，比任何"复用"都更暴力：

- **LCM** (Latent Consistency Model)
- **SDXL Turbo** / **FLUX Schnell**
- **DMD-2**（[[Happy Horse 1.0]] 用这个把 50 步压到 8 步）
- **Rectified Flow**（FLUX 主线）

这条和 (1)(2) orthogonal——蒸馏后少步模型仍可叠加 K/V cache 和 step caching。

**(4) Sparse / Linear Attention** ✅（架构级）

视频生成时 token 数到几万，attention 的 O(n²) 是核心瓶颈：

- **Sparse DiT**：局部窗口
- **Linear / Mamba 替代**
- **3D RoPE + 时空窗口化**（Wan、Sora 路线）

> [!warning] 一句话总结
> **LLM 那种"自回归 token 缓存"在 DiT 上行不通**——但 DiT 有自己一套"步间复用 + 步数压缩 + 稀疏 attention"组合拳。**conditioning 路径上的 K/V cache 是唯一沿用 LLM 同名概念且确实有效的**——image token 自身不能用 KV cache。

## Related

- [[Multimodal Models]] —— 多模态全景导航
- [[ViT]] —— Transformer 在视觉理解端的对应物
- [[Qwen-Image]] / [[Qwen-Image-2.0]] —— MMDiT 在 T2I 的开源代表
- [[UFM]] —— DiT 是 representation-mediated 模块化联合 UFM 的"画笔"
- [[Multimodal Inference Considerations]] —— DiT 推理 profile 和 batching 策略
- [[KV Cache]] —— LLM 的 KV cache 概念（对比理解 DiT 为何不适用）

[^1]: Peebles & Xie (2022). *Scalable Diffusion Models with Transformers*. [[Sources/Papers/2212.09748v2.pdf]]
