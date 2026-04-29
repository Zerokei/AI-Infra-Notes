---
aliases: []
created: 2026-04-08
updated: 2026-04-29
---

# Transformer

Transformer 是当代 LLM 训练与推理所有讨论的基础模型架构[^1]：自回归生成、KV cache、张量并行、PagedAttention、speculative decoding 等推理优化都以它为前提。本页面只覆盖 AI Infra 视角下需要建立的概念——架构骨架、attention 机制、encoder-decoder 与 decoder-only 的差异，以及由此衍生的算力 / 显存 / 并行特征——不展开深度学习理论。

![[Attachments/pics/Pasted image 20260408092156.png|450]]
*图：原始 Transformer 论文中的 encoder-decoder 整体结构。*[^1]

## Background

2017 年 Vaswani 等人提出 Transformer[^1]，最初目标是替代 RNN/LSTM 在机器翻译中的角色。RNN 的根本问题在于序列依赖：第 $t$ 步必须等第 $t-1$ 步算完，无法并行；而长距离依赖会随时间衰减。Transformer 用 **attention** 取代循环，使任意两个 token 直接交互，不受距离限制；同时序列内所有位置可以并行计算，正好契合 GPU 的批量矩阵乘特性。这种"可并行 + 长距离"的组合是 Transformer 能够规模化到千亿参数、并最终演化为 LLM 主干的根本原因[^2]。

## Mechanism

### Attention 机制

#### Self-Attention

Self-Attention 让每个 token 在编码时同时考虑序列中其他所有 token，任意两个 token 之间直接计算关系，不受距离限制。

输入矩阵为：

$$
X \in \mathbb{R}^{n \times d}
$$

通过三个线性变换得到 Query、Key、Value：

$$
Q = XW^Q,\quad K = XW^K,\quad V = XW^V
$$

其中 $W^Q, W^K, W^V \in \mathbb{R}^{d \times d_k}$，$Q, K, V \in \mathbb{R}^{n \times d_k}$。

Attention 计算为：

$$
\text{Attention}(Q, K, V)
=
\text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V
$$

分四步：

1. **相似度**：$S = QK^T \in \mathbb{R}^{n \times n}$，表示 token 两两之间的相关性。
2. **缩放**：除以 $\sqrt{d_k}$，防止点积过大使 softmax 落入梯度极小的区域。
3. **归一化**：$A = \text{softmax}(S)$，每一行表示当前 token 对所有 token 的注意力分布。
4. **聚合**：$Z = AV \in \mathbb{R}^{n \times d_k}$，每个 token 的新表示是所有 value 的加权和。

对第 $i$ 个 token，其输出为 $z_i = \sum_j A_{ij} v_j$——按权重从所有 token 收集信息。

![[Attachments/pics/Pasted image 20260328182526.png]]
*图：Self-Attention 中 Q、K、V 的来源与 attention 分数的几何含义。*[^2]

> [!info]- Self-Attention 矩阵流水线全貌
> ![[Attachments/pics/Pasted image 20260328182254.png]]
> *图：从 X 到 Z 的完整矩阵运算流程。*[^2]

#### Masked Self-Attention

与 Self-Attention 完全相同，但在 softmax 之前加入 Mask，把未来位置的分数设为 $-\infty$（softmax 后变为 0），使当前 token 只能看到自己和之前的 token。作用是保证生成过程满足自回归约束，并防止训练时"偷看答案"。这是 [[KV Cache]] 能够生效的前提：旧 token 的输出不依赖未来 token，新增 token 不会改写历史。

#### Multi-Head Attention

使用 $h$ 组不同的投影矩阵，在不同子空间中分别计算 attention：

$$
\text{head}_i = \text{Attention}(QW_i^Q, KW_i^K, VW_i^V)
$$

然后将多个头拼接，再通过输出矩阵 $W^O$ 映射回原维度：

$$
Z = \text{Concat}(head_1, \dots, head_h)W^O
$$

每个 head 维度通常为 $n \times d_k$，拼接后得到 $n \times (h \cdot d_k)$。多头让模型在不同子空间学习不同的依赖模式，提高表达能力。

![[Attachments/pics/Pasted image 20260407095836.png]]
*图：Multi-Head Attention 把 Q/K/V 切成多个子空间并行计算。*[^2]

![[Attachments/pics/Pasted image 20260407103313.png]]
*图：各 head 输出拼接后通过 $W^O$ 映射回原维度。*[^2]

### Embedding 与 Positional Encoding

将离散 token 映射为连续向量 $X \in \mathbb{R}^{n \times d}$——本质上是查 embedding 矩阵 $E \in \mathbb{R}^{|V| \times d}$。由于 Transformer 本身不具备顺序结构（attention 对位置无感），需要显式注入位置信息：

$$
X = \text{Embedding} + \text{Positional Encoding}
$$

原论文采用正弦余弦固定编码：

$$
PE(pos, 2i) = \sin\left(\frac{pos}{10000^{2i/d}}\right),\quad PE(pos, 2i+1) = \cos\left(\frac{pos}{10000^{2i/d}}\right)
$$

现代 LLM 多用 RoPE（旋转位置编码）等相对位置方案，更适合长上下文外推。

### Feed Forward Neural Network

每个 attention 子层后接一个 position-wise 的两层 MLP：

$$
\text{FFN}(x) = \max(0, xW_1 + b_1)W_2 + b_2
$$

特点是中间带非线性激活、对每个位置独立作用、维度变化为 $d \rightarrow d_{ff} \rightarrow d$。作用是为 attention 聚合后的结果引入非线性、做进一步特征变换。

> [!info]+ 现代 LLM 的 MLP 几乎都换成 SwiGLU
> Llama / Qwen / DeepSeek 等已经不用原始 ReLU FFN，主流变体：
>
> - **激活函数**：ReLU → **SiLU**（$x \cdot \sigma(x)$）或 **GELU**，更平滑、训练更稳。
> - **Gated MLP（SwiGLU）**：把原本的两矩阵 FFN 扩成**三矩阵**——gate + up + down：
>   $$\text{SwiGLU}(x) = (\text{SiLU}(xW_{\text{gate}}) \odot (xW_{\text{up}}))\, W_{\text{down}}$$
>   多一个"门控"乘法，质量明显提升，参数增加约 50%。
> - **典型维度**：Qwen 的 $d \to d_{ff} \to d = 4096 \to 11008 \to 4096$（$d_{ff} \approx 2.7 d$，与 SwiGLU 的参数增加相匹配）。
>
> 平时见到的"LLM 的 FFN"几乎都是 SwiGLU 形式，不是原始 ReLU 版本。

### 残差连接与 Layer Normalization

每个子层（attention 或 FFN）的输出形式为 $\text{Output} = \text{LayerNorm}(x + \text{Sublayer}(x))$。残差保留原始信息、缓解梯度消失，使深层模型可训练；LayerNorm 统一特征尺度、稳定训练。原始 Transformer 用 Post-LayerNorm（先残差后归一化），现代 LLM 普遍切到 Pre-LayerNorm 并用 RMSNorm 替代标准 LN，进一步降低算子开销并提高训练稳定性。

## Variants & Comparisons

### Encoder-Decoder 架构

原始 Transformer[^1] 的完整架构，用于翻译等 seq2seq 任务，由两侧堆叠 $N$ 层组成：

- **Encoder** 每层包含 Multi-Head Self-Attention + FFN，每个子层都带残差 + LayerNorm。
- **Decoder** 每层包含 3 个子模块：Masked Multi-Head Self-Attention（防止看到未来 token）、Encoder-Decoder Attention（Cross-Attention，Q 来自 Decoder，K 和 V 来自 Encoder）、FFN。
- **Projection**：Decoder 输出隐藏表示 $H$ 后，通过 Linear 层映射到词表空间再做 Softmax，$P = \text{softmax}(HW_o)$，$W_o \in \mathbb{R}^{d \times |V|}$。

### Decoder-Only 架构（LLM）

现代 LLM（GPT、LLaMA、Claude 等）采用 decoder-only 架构——只保留 Decoder，去掉 Encoder 和 Cross-Attention：

```
输入 token
  ↓
Embedding + Positional Encoding
  ↓
┌────────────────────────────┐
│ Masked Self-Attention       │  × N 层
│ Feed Forward Neural Network │
│ (每层带残差连接 + 归一化)    │
└────────────────────────────┘
  ↓
Linear → Softmax → 输出概率分布
```

与完整 Transformer Decoder 的唯一结构区别是**没有 Cross-Attention**，其余组件完全一致。Decoder-only 的好处是结构简洁、训练目标统一（全部是 next-token prediction），且天然适合 [[KV Cache]] 加速——这是 Encoder-Decoder 推理时无法享受的优化。

现代 LLM 在此基础上还替换了几个具体组件：

| 组件 | 原始 Transformer | 现代 LLM |
|---|---|---|
| 位置编码 | 正弦固定编码 | RoPE 等 |
| 归一化 | Post-LayerNorm | Pre-LayerNorm / RMSNorm |
| 激活函数 | ReLU | SwiGLU / GeGLU |
| 注意力 | 标准 MHA | GQA / MQA 等 |

但这些不改变核心的 attention + 自回归生成逻辑，KV Cache、Speculative Decoding 等推理优化对所有变体都适用。

### Dense vs MoE

上述标准 FFN 是 **Dense 架构**——每个 token 都经过同一个 MLP。**MoE（Mixture of Experts）架构**则不同：

- MLP 被替换为多个更小的"专家"子 MLP（例如 8 个）；
- 一个 **router** 网络检查每个 hidden state，决定该 token 应激活哪些专家（例如每 token 仅激活 8 个中的 2 个）；
- 每个专家学到的"专长"是训练涌现，无法事先设计、难以人工解读。

> [!note]+ MoE 用 per-parameter 效率换可规模化训练
> 相同总参数量下，Dense 模型通常优于 MoE——Dense 每个参数都作用于每个 token，效率更高。MoE 的价值在于**可训练性**：
>
> - 训练 600B Dense 模型目前计算不可行；
> - 但 600B MoE 模型每 token 只激活 50B 参数，训练可行。
>
> 这是工程权衡：用较低的 per-parameter 效率换取 Dense 无法达到的规模，再以规模获取 Dense 无法达到的能力。

代表模型：Mixtral 8×7B、DeepSeek-V2/V3、Qwen-MoE。对应的推理工程挑战是 **expert parallelism**（专家并行），见 [[vLLM]] 后续补充。

## Trade-offs

### 算力可并行但显存负担转移到 KV Cache

Transformer 抛弃了 RNN 的串行依赖，序列内所有位置可在 GPU 上一次矩阵乘完成——Prefill 阶段几乎总是 flops bound。但代价是 attention 的 $O(n^2)$ 复杂度：序列长度翻倍，attention 矩阵和 KV Cache 都按平方/线性放大。Decode 阶段更是几乎总是 memory bound，每生成一个 token 都要把全部权重 + 全部历史 KV 从 HBM 读一遍。具体的 ops:byte 与 bound 分析见 [[LLM Inference Optimization]]。

### 层数与头数倾向方形比例

规模增长时，参数该怎么分配——更多层（深）还是更多头（宽）？

- **更多层数**支持更深的推理，每层都是对 hidden state 的一轮精炼；
- **更多头数**支持更丰富的 attention 模式，提供更多"视角"理解 token 关系。

研究表明极端"窄而深"或"宽而浅"配置通常表现较差——模型受益于**广度与深度的平衡**，多数成功模型保持接近方形的比例。典型数字：

| 模型 | 层数 $L$ | head 数 $H$ | head 维度 $d_h$ | hidden size $d = H \cdot d_h$ |
|---|:---:|:---:|:---:|:---:|
| Llama-2-7B | 32 | 32 | 128 | 4096 |
| Llama-2-13B | 40 | 40 | 128 | 5120 |
| Llama-2-70B | 80 | 64 | 128 | 8192 |
| Qwen (Nano-vLLM 版本) | 24 | 32 | 128 | 4096 |

规模增加时，**层数和头数一起涨**，而 head 维度（128）几乎不变。

> [!tip] 模型能力的真正杠杆在训练数据与方法，而非架构极端
> 层数、头数等架构参数的可调空间有限。实际决定模型能力的是**训练数据**（有什么知识可学）与**训练方法**（知识如何被有效学到）。

### 模型并行的天然切分维度

Transformer 的层数 $L$、头数 $H$、hidden size $d$、序列长度 $n$ 各自给出一个并行维度：层数对应 pipeline parallelism，头数和 hidden size 对应 tensor parallelism，序列长度对应 sequence parallelism，batch 维度对应 data parallelism。这种"天然多维可切分"是 Transformer 能够规模化到千卡训练的根本原因，也是现代 LLM 推理框架（[[vLLM]]、SGLang 等）调度层的设计起点。

## Related

- [[KV Cache]] —— Masked Self-Attention 在自回归推理下的核心缓存优化
- [[LLM Inference Optimization]] —— 基于 Transformer 算力 / 带宽特征展开的工程优化全景
- [[PagedAttention]] —— 管理 KV Cache 显存碎片的页表式方案
- [[vLLM]] —— 以 PagedAttention 为核心的开源推理框架
- [[SGLang]] —— RadixAttention + 编程接口

[^1]: [Attention Is All You Need](https://arxiv.org/pdf/1706.03762)
[^2]: [[Sources/Clippings/The Illustrated Transformer]]
