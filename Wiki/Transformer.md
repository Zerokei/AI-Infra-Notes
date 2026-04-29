---
date-created: 2026-04-08 09:19
date-updated: 2026-04-15 01:40
tags:
  - 🤖
---

# Transformer

> [!abstract] Model architecture
> ![[Pasted image 20260408092156.png|450]]

## Attention 机制

### Self-Attention

- 让每个 token 在编码时，同时考虑序列中的其他 token
- 任意两个 token 之间直接计算关系，不受距离限制

#### Q、K、V 变换

输入矩阵为：

$$
X \in \mathbb{R}^{n \times d}
$$

通过三个线性变换得到：

$$
Q = XW^Q,\quad K = XW^K,\quad V = XW^V
$$

其中：

- $W^Q, W^K, W^V \in \mathbb{R}^{d \times d_k}$
- $Q, K, V \in \mathbb{R}^{n \times d_k}$

#### Attention 计算

$$
\text{Attention}(Q, K, V)
=
\text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V
$$

#### 计算流程

1. **相似度计算**

$$
S = QK^T \in \mathbb{R}^{n \times n}
$$

表示 token 两两之间的相关性。

2. **缩放**

$$
S = \frac{QK^T}{\sqrt{d_k}}
$$

作用是防止点积过大，使 softmax 落入梯度很小的区域。

3. **归一化**

$$
A = \text{softmax}(S)
$$

其中每一行表示当前 token 对所有 token 的注意力分布。

4. **信息聚合**

$$
Z = AV \in \mathbb{R}^{n \times d_k}
$$

即每个 token 的新表示是所有 value 的加权和。

#### 直观理解

对于第 $i$ 个 token，其输出为：

$$
z_i = \sum_j A_{ij} v_j
$$

这表示当前 token 从所有 token 中按权重收集信息。

![[Pasted image 20260328182526.png]]

> [!info]- 计算流程详解
> ![[Pasted image 20260328182254.png]]

### Masked Self-Attention

与 Self-Attention 完全相同，但在 softmax 之前加入 Mask，把未来位置的分数设为 $-\infty$（softmax 后变为 0），使当前 token 只能看到自己和之前的 token。

作用：

- 保证生成过程满足自回归约束
- 防止训练时"偷看答案"

### Multi-Head Attention

使用多组不同的投影矩阵，在不同子空间中分别计算 attention：

$$
\text{head}_i = \text{Attention}(QW_i^Q, KW_i^K, VW_i^V)
$$

然后将多个头拼接：

$$
Z = \text{Concat}(head_1, \dots, head_h)W^O
$$

其中：

- 每个 head 的维度通常为 $n \times d_k$
- 拼接后得到 $n \times (h \cdot d_k)$
- 再通过输出矩阵 $W^O$ 映射回原维度

作用：

- 捕捉不同类型的关系
- 提高表达能力
- 让模型在不同子空间中学习不同依赖模式

![[Pasted image 20260407095836.png]]

![[Pasted image 20260407103313.png]]

### Feed Forward Neural Network

$$
\text{FFN}(x) = \max(0, xW_1 + b_1)W_2 + b_2
$$

#### 特点

- 两层全连接网络（MLP）
- 中间带非线性激活函数
- 对每个位置独立作用（position-wise）
- 维度变化通常为：

$$
d \rightarrow d_{ff} \rightarrow d
$$

#### 作用

- 引入非线性
- 对 attention 聚合后的结果做进一步特征变换

> [!info]+ 现代 LLM 的 MLP 变体：SiLU 替代 ReLU、SwiGLU 取代两矩阵 FFN
> Llama / Qwen / DeepSeek 等现代 LLM 已经不用原始 ReLU FFN，主流变体：
>
> - **激活函数**：ReLU → **SiLU**（$x \cdot \sigma(x)$）或 **GELU**，更平滑、训练更稳。
> - **Gated MLP（SwiGLU）**：把原本的两矩阵 FFN 扩成**三矩阵**——gate + up + down：
>   $$\text{SwiGLU}(x) = (\text{SiLU}(xW_{\text{gate}}) \odot (xW_{\text{up}}))\, W_{\text{down}}$$
>   相比原始 FFN 多一个"门控"乘法，质量明显提升，参数增加约 50%。
> - **典型维度**：Qwen 的 $d \to d_{ff} \to d = 4096 \to 11008 \to 4096$（$d_{ff} \approx 2.7 d$，与 SwiGLU 的参数增加相匹配）。
>
> 平时见到的"LLM 的 FFN"几乎都是 SwiGLU 形式，不是原始 ReLU 版本。

> [!info] Residuals
> 残差连接的形式为：
>
> $$
> \text{Output} = x + \text{Sublayer}(x)
> $$
>
> 作用：
> - 保留原始信息
> - 缓解梯度消失
> - 提升深层模型训练稳定性

> [!info] Layer Normalization
> 层归一化形式为：
>
> $$
> \text{LayerNorm}(x)
> $$
>
> 作用：
> - 稳定训练过程
> - 统一特征尺度

### Dense vs MoE 架构

上面描述的标准 FFN 是 **Dense 架构**——每个 token 都经过同一个 MLP。**MoE（Mixture of Experts）架构**则不同：

- MLP 被替换为多个更小的"专家"子 MLP（例如 8 个）；
- 一个 **router** 网络检查每个 hidden state，决定该 token 应激活哪些专家（例如每 token 仅激活 8 个中的 2 个）；
- 每个专家学到的"专长"是训练涌现，无法事先设计、难以人工解读。

> [!note]+ MoE 流行的原因在于可规模化，而非每参数质量更好
> 相同总参数量下，Dense 模型通常优于 MoE——因为 Dense 每个参数都作用于每个 token，效率更高。MoE 的价值在于**可训练性**：
>
> - 训练 600B Dense 模型目前计算不可行；
> - 但 600B MoE 模型每 token 只激活 50B 参数，训练可行。
>
> 这是工程权衡：用较低的 per-parameter 效率换取 Dense 无法达到的规模，再以规模获取 Dense 无法达到的能力。

代表模型：Mixtral 8×7B、DeepSeek-V2/V3、Qwen-MoE。对应的推理工程挑战是 **expert parallelism**（专家并行），见 [[vLLM]] 后续补充。

---

## Encoder-Decoder 架构

> 原始 Transformer[^1] 的完整架构，用于翻译等 seq2seq 任务。

### Embedding + Positional Encoding

将离散 token 映射为连续向量：

$$
X \in \mathbb{R}^{n \times d}
$$

- 每个 token 对应一个 $d$ 维向量
- 本质上是查 embedding 矩阵 $E \in \mathbb{R}^{|V| \times d}$

由于 Transformer 本身不具备顺序结构，需要显式引入位置信息：

$$
X = \text{Embedding} + \text{Positional Encoding}
$$

论文中的位置编码形式为：

$$
PE(pos, 2i) = \sin\left(\frac{pos}{10000^{2i/d}}\right)
$$

$$
PE(pos, 2i+1) = \cos\left(\frac{pos}{10000^{2i/d}}\right)
$$

### Encoder

由 $N$ 个相同结构的层堆叠而成，每层包含：

1. **Multi-Head Self-Attention**
2. **Feed Forward Neural Network**

每个子层都带残差连接 + Layer Normalization。

### Decoder

同样由 $N$ 个相同层堆叠而成，每层包含 3 个子模块：

1. **Masked Multi-Head Self-Attention** — 防止看到未来 token
2. **Encoder-Decoder Attention（Cross-Attention）** — Q 来自 Decoder，K 和 V 来自 Encoder，从 Encoder 的表示中提取相关信息
3. **Feed Forward Neural Network**

### Projection

经过 Decoder 后得到隐藏表示 $H$，通过 Linear 层映射到词表空间再做 Softmax：

$$
P = \text{softmax}(HW_o)
$$

其中 $W_o \in \mathbb{R}^{d \times |V|}$，每一行表示当前位置生成各个词的概率。

---

## Decoder-Only 架构（LLM）

现代 LLM（GPT、LLaMA、Claude 等）采用 decoder-only 架构，即只保留上述 Decoder 部分，去掉 Encoder 和 Cross-Attention：

```
输入 token
  ↓
Embedding + Positional Encoding
  ↓
┌────────────────────────────┐
│ Masked Self-Attention       │  × N 层
│ Feed Forward Neural Network │
│ (每层带残差连接 + 归一化)     │
└────────────────────────────┘
  ↓
Linear → Softmax → 输出概率分布
```

与完整 Transformer Decoder 的唯一结构区别：**没有 Encoder-Decoder Attention（Cross-Attention）**。其余组件完全一致。

现代 LLM 在此基础上还替换了一些具体组件：

| 组件 | 原始 Transformer | 现代 LLM |
|---|---|---|
| 位置编码 | 正弦固定编码 | RoPE 等 |
| 归一化 | Post-LayerNorm | Pre-LayerNorm / RMSNorm |
| 激活函数 | ReLU | SwiGLU / GeGLU |
| 注意力 | 标准 MHA | GQA 等 |

但这些不改变核心的 attention + 自回归生成逻辑。

### 层数 vs 头数的权衡

规模增长时，参数该怎么分配——更多层（深）还是更多头（宽）？

- **更多层数**：支持更深的推理，每层都是对 hidden state 的一轮精炼；
- **更多头数**：支持更丰富的 attention 模式，提供更多"视角"理解 token 关系。

能否采用"窄而深"（少头多层）或"宽而浅"（多头少层）的极端配置？研究表明不行——极端不平衡的架构通常表现较差。像人类学习一样，模型受益于**广度与深度的平衡**，大多数成功模型都保持接近方形的比例。

典型数字：

| 模型 | 层数 $L$ | head 数 $H$ | head 维度 $d_h$ | hidden size $d = H \cdot d_h$ |
|---|:---:|:---:|:---:|:---:|
| Llama-2-7B | 32 | 32 | 128 | 4096 |
| Llama-2-13B | 40 | 40 | 128 | 5120 |
| Llama-2-70B | 80 | 64 | 128 | 8192 |
| Qwen (Nano-vLLM 版本) | 24 | 32 | 128 | 4096 |

规模增加时，**层数和头数一起涨**，而 head 维度（128）几乎不变。

> [!tip] 模型能力的真正杠杆在训练数据与方法，而非架构极端
> 层数、头数等架构参数的可调空间有限。实际决定模型能力的是**训练数据**（有什么知识可学）与**训练方法**（知识如何被有效学到）。

---

[^1]: [Attention Is All You Need](https://arxiv.org/pdf/1706.03762)
[^2]: [The Illustrated Transformer (by Jay Alammar)](https://jalammar.github.io/illustrated-transformer/)
