---
date-created: 2026-04-08 09:20
date-updated: 2026-04-08 16:25
tags:
  - 🤖
---

# LLM Inference Optimization

## 前置知识

- [[Transformer#Decoder-Only 架构（LLM）]] — LLM 使用的模型结构
- [[Transformer#Attention 机制]] — Self-Attention、Multi-Head Attention
- [[KV Cache]] — 自回归推理的核心优化：缓存历史 token 的 K 和 V，避免重复计算

## 推理的开销模型

### 三项开销

每一步推理涉及三项开销，耗时取决于最慢的那项[^2][^3]：

$$t = \max\left(\frac{\text{FLOPs}}{C},\quad \frac{\text{显存读写量}}{M},\quad \frac{\text{网络传输量}}{B}\right)$$

以 A100 为例：

| 指标 | 符号 | 值 | 含义 |
|---|---|---|---|
| 算力 | $C$ | 312 TFLOPS/s | 每秒能做多少次运算 |
| 显存带宽 | $M$ | 1.5 TB/s | 每秒能读写多少数据 |
| 网络带宽 | $B$ | ~400 GB/s | 机器间每秒能传多少数据 |

单 GPU 推理时，网络通信可忽略。多卡并行（模型并行）时，网络通信成为第三个瓶颈维度。

### ops:byte 比值与 Memory Bound

$$R = \frac{C}{M} = \frac{312 \times 10^{12}}{1.5 \times 10^{12}} = 208$$

含义：每从显存读 1 byte，GPU 可以做 208 次运算。

- 每个权重参与的运算 **< 208 次** → **Memory Bound**（GPU 算完了在等数据）
- 每个权重参与的运算 **> 208 次** → **Flops Bound**（数据到了还在算）

batch（并发请求数）越大，每个权重被复用的次数越多（读一次权重，给 $B$ 个请求各算一次）。临界点是 batch = 208，超过则从 memory bound 翻转为 flops bound。

batching 几乎不增加单个请求的延迟[^3]。以 A100 上 4096×4096 矩阵乘法为例：

| | batch=1 | batch=256 |
|---|---|---|
| 读权重 | 330 μs | 350 μs |
| 计算 | 1.6 μs | 400 μs |
| 利用率 | 0.5% | ~100% |

读权重时间几乎不变，增加的只是计算时间——而这些计算本来就在 GPU 等待数据的空闲期内完成。

### Prefill vs Decode

**Decode（逐 token 生成）→ 几乎总是 Memory Bound**

batch=1 时每个权重只做 1 次运算，远低于 208。实际服务中 batch 很少到 200+，所以 decode 阶段几乎总是 memory bound。此时单 token 生成耗时约为：

$$t_{token} \approx \frac{2P}{N \times M}$$

其中 $P$=参数量，$N$=GPU 数。如 52B 模型、4 × A100 下约 **17 ms/token**。

> [!warning] 简化假设
> - 上述只考虑了读模型权重的带宽开销。实际每步还需要读 KV Cache，上下文很长时这部分开销会变得显著。
> - Self-attention 无法像 FFN 那样高效 batching——每个请求的 K、V 不同，无法跨请求共享[^3]。

**Prefill（处理 prompt）→ 通常是 Flops Bound**

Prefill 一次性并行处理整个 prompt 的所有 token，每个权重被所有 token 复用，复用次数很容易超过 208 → flops bound。这就是 Prefill 和 Decode 计算特性不同的根本原因。

### KV Cache 的显存开销

每一层 decoder 都要为每个请求的每个历史 token 存一份 K 和一份 V：

$$\text{KV Cache (bytes)} = 2 \times L \times B \times T \times H_{kv} \times D \times s$$

| 参数 | 含义 |
|---|---|
| $L$ | 层数 |
| $B$ | 并发请求数 |
| $T$ | 已缓存 token 数（prompt + 已生成） |
| $H_{kv}$ | KV head 数 |
| $D$ | head 维度 |
| $s$ | 每个元素字节数（bf16=2, fp8=1） |

每生成一个 token，KV Cache 固定增长（Token Tax）：以 Mistral 7B 为例，batch=1 时约 **128 KiB/token**[^1]。

GPU 显存要同时容纳模型权重和 KV Cache。权重固定不变，KV Cache 随 $T$ 和 $B$ 线性增长——**长上下文 + 大 batch 是显存爆炸的常见原因**。

### 前向传播的理论下限

前向传播本质上是 $L$ 次连续矩阵乘法（$L$ 约为层数的 2 倍）。在忽略常数的情况下[^3]：

$$t_{forward} \approx \frac{L \cdot R^3}{C}$$

一个反直觉的结论：**前向传播时间不取决于模型宽度**，只取决于层数 $L$、GPU 算力 $C$ 和 ops:byte 比值 $R$。宽度增大可以通过更多 GPU 并行来抵消。

将 GPU 分块和 batch 同时缩小 $k$ 倍，利用率降到 $1/k$，但速度提升 $k^2$ 倍。工程中接受 40% 利用率是常见做法。

## 优化方法

### 减少 KV Cache 显存

从显存公式出发，每个因子都是一个优化方向：

- **减少 $H_{kv}$**：让多个 Query head 共享同一组 K/V，而不是每个 head 独立存一份。比如 Mistral 7B 有 32 个 Query head 但只有 8 组 K/V，KV Cache 直接缩小 4 倍。
- **减小 $s$**：用更低精度存储 KV Cache（如 bf16→fp8），缓存减半。
- **限制 $T$**：只缓存最近 $W$ 个 token（滑动窗口），显存上限固定为 $W$ 而非无限增长。
- **减小 $B$**：减少并发请求数，线性缩减。

### 量化（Quantization）

模型参数位数越多越精确，但也越占空间[^4]：

| 类型 | 每参数字节 | 70B 模型显存 |
|---|---|---|
| FP32（全精度） | 4 | 280 GB |
| BF16（训练常用） | 2 | 140 GB |
| INT8 | 1 | 70 GB |
| INT4 | 0.5 | 35 GB |

量化就是把参数从高位压缩到低位，用更少的显存存模型和 KV Cache——对应公式里的 $s$ 变小。代价是精度下降，值之间的区分度变低。

**核心思想：线性映射。** 找到数据中绝对值最大的值 $\alpha$，等比缩放到整数范围：

$$s = \frac{127}{\alpha}, \quad x_q = \text{round}(s \cdot x)$$

反量化时除以 $s$ 近似还原，但会有误差——两个原本不同的浮点数可能被映射到同一个整数。

模型里需要量化的有两类东西：**权重**（训练后固定，可以离线精心量化）和**激活**（输入经过每层时产生的中间结果，每次输入都不一样，量化更难）。

量化的时机也有两种：训练完成后再量化（**PTQ**，快但精度损失可能较大），或训练时就模拟量化影响（**QAT**，更准确但需要重新训练）。

> [!info]- 进阶：常见的具体量化方法
> - **GPTQ**：4-bit PTQ，量化一个权重后把误差补偿到相邻权重，逐层处理
> - **GGUF**：分块量化，支持 CPU offload，适合显存不足时部分层跑在 CPU 上
> - **BitNet 1.58b**：权重只有 {-1, 0, 1} 三个值，矩阵乘法变为纯加减法

### PD 分离

Prefill 是计算密集型（flops bound），Decode 是带宽受限型（memory bound）。将两者在架构和调度层面解耦，可以针对性地分配资源：

- Prefill 阶段：分配更多算力，一次性并行处理整个 prompt 并构建 KV Cache
- Decode 阶段：优化显存带宽利用，支持 continuous batching（新请求随时插入，不等当前 batch 全部结束）

[^1]: [KV Cache in LLM Inference (by Ayoub Nainia)](https://pub.towardsai.net/kv-cache-in-llm-inference-7b904a2a6982)
[^2]: [Transformer Inference Arithmetic (by kipply)](https://kipp.ly/p/transformer-inference-arithmetic)
[^3]: [How Fast Can We Perform a Forward Pass? (by Jacob Steinhardt)](https://bounded-regret.ghost.io/how-fast-can-we-perform-a-forward-pass/)
[^4]: [A Visual Guide to Quantization (by Maarten Grootendorst)](https://newsletter.maartengrootendorst.com/p/a-visual-guide-to-quantization)
