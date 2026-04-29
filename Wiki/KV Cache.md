---
date-created: 2026-04-08 17:00
date-updated: 2026-04-08 17:00
tags:
  - 🤖
---

# KV Cache

## 为什么需要 KV Cache？

在自回归生成中，每一步 decoder 需要对**所有已有 token** 做 attention。如果不做优化（"vanilla" 方式）：

- 第 $n$ 步需要处理 $n$ 个 token，self-attention 代价为 $O(n^2)$
- 生成 $n$ 个 token 的总代价为 $O(1^2 + 2^2 + \cdots + n^2) = O(n^3)$

通过 KV Cache，每步只需计算**新 token** 的输出，复杂度从 $O(n^2)$ 降为 $O(n)$，总代价降为 $O(n^2)$[^1]。

## 核心思想：增量计算

Masked Self-Attention 的输出具有**增量性**：新增一个 token 不会改变之前 token 的输出。因此：

$$
Y_{n+1} = \begin{bmatrix} Y_n \\ y_{n+1} \end{bmatrix}
$$

前 $n$ 行不变，只需计算新的 $y_{n+1}$。

## 数学推导：为什么增量计算是正确的？

> 以 Masked Self-Attention 为例，逐步推导为什么新增 token 不影响旧 token 的输出。

### Step 1：Q、K、V 可以增量拼接

矩阵乘法对行是独立的——把两个矩阵上下拼起来再乘 $W$，等于分别乘 $W$ 再拼起来：

$$
Q_{n+1} = X_{n+1} W^Q
= \begin{bmatrix} X_n \\ x_{n+1} \end{bmatrix} W^Q
= \begin{bmatrix} X_n W^Q \\ x_{n+1} W^Q \end{bmatrix}
= \begin{bmatrix} Q_n \\ q_{n+1} \end{bmatrix}
$$

$K$ 和 $V$ 同理。所以新 token 只需算自己的 $q_{n+1}, k_{n+1}, v_{n+1}$，拼到旧的后面即可。

### Step 2：分块展开 attention 矩阵

将 $Q_{n+1} K_{n+1}^T$ 用分块矩阵乘法展开：

$$
Q_{n+1} K_{n+1}^T
= \begin{bmatrix} Q_n \\ q_{n+1} \end{bmatrix}
\begin{bmatrix} K_n^T & k_{n+1}^T \end{bmatrix}
= \begin{bmatrix}
Q_n K_n^T & Q_n k_{n+1}^T \\
q_{n+1} K_n^T & q_{n+1} k_{n+1}^T
\end{bmatrix}
$$

得到一个 $(n+1) \times (n+1)$ 的分块矩阵：

- **左上** $Q_n K_n^T$：旧 token 之间的 attention 分数（和之前完全一样）
- **右上** $Q_n k_{n+1}^T$：旧 token 对新 token 的分数（即将被 mask 掉）
- **左下** $q_{n+1} K_n^T$：新 token 对旧 token 的分数
- **右下** $q_{n+1} k_{n+1}^T$：新 token 对自己的分数

### Step 3：Mask 让旧 token "看不到"新 token

Mask 的作用是：每个 token 只能看到自己和之前的 token。右上角（旧 token 看新 token）被设为 $-\infty$：

$$
\text{Mask}\left(\frac{1}{\sqrt{d_k}} Q_{n+1} K_{n+1}^T\right)
= \begin{bmatrix}
\text{Mask}\left(\frac{1}{\sqrt{d_k}} Q_n K_n^T\right) & -\infty \\
\frac{1}{\sqrt{d_k}} q_{n+1} K_n^T & \frac{1}{\sqrt{d_k}} q_{n+1} k_{n+1}^T
\end{bmatrix}
$$

注意：左上角保持原来的 mask 模式不变；最后一行不需要额外 mask（第 $n+1$ 个 token 可以看到所有 token）。

### Step 4：softmax 逐行独立，$e^{-\infty} = 0$

softmax 对每一行独立做归一化，因此：

- **前 $n$ 行**：右上角是 $-\infty$，经 softmax 后变为 $0$。所以前 $n$ 行的 softmax 结果和没有第 $n+1$ 个 token 时**完全一样**
- **第 $n+1$ 行**：正常计算 softmax

$$
\text{softmax}(\cdots)
= \begin{bmatrix}
\text{softmax}\left(\text{Mask}\left(\frac{1}{\sqrt{d_k}} Q_n K_n^T\right)\right) & 0 \\
\text{softmax}\left(\frac{1}{\sqrt{d_k}} q_{n+1} K_{n+1}^T\right)
\end{bmatrix}
$$

### Step 5：乘以 V，前 n 行输出不变

$$
Y_{n+1}
= \begin{bmatrix}
\text{softmax}\left(\text{Mask}\left(\frac{1}{\sqrt{d_k}} Q_n K_n^T\right)\right) & 0 \\
\text{softmax}\left(\frac{1}{\sqrt{d_k}} q_{n+1} K_{n+1}^T\right)
\end{bmatrix}
\begin{bmatrix} V_n \\ v_{n+1} \end{bmatrix}
$$

- **前 $n$ 行**：权重矩阵右侧是 $0$，所以 $v_{n+1}$ 的权重为零，结果就是原来的 $Y_n$
- **第 $n+1$ 行**：$y_{n+1} = \text{softmax}\left(\frac{q_{n+1} K_{n+1}^T}{\sqrt{d_k}}\right) V_{n+1}$

$$
\boxed{Y_{n+1} = \begin{bmatrix} Y_n \\ y_{n+1} \end{bmatrix}}
$$

> [!success] 结论
> 前 $n$ 行的输出完全不变，只需要计算新的 $y_{n+1}$。这就是 KV Cache 的数学基础。

## 缓存什么？

计算 $y_{n+1}$ 时需要所有历史 token 的 Key 和 Value：

$$
y_{n+1} = \text{softmax}\left(\frac{q_{n+1} K_{n+1}^T}{\sqrt{d_k}}\right) V_{n+1}
$$

因此需要缓存 $K_n$ 和 $V_n$，每步追加 $k_{n+1}$ 和 $v_{n+1}$。

> [!tip] 为什么不缓存 Q？
> 因为 $y_{n+1}$ 只用到当前 token 的 $q_{n+1}$，不需要历史的 Query。

## Multi-Head Attention

每个 head 独立计算 attention，各自维护自己的 KV Cache，最终拼接后乘输出矩阵 $W^O$。由于每个 head 都是增量的，整体也是增量的。

> [!info]- Cross-Attention 的情况（仅 encoder-decoder 架构）
> 在 encoder-decoder 架构中，$K$ 和 $V$ 来自 encoder，在整个解码过程中**固定不变**。只有 $Q$ 来自 decoder。因此：
>
> - $K$ 和 $V$ 只需计算一次，之后直接复用
> - 每步计算 $y_{n+1}$ 是 $O(m)$（$m$ 为 encoder 输入长度），$m$ 固定则等价于 $O(1)$

---

[^1]: [Transformer Autoregressive Inference Optimization (by Lei Mao)](https://leimao.github.io/article/Transformer-Autoregressive-Inference-Optimization/)
