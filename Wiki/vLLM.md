---
date-created: 2026-04-14 20:30
date-updated: 2026-04-15 02:00
tags:
  - 🤖
---

# vLLM

## 前置知识

- [[PagedAttention]] — vLLM 最核心的技术，KV Cache 的分块管理
- [[KV Cache]] — 为什么需要缓存 K/V
- [[LLM Inference Optimization#Prefill vs Decode]] — 推理两阶段的特性差异

> [!abstract] vLLM 是 LLM 服务事实上的开源标准
> vLLM 是一个开源 LLM 推理引擎，底层以 PagedAttention 管理 KV Cache，外加 Continuous Batching、Prefix Caching、Tensor Parallelism 等一整套生产级工程特性。核心设计哲学：**把 KV Cache 当虚拟内存管、把请求队列当 OS 调度器管**[^1][^2]。

## 从最小实现理解：Nano-vLLM

Nano-vLLM 是 DeepSeek-V3/R1 论文作者之一用约 1200 行 Python 精炼出的 vLLM 最小实现，支持 prefix caching、tensor parallelism、CUDA graph 与 torch compile 优化，吞吐量基本持平完整版 vLLM——是理解推理引擎架构的最佳切入点[^1]。

### 先建立直觉：推理引擎到底在干什么？

#### 朴素方案的四个痛点

假设你有一张显存够大的 GPU 已加载 LLM 权重，一批用户请求涌进来。最朴素的做法是：

```python
for request in user_requests:
    output = model.generate(request.prompt)
    return_to_user(request, output)
```

逐个处理、一次一个。这种做法能运行，但会立即暴露四个问题：

1. **GPU 大多数时间在空等**：GPU 是为"一次算成百上千任务"设计的，每次只喂一个 sequence 是严重浪费；再加上每次调用都有固定的启动开销^[kernel launch + CPU↔GPU 数据搬运，典型 5–50 μs，与计算量无关]，单请求场景下利用率常常低于 1%。
2. **KV Cache 显存碎片**：为未知长度的请求预留 `max_tokens` 大小显存，实际仅用约 1/40，浪费 60–80%（详见 [[PagedAttention]]）。
3. **相同前缀被反复计算**：所有聊天请求共用同一个 system prompt（几百 token），每来一个请求都从头算一遍该前缀的 KV Cache，纯粹的重复计算。
4. **快请求等慢请求**：三个请求塞在一个 batch，总耗时取决于最慢者；短请求生成完了也无法立即返回。

推理引擎的存在就是**同时解决这四件事**。

#### 四大组件的直觉：餐厅类比

把推理引擎想象成一家正在满员运行的餐厅：用户请求就是来吃饭的客人，引擎要高效地服务他们。四件要做好的事对应四个角色：

| 餐厅角色 | 引擎组件 | 负责什么 |
|---|---|---|
| 门口**领位员** | **Scheduler** | 决定谁先入座、是否拼桌、满员时谁先让位 |
| **餐位管理员** | **Block Manager** | 管理桌子（block）与每桌的点单记录（KV Cache）：分配、释放、共用 |
| **厨房** | **Model Runner** | 按点单做菜（真正跑 GPU 前向） |
| **传菜员** | **Step Loop** | 串起全流程：点单送厨房、菜送回客人 |

几个关键细节换个视角就清楚：

- **Block（桌子）固定大小**——无论 2 人还是 8 人桌，都按 4 人桌（256 token）分；10 人大客就拼 3 张。这正是 [[PagedAttention]] 借鉴 OS 分页的思路。
- **Prefix caching（共享点单）**：两桌客人如果前半份菜单一模一样（相同 system prompt），领位员让他们共用同一份点单记录，省去厨房重做。
- **Continuous batching（随时拼桌）**：厨房上完一批菜时，领位员可以**立刻**把新客人加进下一批，不必等前一批吃完。

> [!info] 类比有边界：生成式 LLM 的每一步都要重排
> LLM 的"一道菜"= 一个 token，每个请求要"吃"很多道菜（逐 token 生成），不是一锅端。因此 Scheduler 每一步（每个 token）都要重新查看当前运行中有哪些客人、谁吃完了、谁需要等。

#### 端到端场景：3 个用户同时请求

同一时刻三个请求到达：

- **A**：聊天请求，system prompt 500 token + 新提问 50 token；
- **B**：翻译请求，500 token 文本（system prompt 与 A 相同）；
- **C**：代码补全，100 token（无 system prompt）。

追踪它们的旅程：

**t=0** — 三个请求分别经 Tokenizer 切成 Sequence（token ID 数组）。`add_request` 把它们塞入 **Waiting Queue**。

**t=1** — Scheduler 发现 Running Queue 空、Waiting 有 A/B/C。Block Manager 为每个请求计算所需的 block：

- A 的 system prompt 500 token → 2 块（未命中缓存，分配新 block `1`、`2`）；
- B 的 system prompt 500 token → 2 块（**命中 A 已注册的哈希**，指向 block `1`、`2`，`ref_count += 1`）；
- C 的 100 token → 1 块（新 block `3`）。

三个请求全部转入 **Running Queue**。Scheduler 判定本步做 **prefill**（处理 prompt）。

**t=2** — Model Runner 接到 `batch = [A, B, C]`、`action = prefill`，一次 prefill 并行处理三个请求的全部 prompt token，得到每个请求的 logits。Sampling 各采一个 token（比如"你"、"This"、"def"），追加到各自 Sequence 末尾。

**t=3 起** — 进入 **decode 循环**。每步 Model Runner 对 batch 做一次 decode（每 Sequence 生成 1 token）；CUDA Graph 按当前 batch size 匹配预录版本，低开销执行。

**t≈100** — C 最短，最早触发 EOS。Scheduler 把 C 移出 Running、归还其 block（block `3` 回空闲池）。Waiting 若有新请求 D 等待，可立即占用 C 让出的资源。

**t≈500** — B 完成。B 持有的 block 归还——但 block `1`、`2` 的 `ref_count` 还有 A 的引用，**不会真的释放**，仅 `ref_count -= 1`。只有 A 也完成后，block `1`、`2` 才回空闲池。

三件事同时发生：

1. **批处理提效率**：每步 prefill/decode 一次性算三个请求；
2. **前缀共享省显存**：A、B 共用 system prompt 的 block；
3. **动态调度降延迟**：C 完成后立刻释放资源，不等 A、B。

接下来分别看每个组件**具体如何实现**。

### 整体数据流：Prompt → Output

从 prompt 字符串到生成的 token 输出，系统的整体管线：

![[nano-vllm-01-01.png]]
*Nano-vLLM 整体数据流：`LLM.generate` 入口 → Tokenizer → Sequence → Scheduler → Model Runner → Sampling → 输出。*

三个关键抽象：

- **Sequence**：系统内部流动的基本单元，包含变长 token 数组与元信息（block 映射、采样参数、生成状态）。Scheduler、Block Manager、Model Runner 均以 Sequence 为操作对象。
- **Tokenizer**：模型特定（Qwen、Llama、DeepSeek 各自不同），将自然语言切分为 token。同一句话在不同模型下切出的 token 数可能不同。
- **Producer-Consumer 模式**：`add_request` 作为 Producer 将 Sequence 塞入队列；Step Loop 作为 Consumer 从队列取出批次。两者解耦，请求接收不阻塞 GPU 计算。

### 生产者—消费者模式与 Batching

#### 为什么必须解耦

若 `generate` 调用立即同步处理每个 prompt，GPU 每次仅能跑一个 Sequence 的前向。此时：

- 每次 kernel launch 的固定开销（CUDA kernel 初始化、CPU↔GPU 数据传输、同步）完全由单个请求承担；
- 单样本矩阵乘法的 GPU 利用率通常远低于 1%。

Producer-Consumer 模式使 Consumer 可先累积多个 Sequence，再以 batch 形式一次跑完 kernel。固定开销被摊薄到整个 batch，吞吐量可提升 1–2 个数量级。

#### Batching 的吞吐—延迟取舍

| Batch 大小 | 吞吐量 | 单请求延迟 |
|:---:|:---:|:---:|
| 大 | 高（GPU 利用率好） | 差（要等 batch 中最慢的完成） |
| 小 | 低（GPU 空转） | 好 |

> [!note] 生产环境下 batch size 不是固定参数，由 Scheduler 每步动态决定
> vLLM 等现代推理引擎采用 **Continuous Batching**：每一次 decode 迭代都可插入新请求或移除已完成请求。这是一种调度策略，而非仅仅"设置一个大 batch size"。下面 Scheduler 部分会看到这是如何做到的。

#### Scheduler 的第一层决策：Prefill 还是 Decode

两阶段在计算特性上完全不同（已见 [[LLM Inference Optimization#Prefill vs Decode]]）：

| 阶段 | 输入 | 计算 | 瓶颈 |
|---|---|---|---|
| Prefill | 整个 prompt，$N$ token | 一次性大矩阵乘法 | Compute-bound |
| Decode | 每步 1 个新 token | 小规模矩阵乘法 + KV Cache 读取 | Memory-bound |

Scheduler 每个 step 只能选择**"prefill 一批"**或**"decode 一批"**，不能混——两者的 kernel 形状差异过大，混在一起会严重降低 GPU 利用率。

### Scheduler：Waiting / Running 双队列

*类比对应：餐厅里的领位员。Scheduler 要做好三件事——**入座**（新请求入队）、**让位**（资源耗尽时抢占）、**归队**（请求完成后释放资源）。*

![[nano-vllm-01-02.png]]
*Scheduler 维护 Waiting 与 Running 两个队列；Block Manager 在两者间把关资源分配，资源耗尽时执行抢占。*

Scheduler 是引擎的大脑，维护两个核心队列：

- **Waiting Queue**：已提交但尚未开始处理的 Sequence。`add_request` 到来的新请求总是先进入此队列。
- **Running Queue**：正在处理中的 Sequence（处于 prefill 或 decode 阶段）。

状态转换规则：

- **进入系统**：`add_request` → Waiting 队列；
- **Waiting → Running**：Block Manager 为其分配到足够的空闲 block 时，转入 Running；
- **Running → Running**：每个 step 生成一个新 token；
- **Running → 完成**：生成 EOS 或达到 `max_tokens`，移除 Sequence、归还所有 block；
- **Running → Waiting 队首**：资源耗尽时抢占（见下）。

Scheduler 每个 step 从 Running 挑一批 Sequence，连同 action（prefill / decode）一起交给 Model Runner 执行。

#### 资源耗尽时的抢占（Preemption）

GPU 显存有限。当 Running 中某个 Sequence 要生成下一 token，却发现**没有空闲 block 存新的 KV**，Scheduler 对它执行**抢占**：

1. 从 Running 队列移出；
2. 放回 **Waiting 队列的队首**；
3. 释放它持有的资源给其他 Sequence；
4. 资源空出后，它将是 Waiting 中第一个被恢复的请求。

关键设计：放回队首而非队尾——保证被抢占请求的**公平恢复语义**，不会因抢占被挤到所有新请求之后。

> [!note]+ Nano-vLLM 只讲了抢占动作本身，生产 vLLM 有 Recompute / Swap 两种策略
> 原文未展开被抢占 Sequence 的 KV Cache 如何处理，实际生产有两种选择：
>
> | 策略 | 做法 | 优点 | 代价 |
> |---|---|---|---|
> | **Recompute**（重算） | 释放全部 block，恢复时从 prompt 重算 | 省显存 | 重复计算 |
> | **Swap**（换出） | block 内容 DMA 到 CPU 内存，恢复时再搬回 | 省算力 | 占用 CPU 内存 + PCIe 带宽 |
>
> **为什么 Swap 省算力**：Recompute 恢复时要重跑整个 prompt 的 prefill（GPU compute，重复劳动）；Swap 只做 CPU ↔ GPU 的 KV Cache 搬运（PCIe IO，不占 GPU 计算单元）——本质是**用闲置的 PCIe 带宽换紧俏的 GPU 算力**。这条"用 IO 换 compute"的思路在系统软件里很常见（OS 的 swap、数据库 buffer pool 溢写都是同路数）。
>
> **何时反而 Recompute 更好**：prompt 短（重算便宜）、PCIe 繁忙、CPU 内存吃紧放不下大量 KV Cache。
>
> vLLM 由 `swap_space` 参数控制，默认走 Swap；Nano-vLLM 为简化仅实现 Recompute。"vLLM 如何在 OOM 边缘保持可用"这类面试题答的就是这两条策略的权衡。

#### Sequence 完成的处理

生成到 EOS 或达到 `max_tokens` 时：

1. 从 Running 队列移除；
2. Block Manager 归还其所有 block 至空闲池；
3. 结果从 Step Loop 输出返给用户。

归还的 block 立刻可被 Waiting 队列中的下一个请求使用——这是 vLLM 在高并发下保持高显存利用率的关键。

### Block Manager：KV Cache 控制平面

*类比对应：餐位管理员。Block Manager 管两件事——**桌子的分配与释放**（block 生命周期），以及**哪些桌的点单可以共用**（prefix caching 的哈希匹配）。*

![[nano-vllm-01-03.png]]
*Block Manager 将 Sequence 切为定长 block；通过块表 + 哈希表管理分配与前缀复用。CPU 侧仅维护元数据（控制平面），真实 KV 数据在 GPU（数据平面）。*

Block Manager 是 vLLM 内存管理创新（即 [[PagedAttention]]）的实现载体。

#### 从 Sequence 到 Block

Sequence 是变长 token 数组——可能是 10 个 token，也可能是 10000 个。变长分配对 GPU 内存管理低效，Block Manager 的办法是将 Sequence 切成**定长 block**（Nano-vLLM 默认每块 256 token）。

例如一个 700 token 的 Sequence 占用 3 个 block：两个满块（各 256 token）+ 一个部分填充块（188 token，余 68 槽位未用）。**不同 Sequence 的 token 从不共享同一个 block**；但单个长 Sequence 可横跨多个 block。

> [!info] 块大小的工程权衡
> 原版 vLLM 默认 `block_size=16`，Nano-vLLM 为简化选择 256。块小则管理粒度细、内部碎片少、前缀共享匹配机会多；块大则查表次数少、元数据开销低。生产环境下会根据 KV 头数、上下文长度、并发规模折中选择。

#### 前缀缓存：哈希 + 引用计数

Block Manager 的亮点在**跨请求复用**：

1. 对每个填满的 block，根据其 token 内容（以及前序哈希链）计算一个**内容哈希**；
2. 维护一张全局 **hash → block_id** 映射表；
3. 新 Sequence 到来时，对其前缀 block 计算哈希并查表：
   - 命中：复用已有 block，引用计数 `+1`，**无需重新计算或分配**；
   - 未命中：从空闲池申请新 block，写入 KV，将哈希注册到映射表。

这一机制在所有请求共享同一 system prompt（最典型的 chat 应用）时收益巨大：系统提示对应的 KV Cache 在显存中仅存一份，所有请求共享读取。

> [!tip]+ Prefix Caching 的两种共享来源，底层是同一套 COW
> Block Manager 只通过 `ref_count + COW`（见 [[PagedAttention#Copy-on-Write 与前缀共享]]）处理所有共享，不区分来源。共享的来源有两种：
>
> - **请求内共享**：同一请求内部多条生成路径（parallel sampling / beam search）在分歧前共用 prompt block；
> - **请求间共享**：跨不同请求匹配相同前缀的 block（本节讨论的哈希命中，由 `enable_prefix_caching=True` 开启）。
>
> 生产 chat 服务的 cache hit 几乎全来自**请求间共享**（system prompt、多轮历史、few-shot 模板）；请求内共享在主流 `n=1` 的 API 下极少触发。SGLang 的 **RadixAttention** 把请求间共享升级为 radix tree，支持任意长度、任意位置的子序列匹配——这是它相对 vLLM 的招牌优势。

#### 控制平面 vs 数据平面

一个容易忽略但关键的设计：

| 组件 | 位置 | 内容 | 角色 |
|---|---|---|---|
| **Block Manager** | CPU 内存 | block 分配状态、引用计数、哈希映射 | **控制平面**（Control Plane） |
| **KV Cache 本体** | GPU 显存 | K、V 张量数据 | **数据平面**（Data Plane） |

这种分离让 Scheduler 的分配决策**不触碰 GPU 内存**，只更新 CPU 侧元数据——直到真正要执行 kernel 计算时才访问 GPU。可忽略的 CPU 开销换来了调度的高速。

此外，**block 归还时不清零 GPU 显存**——仅标记为空闲，下次分配时直接覆盖写入，避免不必要的内存操作。

### Model Runner：执行与张量并行

*类比对应：厨房。Model Runner 拿到点单后做菜，要处理两种节奏——**一次性大单**（prefill，处理整个 prompt）与**持续追加单品**（decode，每步新增 1 token）。*

![[nano-vllm-01-04.png]]
*Model Runner 收到 Scheduler 给出的 batch 后，依次完成 tensor 准备与 GPU 执行^[执行末尾还有从 logits 采样一个 token 的步骤——通用推理操作，非 vLLM 设计要点]，输出新 token。*

Model Runner 负责在 GPU 上实际执行模型前向。Step Loop 从 Scheduler 取到一批 Sequence 后，将其与 action（prefill / decode）一并交给 Model Runner。

#### Tensor Parallelism：Leader-Worker 通信

![[nano-vllm-01-05.png]]
*Tensor Parallelism 下的多 GPU 通信：Rank 0 作为 Leader 接收命令，经共享内存下发给 Workers，各 Worker 在本地 GPU 执行自己那份计算。*

当模型单 GPU 装不下时，Nano-vLLM 支持**张量并行**（Tensor Parallelism, TP）——将模型权重切分到多张 GPU 协作运行（例如 TP=8 即 8 张 GPU 共跑一份模型）。

通信采用 **Leader-Worker 模式**：

- **Rank 0（Leader）**：接收 Step Loop 发来的命令（方法名 + 参数），写入共享内存，同时执行自己那份计算。
- **Ranks 1…N−1（Workers）**：持续轮询共享内存缓冲区，发现新命令后读取参数、在本地 GPU 上执行对应操作。

每个 Worker 知道自己的 rank，因此能计算属于自己那部分的权重或激活。共享内存方式适合**单机多卡**场景，避免跨机网络的额外开销；跨机场景则退化到 NCCL over Ethernet / Infiniband。

#### 计算前的准备

调用模型前，Model Runner 根据 action 准备输入：

- **Prepare Prefill**：将多个变长 Sequence 批量化，并计算累积序列长度（cumulative sequence lengths）供 attention kernel 高效计算；
- **Prepare Decode**：将每个 Sequence 的单个新 token + 其位置 + KV Cache slot 映射打包为批量。

这一步同时将 CPU 侧的 token ID 数据转换为 GPU tensor——CPU→GPU 数据跨越点正在此处。

#### CUDA Graphs：降低 kernel launch 开销

Decode step 每个 Sequence 只处理 1 个新 token，真实计算量很小；此时 **kernel launch 的固定开销**（每次几微秒）相对于计算本身变得显著。**CUDA Graphs** 的做法是：

- 把一段连续的 GPU 操作"录制"成一张 graph；
- 之后以不同输入"回放"这张 graph，无需重新派发 kernel。

Nano-vLLM **预捕获**（pre-capture）若干常见 batch size 的 CUDA Graph（1, 2, 4, 8, 16, ..., 512）。decode 时按当前 batch size 匹配到最接近的预捕获版本执行，将 launch 开销压到最低。

### 模型计算内部：Part 2

Part 1 将模型计算视作黑盒，关注**引擎外壳**（Scheduler / Block Manager / Model Runner 的协作）。Part 2[^3] 打开黑盒——但本节**只保留与引擎相关的部分**：

- **模型三要素**（词表 / 权重 / 运行时代码）——说明引擎与模型的边界；
- **KV Cache 物理布局**（数据平面）——引擎如何在 GPU 显存中组织 KV；
- **Tensor Parallelism 计算层面**——引擎如何在多卡间切分计算。

> [!info] 模型架构本身（Embedding、MHA、MLP、Dense vs MoE、层数/头数权衡）属于 Transformer 基础，见 [[Transformer]]；本节略去。

#### 模型三要素：Vocabulary + Weights + Runtime Code

![[nano-vllm-02-01.png]]
*能运行推理的模型 = 词表 + 权重 + 运行时代码三者的组合。*

一个能实际运行推理的模型由三部分组成：

- **Vocabulary（词表）**：token 与 token ID 之间的静态映射。模型借此在人类可读文本与数值表示之间翻译。
- **Weights（权重）**：训练学到的参数，即模型积累的"知识"。7B 模型意味着 70 亿参数。
- **Runtime Code（运行时代码）**：定义如何用权重将输入转化为输出的逻辑，真正在 GPU 上执行的部分。

##### 为什么推理引擎要自己实现模型代码

模型提供方发布权重，有时也发布运行时代码，但运行时代码必须针对具体场景优化：**训练 vs 推理**、**不同 GPU 架构**、**不同精度格式**。在训练集群（A100 × N）上最优的实现，放到单张消费级 GPU 做推理未必最优。

因此 vLLM 等推理引擎为每个支持的模型架构实现自己的运行时代码。完整 vLLM 仓库覆盖 Qwen、Llama、DeepSeek、Mistral 等数十种架构；Nano-vLLM 为简化仅支持 Qwen，但模式通用。

#### KV Cache 的物理布局（数据平面）

![[nano-vllm-02-07.png]]
*KV Cache 在 GPU 显存中按 block × layer × (K/V) × token 的多维结构组织；一个逻辑 block 对应 24 层 × 2（K/V）= 48 个物理缓存区。*

Part 1 中 Block Manager 负责 KV Cache 的**控制平面**（CPU 侧元数据）；本节讨论**数据平面**——KV Cache 在 GPU 上实际如何存储。

##### 缓存什么

每个 token 在 attention 计算中产生两个向量：**K（key）** 与 **V（value）**，供后续 token 的 attention 计算使用。为避免每次 decode 都重算所有历史 token 的 K/V，将它们缓存起来（见 [[KV Cache]]）。

##### 物理结构

KV Cache 在 GPU 上是一个多维张量：

| 维度 | 含义 |
|---|---|
| **Block** | 对应 Block Manager 的逻辑 block（Nano-vLLM 为 256 token/块） |
| **Layer** | 24 层 decode layer 各有独立 KV Cache，因 attention 在每层独立计算 |
| **K / V** | 每层两份缓存，K 与 V 各一 |
| **Token** | 块内为每个 token 预留槽位存储其 K/V 向量 |

因此 Block Manager 中**一个逻辑 block 对应 GPU 上 $24 \times 2 = 48$ 个物理缓存区**。

##### Triton Kernels 抽象访存

Nano-vLLM 不直接通过 CUDA API 操作 GPU 内存，而是使用 **Triton kernels**——高层 GPU 程序，编译后产生高效 CUDA 代码。读写 KV Cache 的复杂性由 Triton 抽象掉。

#### Tensor Parallelism 的计算层面

Part 1 讲了 TP 的通信模式（Leader-Worker + 共享内存）；本节讲实际计算怎么切分。

##### Attention 阶段的并行

![[nano-vllm-02-08.png]]
*TP=2 下 Attention 阶段：两张 GPU 都拿到完整的 4096 维 hidden state，各自处理一半 head，最后 all-reduce 合并结果。*

以 TP=2 为例：

1. **两张 GPU 都收到完整的 hidden state**（4096 维）。不是按维度切分，**每张 GPU 都有完整输入**；
2. **每张 GPU 处理一半的 head**：GPU 0 处理 head 0–15，GPU 1 处理 head 16–31；
3. **每张 GPU 产出部分输出**；
4. **all-reduce 合并**：两张 GPU 交换各自的部分输出并求和，最终每张 GPU 都持有完整的 attention 输出。

关键洞察：**并行发生在 head 维度，而非 hidden state 维度**。

##### MLP 阶段的并行

![[nano-vllm-02-09.png]]
*MLP 阶段：两张 GPU 都拿到完整输入，中间维度 11008 被切分为 5504 + 5504，各自计算后 all-reduce。*

MLP 的切分类似：

1. 两张 GPU 都收到完整的 hidden state；
2. **中间维度被切分**：如 MLP 扩展到 11008 维，则每张 GPU 计算 5504 维；
3. 各产出部分输出；
4. all-reduce 合并。

##### 通信的代价

TP 并非免费：all-reduce 需要 GPU 间通信，引入延迟。因此 TP 最适合**单机多卡 + 高速互联**（如 NVLink）场景；跨机（网络互联）时通信延迟会主导。

收益：每张 GPU 仅需存储模型权重的一小部分（TP=2 存一半，TP=8 存 1/8），从而能运行单 GPU 无法容纳的大模型。

[^1]: [Understanding LLM Inference Engines: Inside Nano-vLLM (Part 1)](https://neutree.ai/blog/nano-vllm-part-1)
[^2]: [vLLM GitHub](https://github.com/vllm-project/vllm)
[^3]: [Understanding LLM Inference Engines: Inside Nano-vLLM (Part 2)](https://neutree.ai/blog/nano-vllm-part-2)
