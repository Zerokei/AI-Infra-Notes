---
date-created: 2026-04-15 02:30
date-updated: 2026-04-14 21:07
tags:
  - 🤖
---

# SGLang

## 前置知识

- [[vLLM]] — 理解推理引擎的基本组件与 prefix caching
- [[PagedAttention]] — KV Cache 分块管理
- [[KV Cache]]

> [!abstract] SGLang = RadixAttention（后端）+ 结构化生成 DSL（前端）
> SGLang 是 LMSYS（Chatbot Arena 团队）出的开源 LLM 推理引擎，与 vLLM 并列。两大组件：**RadixAttention** 让 KV Cache 跨请求自动复用（radix tree 管理任意长度前缀，比 vLLM 的整块哈希更细粒度）；**前端 DSL** 用 Python 风格描述多步 LLM 交互。代价极小的情况下，在 agent / 多轮对话 / tree-of-thought 等复杂场景下吞吐可达 vLLM 的 5 倍 [^1]。

## 先建立直觉：SGLang 要解决什么问题

### 复杂 LLM 程序的 KV Cache 重用机会

现代 LLM 应用很少是 " 一个 prompt 一个输出 "——更多是**多步程序**，包含大量 prompt 重用。原文图 3 展示了四种典型模式：

![[sglang-sharing_wide.jpg]]
*蓝色 = 可共享部分、绿色 = 不可共享输入、黄色 = 模型输出。*

| 场景 | 可共享的部分 | 为什么 |
|---|---|---|
| **Few-shot learning** | 示例块 | 多个查询共用同一套 few-shot 示例 |
| **Self-consistency** | 原始 prompt | 一个 prompt 生成多个 sample（对答案投票） |
| **Multi-turn chat** | system prompt + 前几轮对话 | 每轮都重发整个历史 |
| **Tree-of-thought** | 共同前缀的推理步 | 搜索树上不同路径共享早期节点 |

**核心观察**：这些场景都有**大量 KV Cache 可跨请求复用**，但传统系统（包括早期 vLLM）**无法自动识别**——要么需手动配置缓存策略，要么直接放弃。

### SGLang 的双轨设计

SGLang 的做法：**后端 + 前端 co-design**，让复用自动化。

| 层 | 组件 | 做什么 |
|---|---|---|
| **后端 Runtime** | **RadixAttention** | 用 radix tree 管理所有请求的 token 序列；新请求到来时自动匹配并复用前缀 KV Cache |
| **前端 Language** | **SGLang DSL** | Python 嵌入式语言，描述多步 LLM 交互；后端识别并行机会并自动 batch |

后端 RadixAttention 单独使用（例如作为 vLLM 的替代推理后端）已经明显强过整块哈希 prefix cache；配合前端 DSL 才能发挥最大效率——**多路采样、few-shot 结构、多轮交互**的并行 / 复用机会都能被自动识别。

### 对比 vLLM

| 维度 | vLLM | SGLang |
|---|---|---|
| Prefix Cache 机制 | 整 block 哈希匹配 | radix tree 任意长度前缀 |
| 命中粒度 | `block_size` 对齐的整块 | token 级别，部分命中也能复用 |
| 自动程度 | `enable_prefix_caching=True` 开关 | 默认开启，几乎零开销 |
| 前端 | 无，只提供生成 API | 内置 DSL，表达复杂 LLM 程序 |

在 agent、多轮对话、tree-of-thought 等复杂场景下，**SGLang 吞吐可达 vLLM 的 5×**（下文 benchmark 有具体数字）。

> SGLang 的 acknowledgement 明确写了**复用了一部分 vLLM 代码**，并非完全重造——它是 vLLM prefix caching 机制的精细化升级 + 前端 DSL 补全。

## RadixAttention：自动 KV Cache 重用

### 什么是 Radix Tree

RadixAttention 的底层数据结构是 **radix tree（基数树，压缩前缀树）**：

- **标准 trie**：每条边代表**一个字符 / token**；
- **radix tree**：把 " 只有一个子节点的链 "**合并成一条边**，边的标签是**任意长度的序列**^[这也是 "radix" 名字的来源——相比单字符 trie 有更大的 " 基数 "]。

节点数大幅减少，前缀匹配语义不变。

在 SGLang 里：

- **key** = token 序列；
- **value** = 该 token 序列对应的 KV Cache（GPU 上的分页存储）。

所有已处理过的 token 前缀都存在这棵树里。新请求到来时：

1. 在 radix tree 上匹配**最长的**已有前缀；
2. 命中部分的 KV Cache **直接复用**（无需重 prefill）；
3. 剩下不匹配的 token 走常规 forward，生成的 KV **作为新节点插入树**。

### 工作机制：9 步演化示例

原文图 4 展示 RadixAttention 的动态行为——九步里 radix tree 的变化：

![[sglang-radix_attn.jpg]]
*绿色 = 新加节点、蓝色 = 本步被访问的缓存节点、红色 = 被驱逐的节点。*

九步场景：**两轮聊天 + 一批 few-shot 查询 + 一次 self-consistency 采样**。

| 步 | 事件 | 树的变化 |
|:---:|---|---|
| **1** | 初始空 | 只有 root |
| **2** | 第一个聊天请求（system prompt + "Hello" → "Hi"） | 作为单条边整体加入树 |
| **3** | 同会话第二轮 | 匹配第一轮为前缀，**复用**其 KV，新增追加边 |
| **4** | 新聊天会话进来 | 节点 `b` **被分裂**——让两会话共用 system prompt，分叉点在用户消息处 |
| **5** | 第二会话继续 | 内存不够，驱逐节点 `c`（最久未用） |
| **6** | Few-shot 查询（与现有前缀都不同） | root 被分裂为两部分 |
| **7** | 另一批 few-shot 查询（**共享同套示例**） | 节点 `e` 被**分裂**以让新 batch 共享示例 |
| **8** | 第一会话新消息到 | 驱逐第二会话全部节点（最久未用） |
| **9** | 对 node `j` 的问题做 self-consistency 采样 | 驱逐其他节点腾空间 |

**关键观察**：

- **节点分裂（split）是核心操作**：遇到 " 部分共享 " 前缀时，把长边拆成 " 共享段 + 各自独立段 "，让新请求挂到共享段上；
- 所有操作**自动**：前端只管发完整 prompt，runtime 自行匹配、复用、缓存、驱逐。

### LRU 驱逐 + Cache-Aware 调度

两个策略共同提升命中率：

- **LRU 驱逐**：显存紧张时优先驱逐**最近最少使用的叶子节点**^[非叶子节点不能直接驱逐，必须先驱逐子节点；整个过程递归进行]；
- **Cache-Aware Scheduling**：Scheduler 选下一批请求时**优先选命中已有前缀**的请求——让 radix tree 的存储更 " 发挥价值 "。

RadixAttention 与 **continuous batching**、**paged attention** 完全兼容——**就是在 vLLM 那套 KV Cache 分块存储之上，加了一层 radix tree 管理元数据**。

## 前端：SGLang DSL

> [!warning]+ DSL 现已边缘化，生产采用以后端为主
> SGLang 2024 年发布时把 " 前端 DSL + 后端 RadixAttention" 宣传为双轨创新。两年后回看，DSL 的生态位被 **OpenAI-compatible API + 通用编排框架（LangChain / LlamaIndex / DSPy）+ API 层结构化生成参数**（`response_format`、`guided_json` 等）的组合压缩掉了。**99% 的 SGLang 用户通过后端 API 访问**，根本不碰 DSL；LMSYS 后续版本发布博客也几乎不再强调它。
>
> **下面这节作为历史背景了解即可**——知道 DSL 存在、大致能干啥就够了；真碰到 agent / tree-of-thought 等前缀共享密集的复杂多步程序场景，再回头深入。DSL 的灵感来源 [Guidance](https://github.com/guidance-ai/guidance) 也走过同样被淘汰的路径。

SGLang 前端是 **Python 嵌入式 DSL**，用来描述**多步 LLM 交互 + 控制流 + 解码约束**。

### 核心 primitives

5 个关键 API：

| Primitive | 作用 |
|---|---|
| `fork` | 创建当前 prompt 的多个并行副本（对应 self-consistency、tree search） |
| `gen` | 调用 LLM 生成并把结果存入变量（**非阻塞**，支持并发） |
| `[variable]` | 读取已生成的变量 |
| `choices` | 约束生成——输出只能从给定选项里挑（结构化生成） |
| `run` | 执行一个 SGLang 函数并传参 |

### 示例：多维度论文评判

![[sglang-llm_judge.jpg]]
*用 SGLang 实现 " 多维度论文评判 "：LLM 从多个维度评分、合并判断、生成总结、给最终打分。采用 branch-solve-merge prompting 技术 [^4]。*

结合 primitives 看这段代码的结构：

- `fork(...)` 在原 prompt 基础上复制出多个独立的评估路径；
- 每条路径用 `gen` 各自调用 LLM 评估一个维度（**并行**执行，runtime 自动识别机会）；
- `[variable]` 取回结果供下一步使用；
- `choices` 约束 "final grade" 只能取某集合的值（如 A/B/C/D）。

所有 `gen` / `fork` 非阻塞，runtime 把独立的 LLM 调用**自动 batch 起来并行跑**——这就是前后端 co-design 的好处。

### Interpreter vs Compiler 模式

SGLang 程序两种执行方式：

- **Interpreter 模式**：逐语句解释执行，简单直观；
- **Compiler 模式**：把程序 trace 成**数据流图**，交给图执行器——开启更多优化空间（代码移动、指令选择、auto-tuning）。

语法**受 [Guidance](https://github.com/guidance-ai/guidance) 启发**，但 SGLang 额外引入程序内并行与 batching 机制^[Guidance 只做前端控制，没有专门后端；SGLang 前后端 co-design 是关键差异]。

## 性能与采用

### Benchmark 工作负载

SGLang 在 9 类典型 LLM 工作负载上做了测试 [^1]：

| 类别 | 具体任务 |
|---|---|
| **标准评测** | MMLU（5-shot）、HellaSwag（20-shot） |
| **Agent** | ReAct Agent |
| **复杂推理** | Tree-of-Thought（GSM-8K） |
| **结构化输出** | JSON Decode |
| **对话** | Chat-short、Chat-long |
| **RAG 管线** | DSPy RAG |
| **视觉语言** | LLaVA Bench |

硬件：Llama-7B 单卡 A10G、Mixtral-8x7B 8 卡 A10G + TP=8，FP16。基线：vLLM v0.2.5、Guidance v0.1.8、Hugging Face TGI v1.3.0。

### 结果：最高 5× 吞吐

![[sglang-llama_7b.jpg]]
*Llama-7B 吞吐对比：SGLang 在几乎所有工作负载上领先。*

![[sglang-mixtral_8x7b.jpg]]
*Mixtral-8x7B（8 卡 TP）吞吐对比：同样领先，规模更大。*

SGLang 最高吞吐达到其他系统的 **5×**。首 token 延迟（TTFT）也明显改善——prefix cache 命中时尤其显著。

### 提升来自三个来源

1. **RadixAttention 的自动 KV Cache 复用**（最大贡献）：多轮 / few-shot / self-consistency / tree-of-thought 场景下命中率极高；
2. **Interpreter 启用的程序内并行**：同一 SGLang 函数内多个独立 LLM 调用自动并行；
3. **前后端 co-design**：前端把 " 哪些是共享 prompt、哪些可并行 " 的信息显式交给后端。

> [!tip] RadixAttention 默认永久开启
> ablation 实验显示：**即使没有 cache 命中，RadixAttention 也没明显 overhead**——所以 runtime 默认永久开启，不需要像 vLLM 那样靠 `enable_prefix_caching=True` 手动开。

### 当前采用

- **[LLaVA 在线 demo](https://llava.hliu.cc/)** 用 SGLang 做服务后端；
- **[DSPy](https://github.com/stanfordnlp/dspy)** 已集成 SGLang 作为可选后端；
- **LMSYS Chatbot Arena** 背后也用 SGLang（SGLang 团队即 LMSYS）。

[^1]: [Fast and Expressive LLM Inference with RadixAttention and SGLang](https://lmsys.org/blog/2024-01-17-sglang/)
[^2]: [Efficiently Programming Large Language Models using SGLang (arXiv)](https://arxiv.org/abs/2312.07104)
[^3]: [SGLang GitHub](https://github.com/sgl-project/sglang/)
[^4]: [Branch-Solve-Merge Prompting (arXiv)](https://arxiv.org/abs/2310.15123)
