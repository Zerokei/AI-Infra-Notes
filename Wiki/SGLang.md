---
aliases: []
created: 2026-04-15
updated: 2026-04-29
---

# SGLang

SGLang 是 LMSYS（Chatbot Arena 团队）开源的 LLM 推理服务系统，与 [[vLLM]] 同属一线[^1]。后端核心是 **RadixAttention**——用 radix tree 跨请求自动复用任意长度的 KV Cache 前缀，比 vLLM 的整块哈希前缀缓存更细粒度；前端附带一套 Python 嵌入式 DSL 用于描述多步 LLM 程序，让运行时自动识别并行与共享机会。在 agent、多轮对话、tree-of-thought 等前缀重用密集的场景下，吞吐可达基线系统的 5 倍。

## Background

### 复杂 LLM 程序的前缀复用机会

现代 LLM 应用很少是"一个 prompt 一个输出"——更多是包含大量 prompt 重用的**多步程序**。原文图 3 列举了四类典型模式：

![[Attachments/pics/sglang-sharing_wide.jpg|600]]
*图：四类典型 LLM 工作负载中的 KV Cache 共享机会。蓝色为可共享前缀，绿色为不可共享输入，黄色为模型输出。*[^1]

| 场景 | 可共享部分 | 来源 |
|---|---|---|
| Few-shot learning | 示例块 | 多个查询共用同一套 few-shot 示例 |
| Self-consistency | 原始 prompt | 一个 prompt 生成多个 sample 后投票 |
| Multi-turn chat | system prompt + 前几轮对话 | 每轮都重发整段历史 |
| Tree-of-thought | 共同前缀的推理步 | 搜索树上不同路径共享早期节点 |

这些场景都有大量 KV Cache 可跨请求复用，但传统系统（包括早期 vLLM）无法自动识别——要么需要手动配置缓存策略，要么直接放弃。SGLang 的回应是**前后端 co-design**：

| 层 | 组件 | 职责 |
|---|---|---|
| 后端 Runtime | RadixAttention | 用 radix tree 管理所有请求的 token 序列，新请求到来时自动匹配并复用前缀 KV Cache |
| 前端 Language | SGLang DSL | Python 嵌入式语言，描述多步 LLM 交互，后端识别并行机会并自动 batch |

后端 RadixAttention 单独使用就已明显强过整块哈希式 prefix cache；配合前端 DSL 才能把多路采样、few-shot 结构、多轮交互的并行与复用机会全部暴露给运行时。SGLang 的 acknowledgement 也明确写明**复用了部分 vLLM 代码**[^1]，本质上是 vLLM prefix caching 机制的精细化升级加上前端 DSL 补全。

## Mechanism

### RadixAttention 的数据结构

RadixAttention 的底层数据结构是 **radix tree（基数树，压缩前缀树）**：

- 标准 trie：每条边代表一个字符或 token；
- radix tree：把"只有一个子节点的链"合并成一条边，边的标签是任意长度的序列^[这也是 "radix" 名字的来源——相比单字符 trie 有更大的"基数"]。

节点数大幅减少，前缀匹配语义不变。在 SGLang 中：

- **key** = token 序列；
- **value** = 该 token 序列对应的 KV Cache（GPU 上的分页存储）。

所有已处理过的 token 前缀都存在这棵树里。新请求到来时：

1. 在 radix tree 上匹配最长的已有前缀；
2. 命中部分的 KV Cache 直接复用，无需重 prefill；
3. 剩下不匹配的 token 走常规 forward，生成的 KV 作为新节点插入树。

### Radix Tree 的九步演化示例

原文图 4 用九步展示 RadixAttention 的动态行为，场景为"两轮聊天 + 一批 few-shot 查询 + 一次 self-consistency 采样"：

![[Attachments/pics/sglang-radix_attn.jpg|600]]
*图：九步内 radix tree 的演化。绿色为新加节点，蓝色为本步被访问的缓存节点，红色为被驱逐的节点。*[^1]

| 步 | 事件 | 树的变化 |
|:---:|---|---|
| 1 | 初始空 | 只有 root |
| 2 | 第一个聊天请求（system prompt + "Hello" → "Hi"） | 作为单条边整体加入树 |
| 3 | 同会话第二轮 | 匹配第一轮为前缀，复用其 KV，新增追加边 |
| 4 | 新聊天会话进来 | 节点 `b` 被分裂——让两会话共用 system prompt，分叉点在用户消息处 |
| 5 | 第二会话继续 | 内存不够，驱逐节点 `c`（最久未用） |
| 6 | Few-shot 查询（与现有前缀都不同） | root 被分裂为两部分 |
| 7 | 另一批 few-shot 查询（共享同套示例） | 节点 `e` 被分裂以让新 batch 共享示例 |
| 8 | 第一会话新消息到 | 驱逐第二会话全部节点（最久未用） |
| 9 | 对节点 `j` 的问题做 self-consistency 采样 | 驱逐其他节点腾空间 |

**节点分裂（split）是核心操作**：遇到"部分共享"前缀时，把长边拆成"共享段 + 各自独立段"，让新请求挂到共享段上。所有操作完全自动——前端只管发完整 prompt，runtime 自行匹配、复用、缓存、驱逐。

### LRU 驱逐与 Cache-Aware 调度

两个策略共同提升命中率：

- **LRU 驱逐**：显存紧张时优先驱逐最近最少使用的叶子节点^[非叶子节点不能直接驱逐，必须先驱逐子节点；过程递归进行]；
- **Cache-Aware Scheduling**：调度器选下一批请求时优先选命中已有前缀的请求，让 radix tree 的存储更"发挥价值"。

RadixAttention 与 [[PagedAttention|continuous batching、paged attention]] 完全兼容——本质是在 vLLM 那套块化 KV Cache 存储之上，加一层 radix tree 管理元数据。

### SGLang DSL 的核心 primitive

> [!warning] DSL 在生产中已边缘化，了解即可
> SGLang 2024 年发布时把"前端 DSL + 后端 RadixAttention"宣传为双轨创新。两年后回看，DSL 的生态位被 OpenAI-compatible API + 通用编排框架（LangChain / LlamaIndex / DSPy）+ API 层结构化生成参数（`response_format`、`guided_json` 等）的组合压缩掉了。99% 的 SGLang 用户通过后端 API 访问，根本不碰 DSL；LMSYS 后续版本发布博客也几乎不再强调它。下面这节作为历史背景了解即可。

SGLang 前端是 Python 嵌入式 DSL，用来描述多步 LLM 交互、控制流与解码约束。五个关键 primitive：

| Primitive | 作用 |
|---|---|
| `fork` | 创建当前 prompt 的多个并行副本（对应 self-consistency、tree search） |
| `gen` | 调用 LLM 生成并把结果存入变量（非阻塞，支持并发） |
| `[variable]` | 读取已生成的变量 |
| `choices` | 约束生成——输出只能从给定选项里挑（结构化生成） |
| `run` | 执行一个 SGLang 函数并传参 |

![[Attachments/pics/sglang-llm_judge.jpg|600]]
*图：用 SGLang 实现"多维度论文评判"——LLM 从多个维度评分、合并判断、生成总结、给出最终打分，采用 branch-solve-merge prompting 技术[^2]。*[^1]

代码结构对应：`fork(...)` 在原 prompt 基础上复制出多条独立评估路径；每条路径用 `gen` 各自调用 LLM 评估一个维度（并行执行，runtime 自动识别机会）；`[variable]` 取回结果供下一步使用；`choices` 约束 final grade 只能取某集合的值（如 A/B/C/D）。所有 `gen` / `fork` 非阻塞，runtime 把独立的 LLM 调用自动 batch 起来并行跑——这就是前后端 co-design 的红利。

**Interpreter vs Compiler 模式**：SGLang 程序两种执行方式——interpreter 模式逐语句解释执行，简单直观；compiler 模式把程序 trace 成数据流图，交给图执行器，开启代码移动、指令选择、auto-tuning 等优化空间。语法受 [Guidance](https://github.com/guidance-ai/guidance) 启发，但 SGLang 额外引入程序内并行与 batching 机制^[Guidance 只做前端控制，没有专门后端；SGLang 前后端 co-design 是关键差异]。

## Variants & Comparisons

### 与 vLLM 的差异

| 维度 | vLLM | SGLang |
|---|---|---|
| Prefix Cache 机制 | 整 block 哈希匹配 | radix tree 任意长度前缀 |
| 命中粒度 | `block_size` 对齐的整块 | token 级别，部分命中也能复用 |
| 自动程度 | `enable_prefix_caching=True` 开关 | 默认开启，几乎零开销 |
| 前端 | 无，只提供生成 API | 内置 DSL，表达复杂 LLM 程序 |

> [!tip] RadixAttention 默认永久开启
> ablation 实验显示：即使没有 cache 命中，RadixAttention 也没有明显 overhead——所以 runtime 默认永久开启，不需要像 vLLM 那样靠 `enable_prefix_caching=True` 手动开。

### 与 PagedAttention 的关系

[[PagedAttention]] 解决"显存以什么粒度切分"，RadixAttention 解决"前缀以什么数据结构索引"——二者并非互斥。SGLang 在底层同样使用块化 KV Cache，只是把 vLLM 哈希表式的前缀缓存换成 radix tree 索引，前缀匹配从 block 边界对齐升级为 token 级任意长度。

## Trade-offs

### Benchmark 工作负载与硬件

SGLang 在 9 类典型 LLM 工作负载上做了测试[^1]：

| 类别 | 具体任务 |
|---|---|
| 标准评测 | MMLU（5-shot）、HellaSwag（20-shot） |
| Agent | ReAct Agent |
| 复杂推理 | Tree-of-Thought（GSM-8K） |
| 结构化输出 | JSON Decode |
| 对话 | Chat-short、Chat-long |
| RAG 管线 | DSPy RAG |
| 视觉语言 | LLaVA Bench |

硬件配置：Llama-7B 单卡 A10G、Mixtral-8x7B 8 卡 A10G + TP=8，FP16。基线：vLLM v0.2.5、Guidance v0.1.8、Hugging Face TGI v1.3.0。

### 吞吐最高 5 倍领先

![[Attachments/pics/sglang-llama_7b.jpg|600]]
*图：Llama-7B 在 9 类工作负载下的吞吐对比，SGLang 在几乎所有任务上领先。*[^1]

![[Attachments/pics/sglang-mixtral_8x7b.jpg|600]]
*图：Mixtral-8x7B（8 卡 TP）的吞吐对比，规模更大但 SGLang 同样领先。*[^1]

SGLang 最高吞吐达到其他系统的 5 倍。首 token 延迟（TTFT）也明显改善——prefix cache 命中时尤其显著。提升来自三个来源：

1. **RadixAttention 的自动 KV Cache 复用**（最大贡献）：多轮 / few-shot / self-consistency / tree-of-thought 场景下命中率极高；
2. **Interpreter 启用的程序内并行**：同一 SGLang 函数内多个独立 LLM 调用自动并行；
3. **前后端 co-design**：前端把"哪些是共享 prompt、哪些可并行"的信息显式交给后端。

### 业界采用

- [LLaVA 在线 demo](https://llava.hliu.cc/) 用 SGLang 做服务后端；
- [DSPy](https://github.com/stanfordnlp/dspy) 已集成 SGLang 作为可选后端；
- LMSYS Chatbot Arena 背后也用 SGLang（SGLang 团队即 LMSYS）。

## Related

- [[KV Cache]] —— RadixAttention 复用的就是这部分缓存
- [[PagedAttention]] —— 同领域显存管理思路，与 RadixAttention 互补
- [[vLLM]] —— 同代推理引擎，SGLang 在其代码基础上演化
- [[LLM Inference Optimization#KV Cache 的显存开销]] —— prefix cache 命中率为何重要的量化背景

[^1]: [[Sources/Clippings/Fast and Expressive LLM Inference with RadixAttention and SGLang]]
[^2]: Saha et al. (2023). [Branch-Solve-Merge Improves Large Language Model Evaluation and Generation](https://arxiv.org/abs/2310.15123)
