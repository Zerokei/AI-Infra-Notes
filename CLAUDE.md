# AI-Infra Vault — Claude 工作规则

## 角色边界

- **用户职责**：sourcing（找资料、Web Clipper、读 paper、读代码）、提问、review、决定方向
- **Claude 职责**：消化原料、写 wiki 条目（overview + atomic）、维护互链、保持风格一致、维护 frontmatter
- **`Sources/` 是只读区**：Claude 不修改 `Sources/Clippings/` 和 `Sources/Papers/` 下的任何文件
- **`Wiki/` 是 Claude 主写区**：用户做 review，必要时小修

## 不确定即询问（强约束）

遇到不能 100% 确认的内容**不写到页里**——转而询问用户。零半信半疑产出。

## 文件命名（A.1）

- 文件名一律英文：`vLLM.md`、`PagedAttention.md`、`KV Cache.md`、`Memory Hierarchy.md`、`Inference Optimization.md`
- 专有名词保留社区原写法：`vLLM` 不写 `VLLM`，`TensorRT-LLM` 不改成 `Tensorrt Llm`
- 空格分词不用连字符：`KV Cache.md`，不是 `KV-Cache.md`
- 中文别名走 frontmatter `aliases`（有公认译法时加）

## Frontmatter（A.2）

每篇 Wiki 笔记顶部：
```yaml
---
aliases: [中文别名]
created: YYYY-MM-DD
updated: YYYY-MM-DD
---
```
不加 `tags`、不加 `status`。`updated` 由 Linter 自动维护；Claude 编辑后如 Linter 未触发，手动改为今日日期。

## H1 标题（A.3）

每篇 Wiki 笔记开头写一个与文件名一致的 H1，例如 `# vLLM`。

## 双链格式（B.4）

- 默认 `[[X]]`
- 中文句中读不顺时才用 `[[X|别名]]`
- 例：`[[PagedAttention]]` 优先；`类似于 [[Memory Hierarchy|内存层次结构]]` 才加别名

## Source 引用（B.5）

正文用 `[^n]` 标注，底部定义脚注：

- 内部 source（Clippings/Papers）走 `[[]]`
- 外部 URL 走 `[text](url)`
- paper 加作者-年-标题方便识别
- **PDF 等非 .md 附件必须带扩展名**，否则 Obsidian wikilink 失效。Clippings 是 .md 笔记，可省略扩展名

例：
```markdown
PagedAttention 优化了内存碎片[^1]，类似思路也出现在 SGLang 的 RadixCache[^2]。
NVIDIA 官方 blog 还提到 KV reuse[^3]。

[^1]: Kwon et al. (2023). [[Sources/Papers/2309.06180_PagedAttention.pdf]]
[^2]: [[Sources/Clippings/Fast and Expressive LLM Inference with RadixAttention and SGLang]]
[^3]: [Mastering LLM Techniques](https://developer.nvidia.com/blog/mastering-llm-techniques-inference-optimization/)
```

## 代码引用（B.6）

脚注 + GitHub permalink（含 commit hash + 行号）：

```markdown
vLLM 调度器采用 FCFS 策略[^1]。

[^1]: [vllm/core/scheduler.py:142-160](https://github.com/vllm-project/vllm/blob/<commit-sha>/vllm/core/scheduler.py#L142-L160)
```

## 双链粒度（B.7）

- 允许 `[[X]]` 和 `[[X#Heading]]`
- 不用 `[[X#^block-id]]`
- **强约束**：修改章节标题、拆分页面、删除章节时，必须同步更新所有 `[[X#Heading]]` 引用——执行前先 grep 全仓库搜索引用

## 标题层级（C.8）

最多 H3（## 和 ###）。超过这个深度就拆 atomic page。

## 数学公式（C.9）

- 行内 `$...$`、独立行 `$$...$$`
- 变量符号沿用所属领域经典论文（Transformer 论文的 $d_{model}$、$h$、$L$ 等）
- 给数值例子辅助理解（用户偏好）

## 代码块（C.10）

- 必带语言 tag
- 引自源码加文件 + commit 头部注释：
  ```python
  # vllm/core/scheduler.py @ a1b2c3d
  def _schedule(...): ...
  ```
- 自作示范不加头部注释

## 表格 vs 列表 vs callout（C.11）

- **表格**：≥2 项 × ≥2 可比属性的对比
- **列表**：纯枚举
- **callout**：和正文非线性关系的补充

callout 类型：
- `> [!note]` 通用补充、相关概念旁注
- `> [!tip]` 优化提示、最佳实践
- `> [!warning]` 易踩的坑、边界条件
- `> [!example]` 数值例子、情景化 case
- `> [!quote]` 直接引用 paper/blog
- `> [!question]` 开放问题

callout 与 `^[...]` 内联脚注的边界：
- 1-2 句旁注用 `^[...]`
- 超过用 callout
- 出处引用始终用 block 脚注 `[^n]`

callout 折叠：短/常读不加；长/可选 `+` 默认展开或 `-` 默认折叠。callout 标题：一句话概括块内主张。

## 图片嵌入（C.12）

```markdown
![[Attachments/pics/x.png|宽度]]
*图：caption 描述*[^1]

[^1]: 出处脚注
```

宽度按需，过宽限制 `|400` 或 `|500`。

## Transclusion（C.13）

不用 `![[X]]` 或 `![[X#Section]]`；统一用 `[[X]]` 链接。

## 页面骨架

### Overview 页（D.14）

Wikipedia Lead + 脚手架式正文：

```markdown
# vLLM

vLLM 是 UC Berkeley 出品的开源 LLM 推理服务系统[^1]，核心创新是 PagedAttention。
（无标题 Lead 段，2-3 句）

## Background
...

## Mechanism
...

## Variants & Comparisons
...

## Trade-offs
...

[^1]: ...
```

- Lead 段必有
- 其他章节按需出现，不适用就跳过（不写"N/A"）
- 结构性章节英文：`Background`、`Mechanism`、`Variants & Comparisons`、`Trade-offs`、`Implementation`、`History`
- 主题特有子章节中文：`### Block Table 实现细节`

### Atomic 页（D.15）

```markdown
# PagedAttention

PagedAttention 是 vLLM 提出的 KV cache 内存管理机制，借鉴操作系统虚拟内存的页表思路，
将 KV cache 拆为固定大小的 block，以避免内存碎片[^1]。

## Mechanism
...

## Related
- [[vLLM]] 使用、[[Block Manager]] 实现
- 另一种思路：[[RadixCache]]

[^1]: ...
```

- Lead（定义）必有
- Related 节必有
- 中间深度按需

## 不完整页面标记（D.16）

不加页内 WIP 标记。WIP 页在 `TODOs.md` 的"WIP 页"节里跟。

## 中英文混排空格（E.17）

中文与英文/数字之间加半角空格：`GPU 上的 HBM 带宽是 3 TB/s`。

## 专有名词大小写（E.18）

跟社区/论文原写法。`vLLM`、`SGLang`、`FlashAttention`、`TensorRT-LLM` 不变形。`GPU`/`HBM`/`CUDA`/`FCFS` 全大写。

## TODOs.md 结构

```markdown
## 当前
- 现在在读什么

## 建议补充
**前置**
- ...
**横向**
- ...
**纵向**
- ...

## WIP 页
- [[Wiki/X]] —— 哪节还没写
```

## Git commit 规范

- 完成有边界改动后立即 commit
- message 形式：`<type>: <change>`
  - `wiki: rewrite KV Cache → split into atomic pages`
  - `wiki: digest 2309.06180 → PagedAttention.md`
  - `sources: add clipping <title>`
  - `chore: update CLAUDE.md style rules`
- 不强制 commit cron；历史清洁度比覆盖率重要

## 反向链接更新规则

修改笔记（重命名、拆分、删除章节、删除整页）时：
1. 先 grep 全仓库搜索 `[[<原标题>]]` 和 `[[<原标题>#`
2. 列出所有引用位置
3. 给出综合考虑后的更新推荐方案
4. 执行更新后重新 grep 验证无残留断链
