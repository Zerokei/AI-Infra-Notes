---
aliases: [多模态模型]
created: 2026-05-06
updated: 2026-05-06
---

# Multimodal Models

多模态模型（multimodal models）指能同时处理多种信息形式（**modality**——文本、图像、音频、视频）的模型。和纯文本 LLM 的根本差别是：要把"长得不一样"的输入（文本 token、图像 patch、音频频段）映射到同一个表示空间，模型才能"同时听懂"它们[^1]。

本页是多模态领域的**导航 + 概览页**——梳理子方向、列举主流商业化产品、给出 ML infra 视角的关键差异。**不深入任何具体技术**，深入内容请跳到对应 atomic 页。

## 概念地图：多模态有哪些子方向

按"模型吃哪些 modality + 输出什么"，多模态家族分这几类：

| 缩写 | 全称 | 输入 → 输出 | 代表 |
|---|---|---|---|
| **VLM** | Vision-Language Model | 图像 + 文本 → 文本 | [[LLaVA]]、Qwen-VL、[[Kimi-VL]] |
| **ALM**（也叫 SLM） | Audio/Speech-Language Model | 音频 + 文本 → 文本 | Qwen-Audio、Audio Flamingo |
| **Video-LLM** | Video Language Model | 视频 + 文本 → 文本 | Video-LLaMA、VideoLLaVA |
| **VLA** | Vision-Language-Action | 图像 + 文本 → 动作 token | π0、GR00T、RT-2（机器人 / 具身） |
| **Omni / Any-to-Any** | （无固定缩写） | 任意 modality → 任意 modality | [[GPT-4o]]、[[Qwen2.5-Omni]] |

另外有一类**单向生成专家**——专做 X→Y 的生成，不以"理解"为主：T2I（文生图，如 Stable Diffusion / FLUX）、T2V（文生视频，如 Sora）、TTS（文本转语音，如 ElevenLabs）、I2I（图像编辑，如 [[Qwen-Image-Edit]]）。

更高层有几个常见伞概念：

- **MLLM**（Multimodal LLM）—— 学术综述最常用，泛指 ≥2 种 modality 的 LLM
- **LMM**（Large Multimodal Model）—— OpenAI / Google 偏好的叫法，含义同 MLLM
- **UFM**（Unified Multimodal Foundation Model）—— 更窄子集，特指**既能理解又能生成**的模型，详见 [[UFM]]

> [!note] 本页主聚焦
> 后续以 **VLM 主线 + UFM** 为主（也是当前商业化最深的两条线）。VLA 在"周边方向"节简提；ALM / Video-LLM / 单向 T2V / TTS 不在本页深入。

## 演进里程碑（2021–2026）

| 年-月 | 模型 | 突破 |
|---|---|---|
| 2021 | [[CLIP]] | dual-encoder 对比预训练，首次大规模图文表示对齐[^2] |
| 2023 | [[LLaVA]] | "vision encoder + connector + LLM" 三段式，开源易复制[^3] |
| 2024-05 | [[Chameleon]] | 首个开源 from-scratch early-fusion 模型，一个网络包圆[^4] |
| 2024-10 | [[GPT-4o]] | 首个产业级"全模态"，文本+音频+图像+视频 → 文本+音频+图像[^7] |
| 2025-03 | [[Qwen2.5-Omni]] | 开源 any-to-any，Thinker-Talker + TMRoPE[^8] |
| 2025-04 | [[Llama 4]] | 开源 native multimodal MoE[^9] |
| 2025-04 | [[Kimi-VL]] | 开源 MoE VLM，Thinking 变体加 long CoT[^10] |

三条主线：(1) **看懂图**（CLIP / LLaVA 路线，主流 VLM）→ 详见 [[VLM Architecture]]；(2) **看懂 + 能生成**（Chameleon / GPT-4o 路线，UFM）→ 详见 [[UFM]]；(3) **效率优化**（MoE、long context、原生分辨率），叠加在前两条主线上。

## 主流 VLM 是怎么工作的

90% 你日常用的多模态产品（GPT-4o、Gemini、豆包视觉、Qwen-VL）都基于 **vision encoder + connector + LLM** 三段式：图像被 [[ViT]] 切成 patch 编码成 embedding 序列 → connector（一层小 MLP）做维度对齐 → 与文本 token 拼接送入 LLM → 输出文本。

→ 详细工作流（含 LLaVA 实例）+ visual instruction tuning + cross-attention 注入变体 见 [[VLM Architecture]]。

## 既能理解又能生成：UFM

普通 VLM 只能"看"，不能"画"。当一个模型同时具备"理解 + 生成"两种能力，就叫 UFM。代表：[[GPT-4o]]、[[Chameleon]]、[[Llama 4]]、[[Qwen-Image-Edit]]。

NJU + CAS Auto + PKU 综述按"理解和生成怎么连起来"分三种范式[^11]：

- **调用外部 API**：LLM 当指挥官调外部生成模型 API（早期方案，已淘汰）
- **LLM + 外挂生成器**（模块化联合）：LLM 输出 prompt 或中间向量驱动独立扩散模型——[[Qwen-Image-Edit]] 是开源代表
- **一个模型全包**（端到端统一）：单一架构内统一处理理解 + 生成（GPT-4o、Chameleon），主流前沿

→ 三种范式的详细机制（AR / Diffusion / AR-Diffusion 混合）+ 各派代表模型 + Diffusion text encoder 选型 见 [[UFM]]。

## 图文商业化代表（infra 视角）

按"产品做什么"分三类，中外各举 3-4 个代表。表格列：模型 | 公司 | 部署 | 关键 infra 标签。详细架构 / 推理 pattern / 规模 / 延迟请跳每个产品的 atomic 页。

### 视觉理解（VLM）

| 模型 | 公司 | 部署 | 标签 |
|---|---|---|---|
| [[GPT-4o]] | OpenAI | 闭源 API | native multimodal AR；语音 232-320 ms 端到端[^7] |
| Claude 4 Vision | Anthropic | 闭源 API | 长文档 / 多图理解强 |
| Gemini 2.5 | Google | 闭源 API | native multimodal；1M+ 上下文 |
| Qwen2.5-VL | 阿里 | 开源 (3B / 7B / 32B / 72B) | 原生分辨率 |
| [[Kimi-VL]] | Moonshot | 部分开源 | MoE (16B 总 / 2.8B 激活) + MoonViT 原生分辨率[^10] |
| 豆包视觉 (Doubao-VL) | 字节 | 闭源 API (火山引擎) | 国内 API 价格 + 并发优势 |

### 图像生成（T2I, Text-to-Image）

| 模型 | 公司 | 部署 | 标签 |
|---|---|---|---|
| Midjourney v7 | Midjourney | 闭源订阅 (Discord / Web) | 美学质量标杆；无 API |
| DALL-E 3 | OpenAI | 闭源 (集成 ChatGPT) | 强 prompt following |
| FLUX.1 | Black Forest Labs | dev / schnell 开源；pro 闭源 | rectified flow + DiT；schnell 4 步出图 |
| Stable Diffusion 3.5 | Stability AI | 开源可商用 | UNet → MMDiT 架构演进代表 |
| Qwen-Image | 阿里 | 部分开源 | 20B MMDiT；[[Qwen-Image-Edit]] 基座[^12] |
| 即梦 (Dreamina) | 字节 | 闭源 API + Web | 国内消费级 + 商品图场景 |
| 可图 (Kolors) | 快手 | 闭源 API + Web | 与可灵视频同生态 |

### 图像编辑

| 模型 | 公司 | 部署 | 标签 |
|---|---|---|---|
| Adobe Firefly Generative Fill | Adobe | 闭源 (集成 PS / Lightroom) | 自研 diffusion；训练数据无版权风险 |
| Gemini Nano Banana (2.5) | Google | 闭源 (Gemini app) | UFM native 生图，多轮编辑流畅[^6] |
| GPT-4o Image Generation | OpenAI | 闭源 (集成 ChatGPT) | UFM native AR 生图[^7] |
| [[Qwen-Image-Edit]] | 阿里 | 开源 | 模块化联合 UFM：frozen Qwen2.5-VL + VAE → MMDiT[^12] |

## Infra 视角的关键差异（速览）

不同子方向的推理 profile 完全不同，部署时需要分别考虑：

- **VLM**：仍是 transformer prefill+decode，但视觉 token 显著放大 prefill（单图 500-4000 token，KV cache 翻倍）
- **T2I**：固定 N 步去噪（典型 20-50 步），无 KV cache，compute-bound 而非 memory-bound，单请求延迟数秒到数十秒级
- **UFM AR 生图**（GPT-4o image gen）：继承 LLM autoregressive 推理，但 1k+ 视觉 token 串行生成 → 单图延迟数十秒级
- **UFM 模块化联合**（Qwen-Image-Edit）："VLM 推理 + diffusion 推理"两段 pipeline，部署比纯 diffusion 复杂

→ 详细差异 + batching / 步数压缩 / 调度策略 见 [[Multimodal Inference Considerations]]。

## 周边方向

HF VLM 2025 综述梳理的几个 2024–2026 能力方向[^5]：

- **Reasoning VLM**：long CoT + RL（[[Kimi-VL]] Thinking、QVQ-72B）
- **Small VLM**：消费 GPU 可跑（SmolVLM、Gemma 3-4B、Qwen 2.5-VL-3B）
- **MoE decoder**：VLM 的 LLM 部分换 MoE（[[Kimi-VL]]、[[Llama 4]]、DeepSeek-VL2）
- **VLA**（Vision-Language-Action）：输出 action token 控制机器人（π0、GR00T N1）
- **Multimodal RAG**：视觉 embedding 直接做检索（ColPali、ColQwen）

## 延伸阅读

- Sebastian Raschka, *Understanding Multimodal LLMs* —— [[Sources/Clippings/Understanding Multimodal LLMs|clipping]]
- HF, *Vision Language Models (Better, faster, stronger)* —— [[Sources/Clippings/Vision Language Models (Better, faster, stronger)|clipping]]
- NJU + CAS + PKU, *统一多模态理解与生成综述（83 页）* —— [[Sources/Clippings/统一多模态理解与生成综述：83页长文梳理进展和挑战|clipping]]
- Andrej Karpathy, *2025 LLM Year in Review* —— [[Sources/Clippings/2025 LLM Year in Review|clipping]]
- 关键 paper：[[Sources/Papers/2103.00020v1.pdf|CLIP]] / [[Sources/Papers/2304.08485v2.pdf|LLaVA]] / [[Sources/Papers/2405.09818v2.pdf|Chameleon]]

[^1]: Sebastian Raschka (2024-11). [[Sources/Clippings/Understanding Multimodal LLMs]]
[^2]: Radford et al. (2021). *Learning Transferable Visual Models From Natural Language Supervision*. [[Sources/Papers/2103.00020v1.pdf]]
[^3]: Liu et al. (2023). *Visual Instruction Tuning*. [[Sources/Papers/2304.08485v2.pdf]]
[^4]: Chameleon Team, Meta (2024). *Chameleon: Mixed-Modal Early-Fusion Foundation Models*. [[Sources/Papers/2405.09818v2.pdf]]
[^5]: merve et al. (2025-05). *Vision Language Models (Better, faster, stronger)*. [[Sources/Clippings/Vision Language Models (Better, faster, stronger)]]
[^6]: Andrej Karpathy (2025-12). *2025 LLM Year in Review*. [[Sources/Clippings/2025 LLM Year in Review]]
[^7]: OpenAI (2024-10). *GPT-4o System Card*. [[Sources/Papers/2410.21276v1.pdf]]
[^8]: Xu et al., Qwen Team, Alibaba (2025-03). *Qwen2.5-Omni Technical Report*. [[Sources/Papers/2503.20215v1.pdf]]
[^9]: Meta AI (2025-04-05). *The Llama 4 herd: The beginning of a new era of natively multimodal AI innovation*. [[Sources/Clippings/The Llama 4 herd The beginning of a new era of natively multimodal AI innovation]]
[^10]: Kimi Team, Moonshot AI (2025-04). *Kimi-VL Technical Report*. [[Sources/Papers/2504.07491v3.pdf]]
[^11]: 让你更懂AI的 (2025-12-08), 介绍 NJU + CAS Auto + PKU 联合综述 *A Survey of Unified Multimodal Understanding and Generation: Advances and Challenges*（参考 750+ paper，83 页）. [[Sources/Clippings/统一多模态理解与生成综述：83页长文梳理进展和挑战]]
[^12]: Qwen Team, Alibaba (2025-08). *Qwen-Image Technical Report*. [[Sources/Papers/2508.02324v1.pdf]]
