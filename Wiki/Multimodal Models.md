---
aliases: [多模态模型]
created: 2026-05-06
updated: 2026-05-06
---

# Multimodal Models

多模态模型（multimodal models）指能同时处理两种以上 modality（文本、图像、音频、视频等）的模型[^1]。本页按时间梳理 2021 年以来的演进里程碑、描述标准 vision encoder + LLM 理解架构、再按 NJU + CAS Auto + PKU 综述的三大范式介绍**统一多模态基础模型 (UFM)**——既能理解又能生成的前沿方向[^11]。

## Background

多模态的根本难点是不同 modality 的离散化方式不同——文本走 BPE/SentencePiece tokenizer，图像走 patch 切分，音频走 spectrogram 或 codec token——表示空间天然不一致[^1]。

演进脉络上，从 2021 年至今的关键里程碑：

| 年-月 | 模型 | 突破 |
|---|---|---|
| 2021 | [[CLIP]] | dual-encoder + 4 亿对 image-text 对比预训练，首次大规模图文表示对齐[^2] |
| 2023 | [[LLaVA]] | visual instruction tuning，预训练 vision encoder + projector + LLM，开源 VLM 民主化[^3] |
| 2024-05 | [[Chameleon]] | 首个主流开源 from-scratch early-fusion 模型，所有 modality 同 token 流训练[^4] |
| 2024-10 | [[GPT-4o]] | 首个产业级 omni 模型：text/audio/image/video 输入 → text/audio/image 输出[^7] |
| 2025-03 | [[Qwen2.5-Omni]] | 开源 any-to-any，Thinker-Talker + TMRoPE 解决音视频时序对齐[^8] |
| 2025-04 | [[Llama 4]] | 开源 native multimodal MoE，early fusion + 10M 上下文[^9] |
| 2025-04 | [[Kimi-VL]] | 开源 MoE VLM（16B 总 / 2.8B active），MoonViT，Thinking 变体加 long CoT[^10] |

## Mechanism: vision encoder + LLM（理解侧）

LLaVA 之后开源 VLM 的标准架构是 "vision encoder + projector + LLM"[^1]：

> 图像 → vision encoder → projector → 与文本 token 拼接 → LLM decoder → 文本

每段的常见选型[^1]：

- **Vision encoder**：通常是预训练的 [[ViT]]，把图像切成 patch（典型 16×16 像素），编码成 embedding 序列。流行选择是 CLIP 的视觉编码器或 SigLIP。
- **Projector / connector / adapter**：linear 层或小 MLP，把 vision encoder 输出的 embedding 投到 LLM 的 embedding 维度。
- **LLM**：复用预训练 LLM。训练时通常先冻结 LLM 只训 projector；instruction-tuning 阶段解冻。LLaVA 的关键贡献 **visual instruction tuning** 就是用 GPT-4 合成图文指令跟随数据让 LLM 学会"看图回答指令"[^3]。

> [!note] 视觉 token 的接入位置有两种变体
> 多数 VLM（LLaVA、Molmo、Qwen2-VL 等）把视觉 token 拼在文本 token 前作为输入序列。少数例外（Llama 3.2 11B/90B、Flamingo）把视觉 embedding 通过 cross-attention 注入 LLM 内部某些层——这样视觉信息不占 KV cache、LLM 主体可冻结作为 text-only drop-in，代价是 LLM 架构要改[^1]。

## UFM: 统一多模态基础模型

上面的标准架构让模型能"看懂"图像（输出文本），但生成图像 / 音频 / 视频是另一个能力。当模型同时具备"理解 + 生成"两种能力时，称为**统一多模态基础模型 (UFM)**[^11]。NJU + CAS Auto + PKU 联合综述（参考 750+ paper）按"理解与生成模块的耦合程度"把 UFM 分三大范式[^11]：

### 1. 外部服务集成

LLM 作"指挥官"调用外部专家模型 API。代表：Visual ChatGPT、HuggingGPT、AudioGPT、SwitchGPT[^11]。

实现简单、模块化便于扩展，但多次调用累积误差、依赖外部模型质量。综述定性为**早期权宜方案，在后续发展中逐渐淡化**[^11]。

### 2. 模块化联合

LLM 作"理解引擎"+ 软连接到独立生成模块（通常是预训练扩散模型）。综述细分为两类[^11]：

- **Prompt-mediated**：LLM 生成自然语言提示词驱动外部生成器。代表：Divter、TIGER、Mini-Gemini、ModaVerse、LLMBind、Spider。
- **Representation-mediated**：LLM 输出中间表征（连续或离散）作为生成模块的条件。代表：GILL（首次特征映射对齐 LLM 语言空间与图像生成模型）、Emu 系列（统一自回归处理交错文本-图像）、SEED/LaVIT（专用视觉分词器）、DreamLLM（Dream Queries 轻量接口）、NExT-GPT/AnyGPT（任意模态间生成）、[[Qwen-Image-Edit]]（frozen Qwen2.5-VL 出语义 + VAE 出外观 → MMDiT diffusion 融合生成；专门做图像编辑）[^12]。

优点是能即插即用现成扩散模型（如 Stable Diffusion）、训练成本低。缺点：提示词中介受自然语言表达力限制（难以精确传递空间布局、动作序列等细粒度信息）；表征中介虽生成质量更高但需额外对齐机制，且生成能力仍受限于外部模块上限[^11]。

### 3. 端到端统一

单一模型架构内统一处理多模态输入输出。当前**最主流和最具挑战性**的方向[^11]，按建模方法细分：

**自回归（AR）**——把不同模态编码为词元序列做 next-token prediction[^11]。**生成图像的具体路径**：vision tokenizer（VQ-VAE / VQ-GAN 类）把图像离散化为视觉 token（一张 256×256 图通常 1k+ token），加进 LLM 词表后由 LLM 自回归预测视觉 token 序列，最后 VQ decoder 还原像素。

代表：

- **早期融合**：[[Chameleon]]（early-fusion token-based + query-key 归一化提升训练稳定性）、MoMa（modality-aware MoE 提升计算效率）
- **离散输入**：Emu3（统一分词联合建模文本/图像/视频，仅 next-token 即完成多任务）、LWM（RingAttention 扩展到 1M 上下文）
- **混合输入**：Janus / Janus-Pro（双编码器，高层语义用于理解、低层细节用于生成，缓解任务冲突）
- **分词器创新**：VILA-U（重建损失 + 对比损失整合）、TokLIP、DDT-LLaMA
- **下一尺度预测 / 扩散头**：VARGPT、MMAR、LatentLM、UniFluid（轻量级扩散/流头避免 VQ 信息损失）

[[GPT-4o]][^7]、[[Llama 4]][^9]、[[Qwen2.5-Omni]][^8] 都属此类——其中 Llama 4 在 early fusion 上加 MoE（Scout 17B active / 109B 总；Maverick 17B active / 400B 总），Qwen2.5-Omni 用 Thinker-Talker 双模块处理音频流。

> [!warning] 纯 AR 生图的固有限制
> 综述明确点出四个问题[^11]：(1) **速度慢**——一张高分辨率图常 1k+ token 逐个生成；(2) **保真度受 VQ tokenizer 重建质量限制**；(3) **不符合图像内在 2D 结构**（强加 1D 因果序列）；(4) **误差累积**。这正是 AR-Diffusion 混合范式兴起的动机之一。

**扩散（Diffusion）**——扩散模型扩展到统一建模[^11]。**关键 nuance**：纯 diffusion 模型自身**不"理解"文字**——文本理解外包给独立 text encoder，编码出的 embedding 在每步去噪时通过 cross-attention 作为条件输入 diffusion 主体。Text encoder 的选择决定语言理解上限：

| Text encoder | 理解能力 | 代表 |
|---|---|---|
| CLIP text encoder | 弱 | Stable Diffusion 1.x/2.x、FLUX 早期 |
| T5 | 中 | Imagen、PixArt |
| LLM / VLM | 强 | [[Qwen-Image-Edit]]（用 frozen Qwen2.5-VL[^12]）、Stable Diffusion 3 |

按原理细分：

- **连续扩散**：Versatile Diffusion（双向文本-图像生成）、UniDiffuser（统一边缘/条件/联合分布）、CoDi（任意模态间生成）、C3Net
- **离散扩散**：UniD3（首次离散扩散统一建模）、D-DiT（双分支：图像连续 + 文本离散）、UniDisc（全离散，在生成与判别上超越 AR）
- **矫正流**：OmniFlow（模块化设计，每个流可独立预训练）

生成质量高、细节丰富，但推理速度慢、理解能力相对较弱。

> [!note] Diffusion 自己学语言？
> 综述提到扩散语言模型（如 LLaDA）的进展[^11]——如果成熟，未来可能出现 diffusion 既能理解又能生成的统一架构，不再依赖外部 text encoder。但目前还在早期。

**自回归-扩散混合（AR-Diffusion Hybrid）**——单一架构同时学 AR + diffusion，兼顾两者优势[^11]。**动机**：AR 强在 sequential / discrete（文本），diffusion 强在 continuous / spatial（图像）；hybrid 让一个模型用各自擅长的方式处理对应模态，规避两者各自的短板。代表：

- **连续扩散**：Transfusion（单 Transformer，AR 文本损失 + 图像扩散损失，双向注意力显著优于因果注意力）、MonoFormer（用预训练 LLM 初始化）
- **离散扩散**：Show-o（next-token + masked-token 预测 + Omni-Attention）、DoraCycle、UniCTokens
- **矫正流**：JanusFlow（基于 Janus 解耦编码器 + 矫正流）、BAGEL、Mogao
- **架构优化**：LMFusion、BAGEL、Mogao 用 modality-aware MoE 缓解任务干扰；X-Fusion 把视觉专家注入冻结 LLM 无需大规模重训练

综述认为混合建模**生成质量显著优于纯 AR，避免了模块化联合的信息传递瓶颈**[^11]。但噪声注入可能损害理解性能，参数共享带来训练冲突。

**其他**——编码器-解码器（OFA、Unified-IO 系列）、状态空间模型（OmniMamba 基于 Mamba-2，避免 Transformer 二次复杂度）、图结构（GraphGPT-o，用 multimodal attribute graph 捕捉跨模态实体关系）[^11]。

> [!note] UFM 的演化路径
> 综述把 UFM 的演化划分为三阶段[^11]：**Specific**（各自为战，理解与生成模型分立）→ **Combine**（功能组合，能力集成实现复杂任务）→ **Emergent**（复杂交错推理的未来愿景，"通过生成图像来思考"）。当前主流前沿（GPT-4o、Llama 4、Qwen2.5-Omni、Chameleon、Janus）处于 Combine 阶段。

## 周边发展方向（HF 综述视角）

UFM 之外，HF VLM 2025 综述梳理了 2024–2026 多模态模型的几个能力方向[^5]：

- **Reasoning VLM**：在视觉理解上加 long CoT + RL。代表：[[Kimi-VL]] Thinking 变体[^10]、QVQ-72B-preview
- **Small VLM**：256M–4B 能在消费 GPU 跑：SmolVLM、Gemma 3-4B、Qwen 2.5-VL-3B
- **MoE decoder**：VLM 的 LLM 部分用 MoE：[[Kimi-VL]]（16B 总 / 2.8B active）[^10]、DeepSeek-VL2、[[Llama 4]][^9]
- **VLA (Vision-Language-Action)**：VLM 输出端加 action token 控制机器人。代表：π0（Physical Intelligence）、GR00T N1（NVIDIA）
- **Multimodal RAG**：ColPali、ColQwen 直接 retrieve 文档截图 embedding，绕过 PDF parsing

## 附: Dual-encoder 表示对齐 (CLIP)

[[CLIP]] 不属于 UFM 范畴（无生成能力），但作为 vision encoder 训练范式的奠基至今广泛影响[^2]。视觉 encoder + 文本 encoder 各自把输入映射成向量，对比学习把匹配对的相似度最大化。CLIP 不能"说话"，但可以零样本做图像分类、跨模态检索；论文报告在 ImageNet 上零样本即可达到 ResNet-50 全监督的精度[^2]。

CLIP 训练出的 CLIP-ViT 后来被 LLaVA 等模型作为 vision encoder 复用[^1]。

## Trade-offs

> [!warning] 多模态对推理 infra 的特殊压力
> 一张图被 [[ViT]] 切 patch 后会产生数百到数千个 token，全部进入 LLM 的 [[KV Cache]]。这对 KV cache 容量、prefill 延迟、batching 策略都有非平凡影响——这也是 [[SGLang]] / [[vLLM]] 等推理引擎单独处理 VLM 的原因之一。

Karpathy 2025 年终回顾认为 Gemini Nano Banana 这种"图文联合生成"代表了 LLM 与人交互的新方向——人偏好视觉而非纯文本输出，未来 LLM 应用图像、infographic、白板等格式"说话"[^6]。这与 UFM 综述的 Emergent 阶段愿景吻合[^11]。

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
