---
aliases: [统一多模态基础模型, Unified Multimodal Foundation Model]
created: 2026-05-06
updated: 2026-05-06
---

# UFM

UFM（Unified Multimodal Foundation Model，统一多模态基础模型）指**同时具备"理解 + 生成"两种能力**的多模态模型——既能"看图回答"，又能"按指令画图 / 说话 / 生视频"。NJU + CAS Auto + PKU 联合综述按"理解和生成怎么连起来"分三种范式[^1]。

## 三范式

### 1. 调用外部 API（外部服务集成）

LLM 当"指挥官"，要画图调 DALL-E API、要做语音调 TTS API。代表：Visual ChatGPT、HuggingGPT、AudioGPT[^1]。

简单 + 模块化便于扩展，但累积误差、能力受外部模型上限。综述定性为**早期权宜方案，已逐渐淘汰**[^1]。

### 2. LLM + 外挂生成器（模块化联合）

LLM 与生成模块（通常是预训练扩散模型）通过更紧密的接口连接。综述按"LLM 怎么把信息传给生成器"再分两类[^1]：

- **靠自然语言传**（prompt-mediated）：LLM 写精细 prompt 给扩散模型。代表：Mini-Gemini、ModaVerse
- **靠中间向量传**（representation-mediated）：LLM 输出向量直接喂给生成器，绕过自然语言损耗。代表：GILL（首次对齐 LLM 语言空间和图像生成模型）、Emu 系列、[[Qwen-Image-Edit]]（frozen Qwen2.5-VL 出语义 + VAE 出外观 → MMDiT 扩散融合）[^2]

优点：复用现成扩散模型、训练成本低。缺点：自然语言中介丢细节，向量中介需额外对齐，生成上限仍受外部扩散模型限制。

### 3. 一个模型全包（端到端统一）

单一模型架构内统一处理理解 + 生成。**当前最主流也最具挑战性**的方向[^1]，按"用什么数学方法生成"细分四种：

**自回归（AR）**——LLM 那套"猜下一个 token"扩展到图像。具体路径：vision tokenizer（VQ-VAE / VQ-GAN）把一张 256×256 图压成 1k+ 个离散视觉 token，加进 LLM 词表，LLM 自回归预测视觉 token，最后 VQ decoder 还原像素。

代表：[[Chameleon]]（首个主流早期融合）、[[GPT-4o]]、[[Llama 4]]、[[Qwen2.5-Omni]]。

> [!warning] 纯 AR 生图的固有限制
> 综述明确点出四个问题[^1]：(1) **慢**——一张高分辨率图常 1k+ token 逐个生成；(2) **保真度受 VQ tokenizer 限制**；(3) **不符合图像内在 2D 结构**——硬塞成 1D 因果序列；(4) **误差累积**。这正是 AR-Diffusion 混合范式兴起的动机。

**扩散（Diffusion）**——扩散模型扩展到统一建模。

关键 nuance：纯 diffusion 自身**不"理解"文字**——文本理解外包给独立 text encoder，编码出的向量在每步去噪时通过 cross-attention 作为条件输入。Text encoder 越强，prompt 理解越精准：

| Text encoder | 理解能力 | 代表 |
|---|---|---|
| CLIP text encoder | 弱 | Stable Diffusion 1.x / 2.x、FLUX 早期 |
| T5 | 中 | Imagen、PixArt |
| LLM / VLM | 强 | [[Qwen-Image-Edit]]（用冻结的 Qwen2.5-VL）[^2]、Stable Diffusion 3 |

代表：UniDiffuser、CoDi、OmniFlow（基于矫正流的模块化设计）。

> [!note] Diffusion 自己学语言？
> 综述提到扩散语言模型（如 LLaDA）的进展[^1]——如果成熟，未来可能出现 diffusion 既理解又生成的统一架构，不再依赖外部 text encoder。但目前还在早期。

**AR-Diffusion 混合**——单一架构同时学 AR + diffusion。AR 强在 sequential / discrete（文本），diffusion 强在 continuous / spatial（图像）；hybrid 让一个模型用各自擅长的方式处理对应模态。

代表：Transfusion（单 Transformer，文本 AR 损失 + 图像扩散损失，开创性工作）、Show-o（next-token + masked-token 预测）、JanusFlow（解耦编码器 + 矫正流）、BAGEL。

综述认为混合建模**生成质量显著优于纯 AR，避免了模块化联合的信息传递瓶颈**[^1]——是当前研究最活跃的方向。

**其他**——编码器-解码器（OFA、Unified-IO）、状态空间模型（OmniMamba 用 Mamba-2 替代 Transformer 二次复杂度）、图结构（GraphGPT-o）[^1]。

## UFM 演化路径

综述把 UFM 演化划分为三阶段[^1]：**Specific**（理解和生成各自为战）→ **Combine**（功能集成实现复杂任务）→ **Emergent**（"通过生成图像来思考"的未来愿景）。当前主流前沿（GPT-4o、Llama 4、Chameleon、Janus）处于 Combine 阶段。

## Related

- [[Multimodal Models]] —— 多模态全景导航
- [[Qwen-Image-Edit]] —— 模块化联合代表（开源旗舰）
- [[GPT-4o]] / [[Chameleon]] / [[Llama 4]] —— 端到端统一 AR 代表
- [[Multimodal Inference Considerations]] —— UFM 各范式的推理 profile

[^1]: 让你更懂AI的 (2025-12-08), 介绍 NJU + CAS Auto + PKU 联合综述 *A Survey of Unified Multimodal Understanding and Generation: Advances and Challenges*（参考 750+ paper，83 页）. [[Sources/Clippings/统一多模态理解与生成综述：83页长文梳理进展和挑战]]
[^2]: Qwen Team, Alibaba (2025-08). *Qwen-Image Technical Report*. [[Sources/Papers/2508.02324v1.pdf]]
