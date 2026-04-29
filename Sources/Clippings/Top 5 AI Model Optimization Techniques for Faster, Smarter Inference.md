---
title: "Top 5 AI Model Optimization Techniques for Faster, Smarter Inference"
source: "https://developer.nvidia.com/blog/top-5-ai-model-optimization-techniques-for-faster-smarter-inference/"
author:
  - "[[Eduardo Alvarez]]"
published: 2025-12-10
created: 2026-04-08
description: "As AI models get larger and architectures more complex, researchers and engineers are continuously finding new techniques to optimize the performance and…"
tags:
  - "clippings"
---
As AI models get larger and architectures more complex, researchers and engineers are continuously finding new techniques to optimize the performance and overall cost of bringing AI systems to production.

Model optimization is a category of techniques focused on addressing inference service efficiency. These techniques represent the best “bang for buck” opportunities to optimize cost, improve user experience, and scale. These techniques range from fast and effective approaches like model quantization to powerful multistep workflows like pruning and distillation.

This post covers the top five model optimization techniques enabled through NVIDIA [Model Optimizer](https://github.com/NVIDIA/TensorRT-Model-Optimizer) and how each contributes to improving the performance, TCO, and scalability of deployments on NVIDIA GPUs.

These techniques are the most powerful and scalable levers currently available in Model Optimizer that teams can apply immediately to reduce cost per token, improve throughput, and accelerate inference at scale.

![A visual showing five cards, each with a small green-themed icon and headline. The techniques listed are: Post-Training Quantization (“Fastest Path to Optimization”), Quantization-Aware Training (“Simple Accuracy Recovery”), Quantization-Aware Distillation (“Max Accuracy and Speedup”), Speculative Decoding (“Speedup without Model Changes”), and Pruning & Distillation (“Slim Model and Keep Intelligence”). All cards use clean white backgrounds with NVIDIA-style green bar/brain/network iconography.](https://developer-blogs.nvidia.com/wp-content/uploads/2025/12/top-model-optimization-techniques-graphic-png.webp)

Figure 1. The top five most impactful model optimization techniques

## 1\. Post-training quantization

Post-training quantization (PTQ) is the fastest path to model optimization. You can leverage an existing model (FP16/BF16/FP8) and compress it to a lower precision format (FP8, NVFP4, INT8, INT4) using a calibration dataset—without touching the original training loop. This is where most teams should begin. It is easy to apply with Model Optimizer, and delivers immediate latency plus throughput wins, even on massive foundation models.

![Comparison of representable ranges and data precision for FP16, FP8, and FP4 formats. FP16 shows the widest range (−65,504 to +65,504) with closely spaced values A and B, representing high precision. FP8 has a narrower range (−448 to +448) with quantized values QA and QB spaced farther apart, indicating lower precision. FP4 shows an even smaller range (−6 to +6), illustrating the trade‑off between range and precision when reducing bit width.
](https://developer-blogs.nvidia.com/wp-content/uploads/2025/12/range-detail-quantizing-fp16-fp8-fp4-png.webp)

Figure 2. What happens to range and detail when quantizing from FP16 down to FP 8 or FP4

<table><tbody><tr><td><strong>Pros</strong></td><td><strong>Cons</strong></td></tr><tr><td rowspan="2">–Fastest time to value<br>–Achievable with small calibration dataset<br>–Memory, latency, and throughput gains stack with other optimizations<br>–Highly custom quantization recipes (<a href="https://developer.nvidia.com/blog/optimizing-inference-for-long-context-and-large-batch-sizes-with-nvfp4-kv-cache/">NVFP4 KV Cache</a>, for example)</td><td rowspan="2">–May require a different technique (QAT/QAD) if the quality floor drops under SLA</td></tr></tbody></table>

*Table 1. Pros and cons of PTQ*

To learn more, see [Optimizing LLMs for Performance and Accuracy with Post-Training Quantization](https://developer.nvidia.com/blog/optimizing-llms-for-performance-and-accuracy-with-post-training-quantization/).

## 2\. Quantization-aware training

Quantization-aware training (QAT) injects a short, targeted fine-tuning phase where the model is tuned to account for low precision error. It simulates quantization noise in the forward loop while computing gradients in higher precision. QAT is a recommended next step when additional accuracy is required beyond what PTQ has delivered.

![Flowchart illustrating the Quantization Aware Training (QAT) workflow. On the left, an original precision model is combined with calibration data and a Model Optimizer quantization recipe to form a QAT-ready model. This model, along with a subset of original training data, enters the QAT training loop. Inside the loop, high-precision weights are updated and then used as “fake quantization” weights during the forward pass. Training loss is calculated, and the backward pass uses a straight-through estimator (STE) to propagate gradients. The loop repeats until training converges.](https://developer-blogs.nvidia.com/wp-content/uploads/2025/12/qat-workflow-png.webp)

Figure 3. A model is prepared, quantized, and iteratively trained with simulated low-precision weights in a QAT workflow

<table><tbody><tr><td><strong>Pros</strong></td><td><strong>Cons</strong></td></tr><tr><td rowspan="2">–Recovers all or most of the accuracy loss at low precision<br>–Fully compatible with NVFP4, especially for FP4 stability</td><td rowspan="2">–Requires training budget plus data<br>–Takes longer to implement than PTQ alone</td></tr></tbody></table>

**Table 2. Pros and cons of QAT**

To learn more, see [How Quantization-Aware Training Enables Low-Precision Accuracy Recovery](https://developer.nvidia.com/blog/how-quantization-aware-training-enables-low-precision-accuracy-recovery/).

## 3\. Quantization-aware distillation

Quantization-aware distillation (QAD) goes one level beyond QAT. With this technique, the student model learns to account for quantization errors while simultaneously aligned to the full precision teacher through distillation loss. QAD doubles down on QAT by adding teaching elements from principles of distillation, enabling you to extract the maximum quality possible while running ultra-low precision at inference time. QAD is an effective option for downstream tasks that notoriously suffer from significant performance degradation after quantization.

![Flowchart of Quantization Aware Distillation (QAD). On the left, an original precision model is combined with calibration data and a quantization recipe to create a QAD-ready student model. This student model is paired with a higher precision teacher model and a subset of the original training data. In the QAD training loop, the student uses “fake quantization” weights in its forward pass, while the teacher performs a standard forward pass. Outputs are compared to calculate QAD loss, which combines distillation loss with standard training loss. Gradients flow back through the student model using a straight-through estimator (STE), and the student’s high-precision weights are updated to adapt to quantization conditions.
](https://developer-blogs.nvidia.com/wp-content/uploads/2025/12/qad-workflow-png.webp)

Figure 4. QAD trains a low-precision student model under teacher guidance, combining distillation loss with standard QAT updates

<table><tbody><tr><td><strong>Pros</strong></td><td><strong>Cons</strong></td></tr><tr><td rowspan="2">–Highest accuracy recovery<br>–Ideal for multistage post-training pipelines for easy setup and robust convergence</td><td rowspan="2">–Additional training cycles after pretraining<br>–Larger memory footprint<br>–Slightly more complex pipeline to implement today</td></tr></tbody></table>

**Table 3. Pros and cons of QAD**

To learn more, see [How Quantization-Aware Training Enables Low-Precision Accuracy Recovery](https://developer.nvidia.com/blog/how-quantization-aware-training-enables-low-precision-accuracy-recovery/).

## 4\. Speculative decoding

The decode step in inference is well known for suffering from sequential processing algorithmic bottlenecks. Speculative decoding tackles this directly by using a smaller or faster draft model (like EAGLE-3) to propose multiple tokens ahead, then verifying them in parallel with the target model. This collapses sequential latency into single steps and dramatically reduces required forward passes at long sequence lengths, without touching model weights.

Speculative decoding is recommended when you want immediate generation speedups without retraining or quantization, and it stacks cleanly with the other optimizations in this list to compound throughput and latency gains.

![A gif showing an example where the input is “The  Quick”. From this input, the draft model proposes “Brown”, “Fox”, “Hopped”, “Over”. The input and draft are ingested by the target model, which verifies “Brown” and “Fox” before rejecting “Hopped” and subsequently everything after. “Jumped” is the target model’s own generation resulting from the forward pass.](https://developer-blogs.nvidia.com/wp-content/uploads/2025/12/speculative-decoding-draft-target-approach-1.gif)

Figure 5. The draft-target approach to speculative decoding operates as a two-model system

<table><tbody><tr><td><strong>Pros</strong></td><td><strong>Cons</strong></td></tr><tr><td rowspan="2">–Radically reduces decode latency<br>–Stacks perfectly with PTQ/QAT/QAD plus NVFP4</td><td rowspan="2">–Requires tuning (acceptance rate is everything)<br>–Second model or head required depending on variant</td></tr></tbody></table>

**Table 4. Pros and cons of speculative decoding**

To learn more, see [An Introduction to Speculative Decoding for Reducing Latency in AI Inference](https://developer.nvidia.com/blog/an-introduction-to-speculative-decoding-for-reducing-latency-in-ai-inference/).

## 5\. Pruning plus knowledge distillation

Pruning is a structural optimization path. This technique removes weights, layers, and/or heads to make the model smaller. Distillation then teaches the new smaller model how to think like the larger teacher. This multistep optimization strategy permanently changes model performance because the baseline compute and memory footprint are permanently lowered.

Pruning plus knowledge distillation can be leveraged when other techniques in this list are unable to deliver the memory or compute savings necessary to meet application requirements. This approach can also be used when teams are open to making more aggressive changes to an existing model to adapt it for specific specialized downstream use cases.

![This diagram shows the successful outcome of knowledge distillation by comparing the teacher network to the smaller, trained student network. The student model, despite being more compact, produces an output probability vector that closely mimics the teacher's vector.](https://developer-blogs.nvidia.com/wp-content/uploads/2025/12/knowledge-distillation-trained-student-teacher-model-outputs-png.webp)

Figure 6. Knowledge distillation-trained student and teacher model outputs

<table><tbody><tr><td><strong>Pros</strong></td><td><strong>Cons</strong></td></tr><tr><td rowspan="2">–Reduces parameter count → permanent plus structural cost savings<br>–Enables smaller models that still behave like large models</td><td rowspan="2">–Aggressive pruning without distill → cliffs accuracy<br>–Requires more work to pipeline versus PTQ alone</td></tr></tbody></table>

**Table 5. Pros and cons of pruning plus knowledge distillation**

To learn more, see [Pruning and Distilling LLMs Using NVIDIA TensorRT Model Optimizer](https://developer.nvidia.com/blog/pruning-and-distilling-llms-using-nvidia-tensorrt-model-optimizer/).

## Get started with AI model optimization

Optimization techniques come in all shapes and sizes. This post highlights the top five model optimization techniques enabled through Model Optimizer.

- **PTQ, QAT, QAD,** and **pruning plus distillation** make your model intrinsically cheaper, smaller, and more memory efficient to operate.
- **Speculative decoding** makes generation intrinsically faster by collapsing sequential latency.

To get started and learn more, explore the deep-dive posts associated with each technique for technical explainers, performance insights, and Jupyter Notebook walkthroughs.