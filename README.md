<img width="527" height="538" alt="image" src="https://github.com/user-attachments/assets/f78634af-a2c1-4961-bb53-09a246c39c12" />

# LLM From Scratch (PyTorch)

This repository implements a complete Large Language Model (LLM) from scratch using PyTorch. The objective is to understand and reproduce the end-to-end process of modern language model development, including architecture design, data processing, optimization, and alignment through reinforcement learning.

---

## Project Overview

The project follows a progressive structure, starting from the fundamental transformer architecture and expanding toward modern LLM training pipelines. Each stage focuses on building and testing individual components before integration into a larger system.

---


The environment is configured for GPU acceleration using CUDA where available. Mixed precision and profiling tools are used for efficient computation and performance monitoring.

---

## Part 0 — Foundations

- Overview of the LLM training pipeline: pretraining, fine-tuning, and alignment
- Environment configuration for PyTorch and CUDA
- Basic project structure and resource allocation for efficient experimentation

---

## Part 1 — Core Transformer Architecture

- Implementation of positional embeddings (absolute learned and sinusoidal)
- Self-attention mechanism from first principles with manual computation examples
- Construction of a single attention head and extension to multi-head attention
- Feed-forward network (MLP) with GELU activation and dimensionality expansion
- Integration of residual connections and LayerNorm
- Assembly of a full transformer block with sequential processing

---

## Part 2 — Training a Minimal LLM

- Byte-level tokenization and dataset creation for autoregressive training
- Sequence batching and label shifting for next-token prediction
- Cross-entropy loss computation for causal language modeling
- Implementation of a custom training loop without high-level frameworks
- Text generation through temperature sampling, top-k, and top-p sampling
- Validation and loss tracking over a held-out dataset

---

## Part 3 — Architectural Improvements

- Replacement of LayerNorm with RMSNorm and comparison of gradient behavior
- Rotary Positional Embeddings (RoPE) for contextualized attention
- SwiGLU activation in feed-forward layers for improved expressiveness
- Key-Value (KV) caching for efficient inference
- Sliding-window attention and attention sink mechanisms
- Rolling buffer KV cache for real-time streaming capabilities

---

## Part 4 — Scaling Techniques

- Transition from byte-level tokenization to Byte Pair Encoding (BPE)
- Implementation of gradient accumulation and mixed precision training
- Adaptive learning rate scheduling and warmup strategies
- Model checkpointing, resuming, and version management
- Logging and experiment visualization using TensorBoard and Weights & Biases

---

## Part 5 — Mixture-of-Experts (MoE)

- Theoretical understanding of MoE architectures: expert routing, gating networks, and load balancing
- Implementation of MoE layers and expert selection mechanisms in PyTorch
- Integration of hybrid dense–MoE transformer layers for scalability

---

## Part 6 — Supervised Fine-Tuning (SFT)

- Dataset formatting with instruction–response structures
- Application of causal LM loss with masked labels for instruction learning
- Implementation of curriculum learning strategies for progressive fine-tuning
- Quantitative evaluation of model responses against reference outputs

---

## Part 7 — Reward Modeling

- Construction of preference-based datasets with pairwise rankings
- Reward model architecture using a transformer encoder
- Training with Bradley–Terry and margin ranking loss functions
- Verification of reward consistency and shaping effects

---

## Part 8 — Reinforcement Learning with PPO

- Policy model derived from the SFT stage with an additional value head
- Reward generation via the trained reward model
- PPO objective implementation balancing reward maximization and KL regularization
- Rollout generation, reward normalization, and stability optimization
- Training monitoring with reward curves, KL divergence, and gradient norms

---

## Part 9 — Reinforcement Learning with GRPO

- Group-relative baseline computation with multiple completions per prompt
- Advantage calculation based on normalized group mean rewards
- PPO-style clipped policy loss without explicit value function
- Direct KL regularization term in the objective function
- Modified training loop with multi-sample rollouts and per-prompt normalization

---

## Technical Stack

- **Language:** Python 3.11  
- **Framework:** PyTorch  
- **Compute Acceleration:** CUDA, cuDNN  
- **Visualization:** TensorBoard, Weights & Biases  
- **Tokenization:** Byte-level and BPE tokenizers  
- **Precision:** Mixed precision (float16 / bfloat16)

---

## Objectives

- Develop a transformer-based language model from the ground up  
- Gain a practical understanding of training dynamics and optimization stability  
- Explore fine-tuning, reward modeling, and reinforcement learning techniques for model alignment  
- Establish a modular, extensible codebase for future experiments and scaling studies

---

## License

This project is released under the MIT License. See the [LICENSE](./LICENSE) file for details.
