### Summary of Qwen2.5 Models: Mathematical and Algorithmic Specifics

This document details the architectural and training specifics of the Qwen2.5 series of models, which are built upon the Transformer-based decoder architecture.

#### Model Architecture

The core architecture for the dense models in the Qwen2.5 series includes several key components:
* Grouped Query Attention (GQA) is used to optimize KV cache utilization.
* The SwiGLU activation function is implemented for non-linear activation.
* Rotary Positional Embeddings (ROPE) are used to encode position information.
* QKV bias is incorporated into the attention mechanism.
* RMSNorm with pre-normalization is applied to ensure stable training.

For the proprietary Mixture-of-Experts (MoE) models, such as Qwen2.5-Turbo and Qwen2.5-Plus, the standard feed-forward network (FFN) layers are replaced with MoE layers. Each MoE layer contains multiple FFN experts and a routing mechanism that dispatches tokens to the top-K experts. The MoE architecture also uses fine-grained expert segmentation and shared expert routing.

The tokenizer is based on byte-level byte-pair encoding (BBPE) with a vocabulary of 151,643 regular tokens. The number of control tokens has been expanded from 3 to 22.

#### Pre-training

The pre-training process is enhanced by significant data scaling and optimization. The pre-training dataset has been increased from 7 trillion tokens to 18 trillion tokens. This data is curated using a multi-dimensional filtering process that leverages Qwen2-Instruct models to evaluate and score training samples. The data mixture is balanced by down-sampling overrepresented domains and up-sampling high-value domains like technology and science.

The training employs a two-phase long-context approach. An initial phase uses a 4,096-token context length, which is then extended to 32,768 tokens for all models except Qwen2.5-Turbo. For Qwen2.5-Turbo, a progressive context length expansion strategy advances through four stages: 32,768, 65,536, 131,072, and 262,144 tokens, with the ROPE base frequency increased from 10,000 to 1,000,000 using the ABF technique. The models' ability to handle long sequences during inference is further improved by implementing YARN and Dual Chunk Attention (DCA).

To prevent test data contamination, a training sequence $s_t$ is removed if a test sequence $s_e$ exists such that the length of their longest common subsequence (LCS) satisfies both conditions:

$|LCS(s_t, s_e)| \ge 13$
$|LCS(s_t, s_e)| \ge 0.6 \times min(|s_t|, |s_e|)$

#### Post-training

Post-training involves two main stages: Supervised Fine-tuning (SFT) and a two-stage reinforcement learning (RL) process.

Supervised Fine-tuning (SFT)
The SFT process uses over 1 million examples and is fine-tuned for two epochs with a sequence length of 32,768 tokens. The learning rate is gradually decreased from $7 \times 10^{-6}$ to $7 \times 10^{-7}$. A weight decay of 0.1 is applied, and gradient norms are clipped at a maximum value of 1.0 to prevent overfitting.

For mathematical problem-solving, chain-of-thought data from Qwen2.5-Math is used, with rejection sampling employed to produce step-by-step reasoning.

Reinforcement Learning (RL)
The RL process is split into Offline RL and Online RL.

1.  Offline RL: This stage uses Direct Preference Optimization (DPO). Responses that pass quality checks are used as positive examples, while those that fail are used as negative examples. A dataset of approximately 150,000 training pairs is constructed. The model is trained for one epoch with a learning rate of $7 \times 10^{-7}$.

2.  Online RL: This stage employs Group Relative Policy Optimization (GRPO). Queries with higher variance in response scores, as evaluated by the reward model, are prioritized to ensure more effective learning. Models are trained with a global batch size of 2048 and 2048 samples per episode. The training set is the same as that used for the reward model.