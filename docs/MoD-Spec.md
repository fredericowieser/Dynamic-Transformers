### Summary of Mixture-of-Depths (MoD) Models

This document presents the Mixture-of-Depths (MoD) technique, a method for dynamically allocating compute in Transformer-based language models by leveraging a static compute budget. The core idea is to allow the network to learn which tokens require more or less computation and to skip unnecessary self-attention and MLP computations for certain tokens at certain layers. This is different from a vanilla transformer, which expends the same amount of compute per token in a forward pass.

#### Model Architecture and Implementation

The MoD architecture is akin to Mixture-of-Experts (MoE) transformers but with a crucial difference. Instead of routing tokens to one of many expert MLPs, MoD routes tokens to either a standard computational block (self-attention and MLP) or a residual connection. This routing is applied to both the MLP and multi-head attention components.

The implementation follows a three-step high-level strategy:
1. A static compute budget is set a priori, which is less than that of an equivalent vanilla transformer. This is done by capping the number of tokens ($k$) that can participate in a block's computations at a given layer.
2. A per-block router emits a scalar weight for each token, expressing the network's preference for that token to participate in the computation or to route around it.
3. The tokens with the top-k scalar weights are selected to participate in the block's computations. This uses a static computation graph with known tensor sizes, but the token identities that participate are dynamic and context-sensitive.

The router weight for a token embedding $x_{i}^{l}$ in layer $l$ is a scalar produced by a linear projection:

$r_{i}^{l}=\omega_{\theta}^{T}x_{i}^{l}$

The block's output for a given token $x_{i}^{l}$ is determined by comparing its router weight $r_{i}^{l}$ to a percentile threshold $P_{\beta}(R^{l})$:

$x_{i}^{l+1} = \begin{cases} f(x_{i}^{l}) + x_{i}^{l}, & \text{if } r_{i}^{l} > P_{\beta}(R^{l}) \\ x_{i}^{l}, & \text{if } r_{i}^{l} \le P_{\beta}(R^{l}) \end{cases}$

Here, $f$ comprises the self-attention and subsequent MLP. The total number of tokens that are processed by the block is the user-defined capacity, $C$ (or $k$), where $C < S$ (sequence length). This dynamic routing strategy, referred to as expert-choice routing, ensures a perfect load balance and removes the need for an auxiliary balancing loss.

#### The Causal Router for Autoregressive Sampling

A significant challenge with expert-choice routing is its non-causal nature, as the top-k operation depends on the router weights of future tokens in the sequence. This is problematic for autoregressive sampling where future tokens are not available.

To work around this, two methods are proposed for training the model to make causal routing decisions during inference.

1. Auxiliary Loss: A simple auxiliary loss is introduced on the router's outputs. This is a binary cross-entropy loss where the router's outputs are the logits and the targets are the top-k selections (1 if selected, 0 if not). This loss pressures the router to produce outputs above 0.5 for tokens that will be in the top-k and below 0.5 for those that will not. During sampling, the router's output can then be used to make a causal decision without looking at future tokens.

2. Auxiliary MLP Predictor: A small auxiliary MLP predictor is used, which receives the same inputs as the main router (with a stop gradient). Its output is a prediction of whether the token will be among the top-k in the sequence. This method does not affect the language modeling objective and, empirically, does not significantly impact the step speed. The paper notes this is a relatively easy auxiliary task, achieving 99% accuracy quickly.

These methods allow for autoregressive sampling by enabling the model to predict which tokens to route to or around a block based on the router's output for the current token alone.

#### Results and Performance

MoD models are shown to match or exceed the performance of baseline transformers with equivalent training FLOPs and wall-clock time. They require a fraction of the FLOPs per forward pass, and can be over 50% faster during post-training sampling.

Key findings from the experiments include:
* MoD models can achieve a lower loss for the same training FLOP budget as a baseline transformer.
* Smaller MoD models can perform as well or better than the isoFLOP-optimal baseline models while being significantly faster to step. For example, a 220M parameter MoD model slightly outperforms the isoFLOP-optimal baseline and is upwards of 60% faster to step during training.
* The optimal MoD variants in the study used a 12.5% capacity (meaning 12.5% of the sequence's tokens were processed by the self-attention and MLP, while 87.5% were routed around the block) and routed every other block.
* Learned routing is crucial, as a stochastic routing approach performed drastically worse than both the baseline and normal MoD transformers.
* The MoD technique can be integrated with MoE models, and this combined approach (MoDE) can yield compounded performance improvements.