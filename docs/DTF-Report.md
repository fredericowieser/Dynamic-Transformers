# Dynamic Transformers

## Abstract

The Transformer's (TF) uniform allocation of computation across all tokens limits scalability, motivating conditional-compute methods such as Mixture-of-Depths (MoD). We propose a new architecture, the Dynamic Transformer (DTF), a token-wise routing mechanism inspired by predictive coding and Variational Predictive Routing (VPR). DTF augments a decoder-only TF with alternating Decision and Dynamic Layers: the Decision layer computes a standard TF block output ("posterior") and a lightweight PriorFFN prediction ("prior"), and the Dynamic Layer uses a surprise-based model-comparison (static vs change) to gate tokens at a fixed capacity, retaining a static compute graph. We adapt a pre-trained Qwen2.5-0.5B model under transfer learning and compare DTF to a re-implemented MoD at matched capacity, including ablations on prior expressivity and a parameter-efficient fine-tuning variant. DTF achieves a small, consistent reduction in validation loss relative to MoD and exhibits the expected routing dynamics, aligning qualitatively with VPR. However, both DTF and MoD underperform compared to a dense TF baseline on downstream benchmarks at a fixed capacity. Results are from a single seed and were not tested for statistical significance. These findings indicate that surprise-based routing is a viable inductive bias for token based conditional computation. We also ran further ablations on DTF to explore the dynamics of this architecture.

**Keywords:** Transformers, Bayesian Surprise, Efficient Deep Learning

***

## Table of Contents

* List of Figures
* List of Tables
* List of Abbreviations
* **1 Introduction**
    * 1.1 The Imperative for Dynamic Computation
        * 1.1.1 Uniformity in Standard Transformers
        * 1.1.2 Mixture of Depths
        * 1.1.3 A Principled Alternative: Surprise-Based Gating
    * 1.2 Research Questions and Contributions
    * 1.3 Thesis Structure
* **2 Background and Related Work**
    * 2.1 Notation
    * 2.2 From Recurrence to Attention
        * 2.2.1 Recurrent Neural Networks
        * 2.2.2 Long Short Term Memory Networks
        * 2.2.3 Transformers
        * 2.2.4 Mixture of Experts (MoE)
        * 2.2.5 Mixture of Depths (MoD)
        * 2.2.6 Other efficient Transformers
        * 2.2.7 Qwen2.5: a modern large scale family of models
    * 2.3 Bayesian Surprise and Hierarchical Models
        * 2.3.1 Bayesian surprise and predictive coding
        * 2.3.2 Variational Predictive Routing (VPR)
        * 2.3.3 Dynamic Latent Hierarchy (DLH)
        * 2.3.4 Neuroscience inspired ideas in Transformers
* **3 Methodology**
    * 3.1 The Dynamic Transformer (DTF) Architecture
        * 3.1.1 The Decision Layer and Prior
        * 3.1.2 The Dynamic Layer and Predictive Router
    * 3.2 A Bayesian Interpretation of the Predictive Router
    * 3.3 The MoD Baseline Architecture
    * 3.4 Training and Benchmarking
        * 3.4.1 Optimisation
        * 3.4.2 Pre-training corpora (this work)
        * 3.4.3 Benchmarks
* **4 Results and Analysis**
    * 4.1 Experimental Setup
    * 4.2 Primary Architecture Comparison: DTF vs. MoD
    * 4.3 Impact of Prior Network Expressivity
    * 4.4 Ablation Study: Parameter-Efficient Adaptation
* **5 Discussion**
    * 5.1 Summary of findings at fixed capacity
    * 5.2 Agreement with predictive-coding theory
    * 5.3 Why do our models underperform the baseline?
    * 5.4 Limitations
    * 5.5 Implications and recommendations
    * 5.6 Future work
    * 5.7 Concluding remarks
* **6 Conclusion**
    * 6.1 Summary of Contributions
    * 6.2 Answers to Research Questions
    * 6.3 Concluding Remarks and Future Directions
* **References**

***

## List of Figures

* **Figure 3.1:** High-level comparison of a standard TF (left) and the proposed DTF (right). The DTF replaces the uniform stack with alternating Decision (green) and Dynamic (yellow) layers.
* **Figure 4.1:** Training loss and perplexity learning dynamics for DTF and MoD at $\gamma=0.5$. Curves are smoothed with an exponential moving average over the last 15 steps.
* **Figure 4.2:** Validation loss (log-scale; EMA, 15 steps) for DTF and MoD at $\gamma=0.5$.
* **Figure 4.3:** Effect of PriorFFN size factor on training dynamics. The learning curves are largely insensitive to the prior's capacity.
* **Figure 4.4:** Effect of PriorFFN size factor on auxiliary prior loss and validation loss. Larger priors fit the posterior more accurately (a), yet the smallest prior yields the lowest final validation loss (b). Curves are smoothed with an exponential moving average over the last 15 steps.
* **Figure 4.5:** Evolution of the Predictive Router's parameters across prior sizes, showing distinct learned policies.
* **Figure 4.6:** Combined gating signal components ($S_{CE}$, $S_{CU}$, and $G_{cont}$) by prior factor over training. Each panel corresponds to one prior size (top-left: 0.0625, top-right: 0.125, bottom-left: 0.25, bottom-right: 0.5). In every panel, the pink trace is $S_{CE}$, the cyan trace is $S_{CU}$, and the unique, for each prior factor size, colour trace in the top is $G_{cont}$. The model consistently shifts emphasis to the CE signal over training. Mean activations remain broadly similar across prior sizes, consistent with a stable routing budget.
* **Figure 4.7:** Training loss and perplexity for Full Fine-tuning vs. LoRA adaptation methods. Full fine-tuning achieves a lower final training loss generally, indicating better generalisation. Curves are smoothed with an exponential moving average over the last 15 steps.
* **Figure 4.8:** Auxiliary loss for the PriorFFN under Full Fine-tuning and LoRA. The fully fine-tuned PriorFFN learns a more accurate prediction of the main block's output. Auxiliary loss curves are smoothed with an exponential moving average over the last 15 steps.
* **Figure 4.9:** Evolution of the four learnable parameters of the DTF's Predictive Router for Full Fine-tuning vs. LoRA adaptation. Both methods learn stable but distinct routing policies.
* **Figure 4.10:** Analysis of gating signal components for Finetune (left) and LoRA (right) adaptation. The red/purple trace is the final gate $G_{cont}$, pink is $S_{CE}$, and cyan is $S_{CU}$. Both methods exhibit the characteristic transition from novelty-based routing (higher $S_{CU}$) to prediction-based routing (higher $S_{CE}$) as training progresses.

***

## List of Tables

* **Table 3.1:** Pre-training data mixture used for fine-tuning. All datasets are available on the Hugging Face Hub. Ratios refer to the subset of the original training split used.
* **Table 4.1:** Benchmark performance for 0.5B parameter models. Scores are accuracy (%). All dynamic models operate at a capacity of $\gamma=0.5$. Bold indicates the best performance among the dynamic variants.
* **Table 4.2:** Impact of PriorFFN expressivity on benchmark performance for 0.5B DTF models. All models operate at a capacity of $\gamma=0.5$. Scores are accuracy (%). Bold indicates the best performance for each benchmark.
* **Table 4.3:** Benchmark performance comparison between a fully fine-tuned DTF (Prior 0.0625) and a parameter-efficient DTF adapted with LoRA. Both models operate at a capacity of $\gamma=0.5$. Scores are accuracy (%). Bold indicates the better performance for each benchmark.

***

## List of Abbreviations

* **DLH:** Dynamic Latent Hierarchies
* **DTF:** Dynamic Transformer
* **FFN:** Feed-Forward Network
* **FLOPS:** Floating Point Operations
* **KL:** Kullback-Leibler (divergence)
* **LLM:** Large Language Model
* **LORA:** Low-Rank Adaptation
* **MSE:** Mean Squared Error
* **MOD:** Mixture of Depths
* **MoE:** Mixture of Experts
* **RMSNorm:** Root Mean Square Layer Normalisation
* **RNN:** Recurrent Neural Network
* **ROPE:** Rotary Position Embedding
* **SiLU:** Sigmoid Linear Unit
* **SwiGLU:** Swish-Gated Linear Unit
* **TF:** Transformer
* **VPR:** Variational Predictive Routing

***

## 1. Introduction

### 1.1 The Imperative for Dynamic Computation

The rise of the Transformer (TF) architecture has become a popular base for many models aimed at predicting sequences with large scale compute and capability [1, 2, 3]. Yet, this success is built upon a computational paradigm that is expressive but seemingly inefficient when dealing with long sequences. The standard TF operates with a uniform processing of all elements in a sequence, a design choice that, while simplifying implementation, imposes constraints on scalability and performance, particularly in the context of long sequence processing [4, 5]. This inefficiency has created a clear imperative for a new class of models capable of dynamic, conditional computation [6, 7, 8, 9]. This work introduces the Dynamic Transformer (DTF), a novel architecture that, builds on top of decoder-only TFs and shares many similarities to Mixture of Depths (MoD) [10], a method that allows a TF based model to route which parts of the sequence should be processed. DTF emerges as a compute-efficient, information-theoretic approach that offers an interesting direction for TFs.

All code and implementation can be found at: [https://github.com/fredericowieser/Dynamic-Transformers](https://github.com/fredericowieser/Dynamic-Transformers)

For clarity, the terms "TF blocks" and "TF layers" are used synonymously throughout this work to refer to the same computational units within the architecture.

#### 1.1.1 Uniformity in Standard Transformers

The main limitation of the standard TF is its uniform application of computational resources [1, 11]. In TFs, we process sequences in terms of tokens, which are vectors that each represent a single element of the sequence at a given time. In decoder-only TFs, at every layer, each token in an input sequence is subjected to the same computationally intensive operations, primarily some variant of causal self-attention and a feed-forward neural network (FFN), these kinds of TFs are referred to as dense. We posit that this approach is misaligned with the nature of information in sequential data, where the relevance and complexity are rarely distributed evenly throughout the sequence [10].

The inefficiency of this paradigm is most starkly revealed by the self-attention mechanism's computational complexity, which scales quadratically [1] with the sequence length T, denoted as $O(T^{2})$. As models are tasked with processing increasingly long contexts, this quadratic scaling creates a bottleneck, rendering the processing of such sequences computationally infeasible for standard architectures, given current hardware limitations [11]. This uniform expenditure means that the same number of floating-point operations (FLOPs) can be spent on a semantically critical keyword as on a common stop word, representing a limitation of the current paradigm present in standard dense TFs, although it should be noted that Mixture of Experts (MoE) [7, 12] and MoD [10] employ distinct approaches to saving compute that are theoretically compatible with each other.

Beyond the raw computational cost, this uniform treatment can also pose a challenge to maintaining the quality of the model's representations over long contexts. As the sequence length grows, the attention mechanism must aggregate information from an ever-larger set of tokens. This may lead to a phenomenon of 'representation degradation', where token embeddings become a noisy average of the entire context, potentially losing their distinctiveness and semantic precision [4, 5]. This dual challenge of prohibitive computational cost and diminishing representational quality underscores the necessity of moving beyond the paradigm of uniform processing.

#### 1.1.2 Mixture of Depths

In response to these inefficiencies, the field has increasingly turned to conditional computation. The dominant and most successful paradigm to date is a class of engineering-driven solutions, exemplified by MoE [12, 6, 7] and, more recently, MoD [10]. These methods introduce token-level dynamism by learning to route tokens based on a heuristic of their predicted "importance".

In a typical MoD layer, a small, learnable router function $R^{(l)}$ maps the representation of each token $x_{i}^{(l)}\in\mathbb{R}^{d}$ at the layer to a scalar score $r_{i}\in\mathbb{R}$ for each $i\in\{1,...,T\}$. An expert-choice routing scheme then selects the TopK tokens, those with the k highest scalar scores, for full processing by the TF block, while the remainder bypass these computationally expensive operations via a residual connection. This mechanism operates based on a predefined capacity $\gamma\in(0,1]$, a real number, which specifies that $k=\lfloor\gamma T\rfloor$ tokens will be processed, where T is the total sequence length. A key advantage of this fixed-k approach is its compatibility with modern hardware accelerators. Because $\gamma$ is a fixed hyperparameter, the model can be implemented with a static computational graph, allowing hardware optimisations to leverage known tensor sizes a priori. This enables significant speed increases in practice compared to dynamic graphs, where variable token counts would nullify such benefits.

These methods, while highly effective, are fundamentally guided by an engineering-centric philosophy. Their routing logic is based on a learned but largely uninterpretable heuristic of importance. The decision to process a token is based on the magnitude of a scalar value - a mechanism optimised for downstream performance but lacking a deeper theoretical justification for why a particular token warrants additional computation. A significant challenge for decoder-only TFs arises because the TopK operation is inherently non-causal during autoregressive sampling. This means that determining whether a token's routing weight is among the TopK requires information from future tokens, which is unavailable at inference time. To address this, MoD introduces an auxiliary causal predictor network, trained to approximate the TopK routing decisions based only on past information [10].

#### 1.1.3 A Principled Alternative: Surprise-Based Gating

We explore a more theoretically grounded paradigm for conditional computation. It moves beyond the heuristic of "importance" and instead grounds its gating decisions in a concept from computational neuroscience and information theory: "surprise" [13] quantified as the degree of belief update when new information is observed. The central hypothesis is that allocating computational resources based on detecting moments of high informational gain provides a more powerful and effective inductive bias for learning.

This approach is directly inspired by a class of neuroscience-inspired temporal models, namely Variational Predictive Routing (VPR) [14] and the Dynamic Latent Hierarchy (DLH) [15]. These models leverage the principles of Predictive Coding [16, 17]-a theory that posits that the brain minimizes prediction error by continuously comparing internal models with sensory input to dynamically gate information flow over time. The DTF architecture proposed herein translates this temporal, event-driven mechanism into the token-wise domain of the TF. Inspiration for DTF also comes from previous work that leverages Bayesian surprise for segmenting and organising episodic memory in Large Language Models (LLM). Notably, The Episodic Memory LLM (EM-LLM) [18] uses online surprise (token-level Negative Log Likelihood) to identify event boundaries in text, demonstrating evidence that TFs and humans segment episodes similarly.

One trajectory in the field of conditional computation, exemplified by MoD, is guided by systems engineering principles. In contrast, the second trajectory, represented by the DTF, is rooted in cognitive and neuroscientific theory. Instead of asking, "Which token is most important?", the DTF asks, "Which token represents the most surprising new information, requiring an update to the model's internal state?". The DTF's primary significance lies in its position as a key exemplar offering a potential path toward more nuanced and interpretable dynamic models.

### 1.2 Research Questions and Contributions

This thesis seeks to address the following research questions:

1.  Can a neuroscience-inspired, event-driven routing mechanism be integrated into a pre-trained TF to enable more efficient scaling while maintaining competitive performance?
2.  How does the empirical performance-compute trade-off of DTF compare to a standard MoD implementation?
3.  Is a parameter-efficient, transfer-learning approach a viable methodology for instantiating such a dynamic architecture from a pre-trained, dense TF?

The primary contributions of this work are:

1.  The design and implementation of the DTF, a novel architecture that integrates a VPR-inspired, predictive routing mechanism into a standard decoder-only TF.
2.  A empirical comparison of the DTF against a re-implemented MoD baseline, evaluated in a parameter-efficient, transfer-learning setting from a Qwen2.5 0.5B pre-trained LLM [3] at a fixed capacity of $\gamma=0.5$, to assess the practical trade-offs between the two paradigms.
3.  An analysis of the DTF's internal routing behaviour, providing empirical evidence for its learning dynamics.

### 1.3 Thesis Structure

The remainder of this thesis is structured as follows:

* **Chapter 2** reviews the foundational work in sequence modelling and provides a critical analysis of the two conditional computation paradigms central to this work.
* **Chapter 3** details the architectural design of the DTF, its Bayesian interpretation, and the experimental methodology.
* **Chapter 4** presents the core empirical results and analyses of the model's performance and internal dynamics.
* **Chapter 5** provides a discussion and critical interpretation of these findings, including an analysis of the project's limitations.
* **Chapter 6** concludes the thesis with a summary of its contributions and directions for future research.

***

## 2. Background and Related Work

This chapter formalises the mathematical and conceptual foundations of the thesis. We begin with notation and a precise treatment of the evolution from recurrent models to attention, culminating in decoder-only TF and more recent scalable variants such as Qwen2.5 [3]. We then examine conditional computation methods in TFs: engineering driven approaches (MoE and MoD) and other efficient TF variants. Next, we develop a principled, probabilistic perspective based on Bayesian surprise and hierarchical generative models, with detailed expositions of VPR and DLH, and we connect these ideas to neuroscience inspired mechanisms for long context TFs.

### 2.1 Notation

**Sequences and embeddings.** A token sequence is $x_{1:T}=(x_{1},...,x_{T})$ with $x_{t}\in \mathcal{V},|\mathcal{V}|=V$. A learned embedding $E:\mathcal{V}\rightarrow\mathbb{R}^{d_{x}}$ maps $x_{t}\mapsto x_{t}^{emb}=E(x_{t})$. For (padded) length T and batch size B, we stack as $X\in\mathbb{R}^{T\times d}$ or $\mathbb{R}^{B\times T\times d}$. Layer index $(l)\in\{1,...,L\}$; multi-head attention uses H heads with per-head dims $d_{k}$, $d_{v}$ (often $d_{k}=d_{v}=d/H$).

**Masks.** Additive attention mask $M\in\mathbb{R}^{T\times T}$ encodes structure: causal masking uses strictly upper-triangular $-\infty$ above the diagonal, 0 elsewhere; padding masks can be added to M.

**Probabilities and loss.** Expectation $\mathbb{E}[\cdot]$, variance $Var[\cdot]$, covariance $Cov[\cdot,\cdot]$, probability $\mathbb{P}(\cdot)$. Binary cross-entropy (BCE) for target $m\in\{0,1\}$ and logit $r\in\mathbb{R}$ is $BCE(\sigma(r),m)=-(m \log \sigma(r)+(1-m)\log(1-\sigma(r)))$.

**Divergence.** KL divergence $D_{KL}(q||p)=\mathbb{E}_{z\sim q}[\log\frac{q(z)}{p(z)}]\ge0$ with equality iff $q=p$.

**Pointwise functions.**
$ReLU(a)=max(0,a)$, $tanh(a)=\frac{e^{a}-e^{-a}}{e^{a}+e^{-a}}$, $\sigma(a)=\frac{1}{1+e^{-a}}$, $Softplus(a) = log(1+e^a)$, $SiLU(a)=a~\sigma(a)$, $GELU(a)\approx0.5~a(1+tanh(\sqrt{2/\pi}(a+0.044715~a^{3})))$. Row-wise softmax for $z\in\mathbb{R}^{n}$ is $softmax(z)_{i}=\frac{e^{z_{i}}}{\sum_{j=1}^{n}e^{z_{j}}}$.

**Normalisation.** Layer Norm on $h\in\mathbb{R}^{d}$: $LayerNorm(h)=g\odot\frac{h-\mu1}{\sqrt{\sigma^{2}+\epsilon}}+b$ with $\mu=\frac{1}{d}\sum_{i}h_{i}$, $\sigma^{2}=\frac{1}{d}\sum_{i}(h_{i}-\mu)^{2}$, gain/bias $g, b\in\mathbb{R}^{d}$, $\epsilon>0$. RMSNorm: $RMSNorm(h)=g\odot\frac{h}{\sqrt{\frac{1}{d}\sum_{i}h_{i}^{2}+\epsilon}}$.

**Position-wise FFN.** For each position, $FFN(x)=W_{2}\phi(W_{1}x+b_{1})+b_{2}$, with nonlinearity $\phi$ (e.g. ReLU, GELU, or gated variants such as SwiGLU: $\phi(a,g)=SiLU(a)\odot g$).

### 2.2 From Recurrence to Attention

#### 2.2.1 Recurrent Neural Networks

We consider a discrete token sequence $x_{1:T}=(x_{1},...,x_{T})$, where each $x_{t}\in\mathcal{V}$ is embedded by a learned map $E:\mathcal{V}\rightarrow\mathbb{R}^{d_{x}}$ into $x_{t}=E(x_{t})$. A (single-layer) RNN maintains a hidden state $h_{t}\in\mathbb{R}^{d_{h}}$ and (optionally) produces an output $y_{t}\in\mathbb{R}^{d_{y}}$ at each time step t. The parameters are $\{W_{xh}\in\mathbb{R}^{d_{h}\times d_{x}}, W_{hh}\in\mathbb{R}^{d_{h}\times d_{h}}, W_{hy}\in\mathbb{R}^{d_{y}\times d_{h}}, b_{h}\in\mathbb{R}^{d_{h}}, b_{y}\in\mathbb{R}^{d_{y}}\}$, and $\phi:\mathbb{R}\rightarrow\mathbb{R}$ denotes a pointwise nonlinearity (e.g. tanh, ReLU). A standard Recurrent Neural Network (RNN) computes hidden states via:
$h_t = \phi(W_{xh}E(x_t) + W_{hh}h_{t-1}+b_h)$, $y_{t}=W_{hy}h_{t}+b_{y}$, with a pointwise nonlinearity [19]. Training uses backpropagation through time (BPTT), which propagates gradients across $O(T)$ steps. Let $\mathcal{L}=\sum_{t}l_{t}$ be the loss; then $\frac{\partial\mathcal{L}}{\partial h_{\tau}}=\sum_{t\ge\tau}(\prod_{j=\tau+1}^{t}J_{j})\frac{\partial\mathcal{L}_{t}}{\partial h_{t}}$, where $J_{j}=\frac{\partial h_{j}}{\partial h_{j-1}}$. Products of Jacobians induce vanishing/exploding gradients when $||J_{j}||$ has spectral norm below/above 1, hampering long range credit assignment [20]. Additionally, RNNs are inherently sequential, preventing parallelisation across time at training and inference.

#### 2.2.2 Long Short Term Memory Networks

Long Short-Term Memory (LSTM) networks, an evolution of RNNs, address vanishing gradients in RNNs by introducing gated memory and a persistent cell state [21]. At each time step t, given input embedding $E(x_{t})$ and previous hidden state $h_{t-1}$, an LSTM layer computes:

$f_{t}=\sigma(W_{xf}E(x_{t})+W_{hf}h_{t-1}+b_{f})$
$i_{t}=\sigma(W_{xi}E(x_{t})+W_{hi}h_{t-1}+b_{i})$
$o_{t}=\sigma(W_{xo}E(x_{t})+W_{ho}h_{t-1}+b_{o})$
$\tilde{c}_{t}=tanh(W_{xc}E(x_{t})+W_{hc}h_{t-1}+b_{c})$
$c_{t}=f_{t}\odot c_{t-1}+i_{t}\odot\tilde{c}_{t}$
$h_{t}=o_{t}\odot tanh(c_{t})$

Here, $f_{t}$ (forget), $i_{t}$ (input), and $o_{t}$ (output) are element-wise gates in $(0,1)^{d}$ (via the logistic sigmoid activation function), controlling how information is discarded, written, and exposed. The candidate $\tilde{c}_{t}$ proposes new content (via tanh), the cell state $c_{t}$ accumulates long-term information, and the hidden state $h_{t}$ is the layer's output. The element-wise product $\odot$ applies gating. This design enables stable gradient flow and selective memory over long contexts, though computation remains sequential across time. The additive memory update $c_{t}=f_{t}\odot c_{t-1}+\cdot\cdot\cdot$ helps mitigate vanishing gradients, but the sequential dependency still limits throughput and parallelism. In practice, this makes training LSTMs comparatively costly on modern hardware, since their computation cannot be parallelised across time to the same extent as TF layers.

#### 2.2.3 Transformers

**Tokens, embeddings, and positions.** As in the RNN setting, a discrete sequence $x_{1:T}$ with $x_{s}\in\mathcal{V}$ is mapped to continuous vectors by a learned embedding $E:\mathcal{V}\rightarrow\mathbb{R}^{d}$, yielding $X^{emb}\in\mathbb{R}^{T\times d}$ with rows $E(x_{s})$. Unlike RNNS, TFs have no inherent notion of order from recurrence, so positional information is injected additively or multiplicatively into the token representations (e.g. absolute learned/sinusoidal embeddings or rotary position encodings applied to Q, K). Thus the layer input is typically $X=X^{emb}+P\in\mathbb{R}^{T\times d}$, where P encodes positions. A key difference to RNNs is that all positions are processed in parallel within a layer (self-attention and position-wise FFNs), whereas RNNs update one step at a time; however, causality is enforced by a mask so that the representation at position t depends only on $\{1,...,t-1\}$ in decoder-only models. Apart from this order handling, the tokenisation and embedding pipeline are analogous to the RNN case.

The TF dispenses with recurrence and convolutions in favour of self-attention and position-wise feed-forward networks (FFNs), arranged as residual blocks with normalisation [1]. In its original encoder-decoder form, the encoder maps an input sequence to a sequence of continuous representations, and the decoder auto-regressively generates an output sequence while attending to its own previous states (masked self-attention) and to the encoder outputs (cross-attention). In decoder-only (causal) language models, a single stack of masked self-attention ensures that the representation of position t depends only on positions < t.

Let $X\in\mathbb{R}^{T\times d}$ denote the row-wise stack of token representations for a length-T sequence (possibly padded), where $T\in\mathbb{N}$ and $d\in\mathbb{N}$ is the model (hidden) dimension. Multi-head attention uses $H\in\mathbb{N}$ heads, with per-head key/query and value dimensions $d_{k}, d_{v}\in\mathbb{N}$ (often $d_{k}=d_{v}=d/H)$. For a single head, the queries, keys, and values are linear projections of X,
$Q=XW_{Q}$, $K=XW_{K}$, $V=XW_{V}$, with $W_{Q},W_{K}\in\mathbb{R}^{d\times d_{k}}$, $W_{V}\in\mathbb{R}^{d\times d_{v}}$.

Scaled dot-product attention compares each query against all keys and returns a weighted sum of values,
$$Attn(Q,K,V)=softmax(\frac{QK^{\top}}{\sqrt{d_{k}}}+M)V$$
where $M\in\mathbb{R}^{T\times T}$ is an additive mask that enforces structure (for causality, M is strictly upper-triangular with $-\infty$ above the diagonal and 0 elsewhere), and the row-wise softmax yields convex weights. The $\sqrt{d_{k}}$ factor stabilises gradients at larger head widths.

Multi-head attention runs (2.1) in parallel across $m\in\{1,...,H\}$ with independent projections $W_{Q}^{(m)}, W_{K}^{(m)}\in\mathbb{R}^{d\times d_{k}}$ and $W_{V}^{(m)}\in\mathbb{R}^{d\times d_{v}}$, producing per-head outputs $head_m = Attn(XW_{Q}^{(m)},XW_{K}^{(m)},XW_{V}^{(m)})\in\mathbb{R}^{T\times d_{v}}$. Concatenating heads along the feature dimension and projecting back to d gives
$MHA(X)=$ Concat $(head_1,...., head_H) W_{O}$, with $W_{O}\in\mathbb{R}^{(Hd_{v})\times d}$.

In encoder-decoder models, each decoder layer typically includes both masked self-attention (with X the decoder states) and cross-attention (with queries from the decoder and keys/values from the encoder outputs). Each TF block comprises a (masked) multi-head attention sub-layer and a position-wise FFN, each wrapped with residual connections and normalisation (Layer Norm or RMSNorm [22, 23]). The FFN applies the same two-layer non-linear transformation to each position independently; the original TF used ReLU, while modern variants commonly adopt gated activations such as SwiGLU [24] to improve optimisation and capacity. Residual connections make the stack a residual (pre-normalised) network, which improves gradient flow to deeper layers during back-propagation.

Attention is permutation-invariant, so positional information must be injected. Absolute positional embeddings (learned or sinusoidal) [1] add position-dependent vectors to token embeddings, while rotary position embeddings (ROPE) apply relative phase rotations to Q and K that preserve dot-product geometry and aid length extrapolation [25]. In all cases, the positional mechanism fixes which token order the model should attend over.

**Decoder-only (causal) formulation.** Autoregressive language models factorise the joint probability of a token sequence $x_{1:T}$ using the chain rule,
$$p_{\theta}(x_{1:T})=\prod_{t=1}^{T}p_{\theta}(x_{t}|x_{<t})$$and train by maximum likelihood, which corresponds to minimising the token-level negative log-likelihood (NLL),$$\mathcal{L}(\theta)=\mathbb{E}_{x_{1:T}\sim p_{data}}[-\sum_{t=1}^{T}\log p_{\theta}(x_{t}|x_{<t})]$$
During training, teacher forcing conditions the model on the ground-truth prefix $x_{<t}$ to compute the conditional at each step, thereby directly optimising for next token prediction (2.3). At inference time, tokens are generated sequentially by feeding previously generated tokens back into the model. To reduce the per-token cost, implementations cache the keys and values (K, V) computed for past positions (the "KV cache"), so that generating the next token only requires projecting the new query and computing attention against the cached K, V. With L layers and sequence length T, caching reduces the incremental complexity from $O(LT^{2}d)$ to $O(LTd)$ per step. Since memory and wall-clock are typically dominated by attention, $I/O$-aware kernels such as FlashAttention accelerate exact attention without approximation [11, 26].

**Training pipeline.** Modern decoder-only LLMs are commonly trained in stages: a large-scale pre-training phase on diverse unlabelled corpora yields a base (pre-trained) model that captures general language regularities; downstream adaptation may then apply supervised fine-tuning (SFT), reinforcement learning from human feedback (RLHF), or related methods to align behaviour with task- or instruction-specific desiderata.

**Parameter-Efficient Fine-Tuning (PEFT).** The prohibitive cost of full fine-tuning for large models has driven the development of PEFT methods. A prominent PEFT technique is Low-Rank Adaptation (LORA) [27], which is based on the hypothesis that the change in model weights during adaptation has a low intrinsic rank. Instead of updating the full weight matrix W, LORA freezes the pre-trained weights and injects a pair of trainable, low-rank decomposition matrices, A and B, into each layer. By only optimising these small adapter matrices, LoRA dramatically reduces the number of trainable parameters and the memory footprint required for adaptation. Critically, this approach introduces no additional inference latency, as the adapter matrices can be merged with the original weights after training is complete.

#### 2.2.4 Mixture of Experts (MOE)

An MoE layer replaces a dense position-wise FFN with a set of $E\in\mathbb{N}$ expert networks $\{E_{i}\}_{i=1}^{E}$, each a feed-forward map $E_{i}:\mathbb{R}^{d}\rightarrow\mathbb{R}^{d}$, and a router that produces sparse, per-token mixing weights [12, 7]. Let $x\in\mathbb{R}^{d}$ denote the representation of a single position (a row of $X\in\mathbb{R}^{T\times d})$. The router computes logits
$$l(x)=xW_{g}\in\mathbb{R}^{E}, \quad W_{g}\in\mathbb{R}^{d\times E}$$and a sparse gating vector $G(x)\in\mathbb{R}^{E}$ via a TopK softmax:$$G(x)=softmax(TopK(l(x),k)), \quad k\in\{1,...,E\}$$where $TopK(\cdot,k):\mathbb{R}^{E}\rightarrow\mathbb{R}^{E}$ keeps the k largest coordinates and sets the rest to -∞ element-wise:$$[TopK(v, k)]_i = \begin{cases} v_i & \text{if } v_i \text{ is among the top k entries of v,} \\ -\infty & \text{otherwise.} \end{cases}$$Let $\mathcal{K}(x)\subseteq\{1,...,E\}$ denote the active expert indices, i.e. those with $G_{i}(x)>0$. The MoE output for $x$ is the sparse mixture$$y=\sum_{i\in\mathcal{K}(x)}G_{i}(x)E_{i}(x)\in\mathbb{R}^{d}$$
In TF blocks, MoE layers typically replace the position-wise FFN sub-layer and are applied independently at each position, preserving the model dimension d and the residual/normalisation scheme. Practical systems often include auxiliary load-balancing terms that encourage uniform expert utilisation over a batch (e.g. "importance" and "load" losses in [12]). MoE increases total parameter count without a proportional increase in per-token compute by keeping $k\ll E$.

#### 2.2.5 Mixture of Depths (MoD)

MoD learns to allocate a fixed token capacity per block and routes the remaining tokens around the block via the residual path [10]. Concretely, in a decoder-only TF the routing applies to both the (masked) self-attention and the subsequent position-wise FFN, so selected tokens are (i) updated by the block and (ii) made available as keys/queries/values for other selected tokens at that layer; tokens not selected are passed through unchanged. Because the capacity per block is fixed a priori, the computation graph and tensor shapes remain static, while the identities of the processed tokens are dynamic and context-dependent.

Let $X^{(l)}=[x_{1}^{(l)};...;x_{T}^{(l)}]\in\mathbb{R}^{T\times d}$ be the layer-l input (one sequence of length T), and let $f^{(l)}$ denote the usual TF block (masked self-attention followed by an FFN). A lightweight router $R^{(l)}:\mathbb{R}^{d}\rightarrow\mathbb{R}$ assigns a score to each token,
$r_{i}^{(l)}=R^{(l)}(x_{i}^{(l)})$, $i=1,...,T$.

Given a user-chosen capacity $k^{(l)}=\lfloor\gamma^{(l)}T\rfloor$ with $\gamma^{(l)}\in(0,1]$, we select the TopK indices
$\mathcal{S}^{(l)}=TopK(r^{(l)},k^{(l)})\subseteq\{1,...,T\}$, $m_{i}^{(l)}=1[i\in\mathcal{S}^{(l)}]$, and apply the block only to the selected rows: $Y_{sel}=f^{(l)}(X_{\mathcal{S}^{(l)}}^{(l)})\in\mathbb{R}^{k^{(l)}\times d}$. Let $rank_{\mathcal{S}^{(l)}}(i)\in\{1,...,k^{(l)}\}$ denote the position of i within $\mathcal{S}^{(l)}$. The residual update "scatters" $Y_{sel}$ back to the original layout:
$$x_{i}^{(l+1)}=\begin{cases}x_{i}^{(l)}+Y_{sel}[rank_{\mathcal{S}^{(l)}}(i)],&i\in\mathcal{S}^{(l)},\\ x_{i}^{(l)},&i\notin\mathcal{S}^{(l)}.\end{cases}$$
Selection is performed independently per sequence in the batch. In practice $\gamma^{(l)}$ is either held fixed across layers or applied to every other layer [10].

**Causal router for autoregressive inference.** The training-time TopK selection over the full sequence is non-causal: whether token i is in $\mathcal{S}^{(l)}$ depends on tokens $j>i$. For autoregressive decoding we therefore train a small causal router $\tilde{R}^{(l)}:\mathbb{R}^{d}\rightarrow\mathbb{R}$ to predict the training-time selection using only the token's current representation. Let $\hat{m}_{i}^{(l)}=1[i\in\mathcal{S}^{(l)}]$ be the binary target derived from TopK, and let $\tilde{r}_{i}^{(l)}=\tilde{R}^{(l)}(stop\_grad(x_{i}^{(l)}))$ be the causal router logit (the stop-gradient prevents leakage of the language-model objective into $\tilde{R}^{(l)}$). We add a binary cross-entropy auxiliary loss
$$\mathcal{L}_{causal}^{(l)}=\frac{1}{T}\sum_{i=1}^{T}BCE(\sigma(\tilde{r}_{i}^{(l)}),\hat{m}_{i}^{(l)})$$
and use $\tilde{R}^{(l)}$ at inference: at each layer and decoding step we compute $\tilde{r}_{i}^{(l)}$ for the currently available prefix tokens and select $TopK(\tilde{r}^{(l)},k^{(l)})$ to determine which tokens participate in that block. Empirically this causal approximation tracks the non-causal TopK very closely (cf. [10]), and incurs negligible overhead.

#### 2.2.6 Other efficient Transformers

* **Universal Transformers:** weight sharing across layers with dynamic halting (per position computation depths) gives a form of adaptive depth while retaining parallel attention [8].
* **Switch Transformers:** a single expert MoE variant with top 1 dispatch and stabilising techniques for extreme scale [6].
* **COLT5:** conditional computation in long-range encoder decoder models; tokens choose light vs. heavy FFN pathways and sparse attention patterns, improving efficiency for long sequences [9].

#### 2.2.7 Qwen2.5: a modern large scale family of models

Qwen2.5 is a family of open weight dense models (0.5B - 72B) and proprietary MoE variants (Turbo/Plus) trained on ~18 Trillion tokens with staged mixtures [3]. Architectural features include a variant of multi-head attention called grouped query attention [28], for efficient KV caching, SwiGLU activations, prenorm RMS Norm, ROPE [25] with base scaling/adjustments, and QKV bias. Post training combines large-scale SFT and multistage reinforcement learning (RL) [29], yielding strong performance across reasoning, coding, multilingual tasks, and long context retrieval.

### 2.3 Bayesian Surprise and Hierarchical Models

#### 2.3.1 Bayesian surprise and predictive coding

Predictive coding models the cortex as a hierarchical generative model that minimises variational free energy F by reconciling top-down predictions with bottom-up sensory data [16, 17]. Given a prior belief p and a posterior belief q after observing new data, Bayesian surprise is canonically measured by the KL divergence $D_{KL}(q||p)$, quantifying the degree of belief update. In streaming settings, a change point (or event boundary) is suggested when surprise exceeds that expected under stationary dynamics, consistent with empirical findings in human attention and event segmentation [13, 30].

#### 2.3.2 Variational Predictive Routing (VPR)

VPR instantiates predictive-coding style inference in a hierarchical latent-variable model for sequences [14]. Let $O_{t}$ denote the observation at objective time $t\in\{1,...,T\}$. The model has N hierarchical levels. At each level $n\in\{1,...,N\}$ and time t, VPR maintains (i) a latent state $s_{t}^{n}\in\mathbb{R}^{d_{s}^{n}}$ that encodes the features represented at level n, and (ii) three deterministic context variables: $x_{t}^{n}\in\mathbb{R}^{d_{x}^{n}}$ (bottom-up encoding), $c_{t}^{n}\in\mathbb{R}^{d_{c}^{n}}$ (top-down context), $d_{t}^{n}\in\mathbb{R}^{d_{d}^{n}}$ (temporal context).

Intuitively, $x_{t}^{n}$ is the current level-wise evidence ("what is observed here?"), $c_{t}^{n}$ injects predictions from higher levels ("what does the parent expect here?"), and $d_{t}^{n}$ summarises the level's recent dynamics ("how does this feature evolve locally?").

$x_{t}^{n+1}=f_{enc}^{n}(x_{t}^{n})$, with $x_{t}^{0}\equiv o_{t}$;
$c_{t}^{n-1}=f_{dec}^{n}(s_{t}^{n},c_{t}^{n})$;
$d_{t+1}^{n}=f_{tran}^{n}(s_{t}^{n},d_{t}^{n})$

The observation is reconstructed from the top-down context at the base, $o_{t}=f_{rec}(c_{t}^{0})$.

**Two competing hypotheses at each level.** At level n and time $t+1$ the model evaluates two explanations for the new evidence $x_{t+1}^{n}$:
* **Static (no-change) hypothesis.** Re-use the most recent posterior at t as the static prior for $t+1$, $p_{st}\equiv q_{\phi}(s_{t}^{n}|x_{t}^{n},s_{<t}^{n},s_{t}^{>n})$, and form the new static posterior $q_{st}\equiv q_{\phi}(s_{t+1}^{n}|x_{t+1}^{n},s_{<t}^{n},s_{t}^{>n})$. The static surprise $D_{st}\triangleq D_{KL}(q_{st} || p_{st})$ quantifies how much the level-n features appear to have changed if we assume "no dynamics".
* **Change (predicted-change) hypothesis.** Predict the next state by advancing the temporal context, $d_{t+1}^{n}=f_{tran}^{n}(s_{t}^{n},d_{t}^{n})$, and form the change prior/posterior, $p_{ch}\equiv p_{\theta}(s_{t+1}^{n}|s_{t}^{n},s_{<t}^{n},s_{t}^{>n})$, $q_{ch}\equiv q_{\phi}(s_{t+1}^{n}|x_{t+1}^{n},s_{t}^{n},s_{<t}^{n},s_{t}^{>n})$, with change surprise $D_{ch}\triangleq D_{KL}(q_{ch}||p_{ch})$.

(Here $s_{<t}^{n}$ denotes earlier level-n states, $s_{t}^{>n}$ denotes the latest states in levels above n; all distributions are diagonal Gaussians in practice [14].)

**Event criteria and intuition.** VPR declares an event (i.e. that the level-n representation should be updated) when at least one of the following holds:
$$CE \text{ (expected change): } D_{st}>D_{ch} \quad (2.8)$$
$$CU \text{ (unexpected change): } D_{st,t+1}>\gamma\cdot\frac{1}{\tau_{w}}\sum_{k=t-\tau_{w}+1}^{t}D_{st,k} \quad (2.9)$$
CE is a model comparison: if the dynamics-aware hypothesis ("a change was expected here") explains the new evidence better than "no change", we treat it as a predictable boundary and update. CU is a simple surprise-threshold: if the instantaneous surprise $D_{st}$ exceeds its recent moving average (window $\tau_{w}\in\mathbb{N}$, threshold $\gamma>0$), we also update to capture unforeseen changes (especially important early in training before the transition model is accurate).

**Decision procedure per time step.** For $t=1,...,T-1$ and $n=1,...,N$:
1.  Bottom-up encodes new observations to level n: $x_{t+1}^{n}=f_{enc}^{n-1}(x_{t+1}^{n-1})$ with $x_{t+1}^{0}=o_{t+1}$.
2.  Form $p_{st}$, $q_{st}$ and compute $D_{st}$; predict $d_{t+1}^{n}$, form $p_{ch}$, $q_{ch}$ and compute $D_{ch}$.
3.  If (2.8) or (2.9) holds, set $s_{t+1}^{n}\sim q_{ch}$ (or $q_{st}$ when appropriate) and open the bottom-up channel to level $n+1$; otherwise block bottom-up propagation and keep $s_{t+1}^{n}=s_{t}^{n}$.

Blocking bottom-up propagation at level n also keeps all levels > n unchanged at this step, which induces nested subjective timescales: higher levels update more sparsely and learn slower-varying abstractions. Empirically, CU bootstraps learning at the start, while CE dominates once the transition model is trained; a KL decomposition shows that CE is largely governed by cross-entropy terms, consistent with its model-selection interpretation [14].

#### 2.3.3 Dynamic Latent Hierarchy (DLH)

DLH, builds on VPR, and embodies the event decision as Bayesian model selection between two prior components in a temporal mixture of Gaussians [15]. For level n, introduce a Bernoulli indicator $e_{t}^{n}\in\{0,1\}$ and define the prior over $s_{t}^{n}$ as
$$p(s_t^n | e_t^n) = \begin{cases} p(s_{t-1}^n), & e_t^n = 0 \text{ (static prior)}, \\ p_\theta(s_t^n | s_{<t}^n), & e_t^n = 1 \text{ (change prior)}. \end{cases}$$At inference, given the amortised posterior $q^{\prime}(s_{t}^{n})$, DLH selects the component that minimises KL:$$q(e_{t}^{n}=1)=1\iff D_{KL}(q^{\prime}||p_{change}^{\prime})<D_{KL}(q^{\prime}||p_{static}^{\prime})$$
DLH retains nested temporal constraints (e.g., $q(e_{t}^{n+1}=1|e_{t}^{n}=0)=0$) to promote hierarchical disentanglement. The ELBO decomposes into reconstruction, latent state KL terms against the mixture components (weighted by $q(e_{t}^{n}))$, and a Bernoulli KL for $e_{t}^{n}$, yielding an objective that clusters temporally persistent states while learning transitions only when warranted by evidence. DLH thereby achieves robust long horizon video prediction with coherent event driven rollouts and improved handling of stochastic changes.

#### 2.3.4 Neuroscience inspired ideas in Transformers

Principles from event cognition (surprise triggered boundaries, hierarchical segmentation) have informed retrieval augmented and memory augmented LLMs. For example, EM-LLM uses online surprise (token level NLL) to segment inputs into episodic events and performs two-stage retrieval similarity based plus temporally contiguous buffering aligning with human contiguity/asymmetry effects [18]. Broadly, predictive processing perspectives suggest integrating surprise driven gating or memory formation into TF computation graphs, complementing engineering driven conditional compute with semantically interpretable mechanisms.

***

## 3. Methodology

This chapter details the architectural design and training protocols for the models evaluated in this thesis. We first present the DTF architecture, which translates the principles of the VPR event-detection engine into a token-wise domain. We then formalise the core gating mechanism as a principled approximation of true Bayesian inference. Subsequently, we outline our re-implementation of the MoD model, which serves as a key computational baseline, and conclude by specifying the common training and evaluation methodologies employed.

### 3.1 The Dynamic Transformer (DTF) Architecture

The DTF architecture modifies a pre-trained, decoder-only TF by replacing its standard stack of uniform decoder layers with a sequence of alternating **Decision Layers** and **Dynamic Layers**, as depicted in Figure 3.1. This structure is an architectural translation of the VPR event-detection engine: the Decision Layer computes the necessary signals to evaluate the static and change hypotheses for each token, before the Dynamic Layer's Predictive Router executes the comparison and conditionally applies further computation.

Operationally, the Decision Layer augments a standard TF block with a lightweight parallel predictor Prior Feed-Forward Network (PriorFFN) (see Figure 3.1, right; green nodes). Given the layer input $H^{(l)}$, it produces three representations per token: the input itself $H_{orig}^{(l)}=H^{(l)}$, the actual block output $H_{post}^{(l)}=H^{(l)}+TF-Block(H^{(l)})$, and a cheap prediction of that output $H_{prior}^{(l)}=H^{(l)}+PriorFFN(RMSNorm(H^{(l)}))$. These views are compared to quantify, for each token, how much its representation changed and whether the change is better explained by the learned predictor than by a no-change hypothesis, thereby providing the sufficient statistics for the subsequent routing decision.

The Dynamic Layer converts these comparisons into a scalar routing score $g_i$ per token and, under a fixed capacity $\gamma$, selects the top $k=\lfloor\gamma T\rfloor$ tokens for further computation (see Figure 3.1, right; yellow nodes). Only the selected tokens are processed by a second TF block and attend to other selected tokens at that depth; the remaining tokens take the residual path unchanged. The sequence is then remerged and passed to the next Decision Layer. This mechanism is analogous to capacity-limited routing in MoD, but the selection criterion is driven by a surprise-based model-comparison signal derived from the Decision Layer rather than a generic importance score. Finally, we note that the terms "prior" and "posterior" are used here to align with the predictive-coding background: in this work they denote deterministic vector representations, not probability distributions.

**Figure 3.1:** High-level comparison of a standard TF (left) and the proposed DTF (right). The DTF replaces the uniform stack with alternating Decision (green) and Dynamic (yellow) layers.

### 3.1.1 The Decision Layer and Prior

The Decision Layer is designed to generate, in parallel, the three representations required by the Predictive Router to perform its hypothesis test. For an input hidden state tensor $H^{(l)}\in\mathbb{R}^{B\times T\times d}$ at layer l, the layer computes:

* **Original State ($H_{orig}^{(l)}$):** The unmodified input to the layer, serving as the baseline for the static hypothesis.
    $$H_{orig}^{(l)}=H^{(l)} \quad (3.1)$$
* **Posterior State ($H_{post}^{(l)}$):** The output of a standard, dense TF block. This represents the "ground truth" updated representation for the layer, against which both hypotheses are evaluated.
    $$H_{post}^{(l)}=H^{(l)}+TF-Block(H^{(l)}) \quad (3.2)$$
* **Prior State ($H_{prior}^{(l)}$):** A computationally inexpensive prediction of the posterior state, serving as the change hypothesis. It is generated by a small, learnable PriorFFN.
    $$H_{prior}^{(l)} = H^{(l)} + PriorFFN (RMSNorm (H^{(l)})) \quad (3.3)$$

**The PriorFFN as a Predictive Model.** The PriorFFN is the architectural embodiment of the "generative model" in predictive-coding terms: it aims to learn a low-cost approximation of the full TF block's transformation. We optimise it with an auxiliary loss $\mathcal{L}_{prior}$ that minimises the mean-squared error (MSE) between the PriorFFN prediction and the block output (with a stop-gradient on the target to avoid leaking gradients into the main block):
$$\mathcal{L}_{prior}=MSE(H_{prior}^{(l)}, \text{stop\_gradient}(H_{post}^{(l)})) \quad (3.4)$$

The overall training objective augments the language-modelling loss with a small weighted sum of the prior losses across Decision Layers D,
$$\mathcal{L}=\mathcal{L}_{LM}+\frac{\lambda_{prior}}{|\mathcal{D}|}\sum_{l\in\mathcal{D}}\mathcal{L}_{prior}^{(l)}; \quad \lambda_{prior}=0.05 \quad (3.5)$$
which in practice contributed less than 1% of the total loss throughout training. This weak, consistency-style regularisation provides a soft supervision signal that aligns each PriorFFN with its corresponding block while keeping the language-modelling objective dominant. In ablations without this term the prior prediction error grew rapidly, the CE/CU calibration deteriorated, and language-modelling performance degraded over time, with perplexity approaching near-random baselines.

The architecture of the PriorFFN follows that of the FFN from the original TF block but with a severely reduced intermediate size. We study the effect of this in Section 4.3. In the case of Qwen2.5 [3] this is a SwiGLU architecture. Let the input vector be $x\in\mathbb{R}^{d_{h}}$, where $d_{h}$ is the hidden size. The intermediate dimension $d_{i}$ is dynamically calculated based on an intermediate size factor $f\in\mathbb{R}$:
$$d_i = \max(2, \lceil d_h f \rceil + (\lceil d_h f \rceil \bmod 2))$$
By construction, $d_{i}\ge2$ and is even (we round up $d_{h}f$ and add one only if it is odd). Enforcing even widths improves kernel tiling and memory alignment on common accelerators, which typically yields more stable and efficient low-level kernels [11].

Consistent with the Qwen2.5 series of models [3] used in this work, we instantiate the PriorFFN with a SwiGLU nonlinearity. Given an input $x\in\mathbb{R}^{d_{h}}$ (after RMS normalisation upstream), the forward pass is
$$PriorFFN(x)=Dropout((SiLU(xW_{1})\odot xW_{3})W_{2})$$
where $W_{1}, W_{3}\in\mathbb{R}^{d_{h}\times d_{i}}$ are the input and gate projections, $W_{2}\in\mathbb{R}^{d_{i}\times d_{h}}$ projects back to the hidden size, $SiLU(y)=y~\sigma(y)$ with $\sigma(y)=1/(1+e^{-y})$, and $\odot$ denotes the Hadamard product. All weight matrices are initialised with $\mathcal{N}(0,0.02^{2})$ and we omit biases. This realises a gated FFN with SwiGLU activation-matching the feed-forward design of Qwen2.5, while keeping the intermediate width $d_{i}$ small for efficiency.

### 3.1.2 The Dynamic Layer and Predictive Router

The Dynamic Layer receives the posterior state $H_{post}^{(l)}$ from the Decision Layer and conditionally applies a second TF block to a subset of tokens. The selection is governed by the Predictive Router, which implements the core VPR based logic.

**Surprise Calculation.** The router first computes two token-wise "surprise" metrics using Mean Squared Error (MSE) as a computationally efficient proxy for KL divergence, where the error is normalised by d, the hidden size of the model. For each token i:
* **Static Surprise ($D_{st,i}$):** Quantifies the magnitude of the representational update under the static hypothesis.
    $$D_{st,i}=\frac{1}{d}||H_{post,i}^{(l)}-H_{orig,i}^{(l)}||_{2}^{2} \quad (3.6)$$
* **Change Surprise ($D_{ch,i}$):** Measures the squared error of the PriorFFN's prediction under the change hypothesis.
    $$D_{ch,i}=\frac{1}{d}||H_{post,i}^{(l)}-H_{prior,i}^{(l)}||_{2}^{2} \quad (3.7)$$

This use of MSE is a principled approximation, as under the assumption of whitened, isotropic Gaussian distributions over hidden states, the KL divergence is proportional to the squared Euclidean distance between the means [31].

**Gating Criteria.** The router applies two criteria, translated from VPR, to determine if a token constitutes an "event". As you will notice instead of using hard gating decisions based on if $D_{st,i}>D_{ch,i}$ and $D_{st,i}>m_{cu}\cdot MA(D_{st,i})$, for the corresponding expected event criterion (CE) and unexpected event criterion (CU), we use a soft gating version, via subtraction. The main benefits of this soft gating are that since these equations are all differentiable we can allow gradient signals to feed back to the PriorFFN and the Predictive Router, allowing the DTF to learn emergent properties.
* **Criterion E (Expected):** A token is an expected event if the PriorFFN's prediction is a better fit for the posterior state than the original state. This comparison is modulated by a learnable offset, $o_{ce}\in\mathbb{R}$ aiming to provide a small bias supporting CE, to help the PriorFFN. The log term is an ad-hoc engineering based solution to help performance of the model by adding a small bias to help the PriorFFN. We initialise $o_{ce}=1.025$ and use $\epsilon=1e-10$.
    $$CE_{i}=D_{st,i}-(D_{ch,i}-log(o_{ce}+\epsilon)) \quad (3.8)$$
* **Criterion U (Unexpected):** A token is an unexpected event if its static surprise significantly exceeds its recent history. This is implemented using an Moving Average (MA) of the static surprise over the last 100 tokens, scaled by a learnable multiplier, $m_{cu}\in\mathbb{R}$. Taking inspiration from the original VPR paper, we initialise $m_{cu}=1.1$. This MA function is the same as used in VPR.
    $$CU_{i}=D_{st,i}-(m_{cu}\cdot MA(D_{st,i})) \quad (3.9)$$

**Differentiable Routing and State Update.** To ensure end-to-end differentiability, the hard comparisons used in VPR are reformulated into a soft, probabilistic mechanism. The raw scalar outputs from the criteria are transformed into probabilities using scaled sigmoid functions, where the inverse temperatures $\beta_{ce}$ and $\beta_{cu}$ are learnable parameters that allow the model to control the sharpness of the gating decision. We initialise these to $\beta_{ce}=-0.3$ and $\beta_{cu}=-0.6$. As opposed to VPR, we have made the predictive router's parameters learnable, which is a theoretical shift from VPR as it was non-parametric. In order to stabilise training and force the inverse temperatures to be positive numbers we wrap the inverse temperatures in a $Softplus(x)=log(1+exp(x))$, enforcing that $\beta_{ce}>0$ and $\beta_{cu}>0$.
$$S_{CE}=\sigma(Softplus(\beta_{ce})\cdot CE) \quad (3.10)$$
$$S_{CU}=\sigma(Softplus(\beta_{cu})\cdot CU) \quad (3.11)$$

These probabilities are combined using a probabilistic OR to form a continuous gating signal $G_{cont}\in[0,1]^{T}$ for each token:
$$G_{cont}=S_{CE}+S_{CU}-(S_{CE}\cdot S_{CU}) \quad (3.12)$$

A vector of these gating decisions, $G_{bin}$, is derived by selecting the tokens with the highest $G_{cont}$ scores (via a TopK), up to a pre-defined capacity $\gamma\cdot T$, similarly to how MoD's routing mechanism works. Let S be the set of selected token indices. The final state $H^{(l+1)}$ is then computed via a gated residual connection. For selected tokens $(i\in\mathcal{S})$, a second TF block is applied; for non-selected tokens, the state passes through unchanged, an identity operation.
$$H_{i}^{(l+1)}=\begin{cases}H_{post,i}^{(l)}+G_{cont,i}\cdot TF\cdot Block(H_{post,i}^{(l)}) & \text{if } i\in\mathcal{S} \\ H_{post,i}^{(l)} & \text{if } i\notin\mathcal{S}\end{cases} \quad (3.13)$$

### 3.2 A Bayesian Interpretation of the Predictive Router

The DTF's routing mechanism, while neuroscientifically inspired, can be formally interpreted as an act of Bayesian Model Comparison [31]. For each token, the router implicitly evaluates the evidence for two competing models seeking to explain the final state, $H_{post}$:
* **Model 1 (The Static Model):** Posits that $H_{post}$ is sufficiently explained by its initial state, $H_{orig}$. The negative log evidence for this model is approximated by the static surprise, $D_{st}$.
* **Model 2 (The Change Model):** Posits that $H_{post}$ is better explained by the dynamic prediction generated by the PriorFFN, $H_{prior}$. The negative log evidence for this model is approximated by the change surprise, $D_{ch}$.

The router's Criterion E, which gates tokens when $D_{st}>D_{ch}$ (ignoring the offset for clarity), is therefore approximately equivalent to selecting the model with the higher marginal likelihood. The decision to route a token for further computation is a direct implementation of this principle: it selects the "Change Model" precisely when that model provides a better explanation for the observed representational update.

### 3.3 The MoD Baseline Architecture

For a direct comparison, we re-implemented a standard MoD model [10]. This architecture modifies a pre-trained TF, also Qwen2.5 based [3], by replacing every other decoder layer with an MoD Layer. Each MoD layer employs a simple, learnable linear router to assign an importance score to each token. It then processes only the TopK scoring tokens (where $k=\gamma\cdot T$ is a fixed capacity) through its TF block, while the remaining tokens bypass the layer via a residual connection. This provides a strong, hardware-efficient baseline for token-based conditional computation. To ensure a fair comparison with the DTF architecture, which does not currently implement a causal router for inference, all MoD experiments in this work utilise the non-causal, training-time TopK routing mechanism rather than its causal counterpart.

### 3.4 Training and Benchmarking

The training strategy was designed for consistency across all model variants and scales, with a focus on efficiently adapting a large, pre-trained decoder-only TF model.

#### 3.4.1 Optimisation

We train with AdamW [32, 33], using decoupled weight decay, a linear warm-up schedule with cosine annealing, as done in the original MoD paper [10]. Mixed precision (FP16/BF16) and activation checkpointing reduce memory. Where applicable, LORA [27] enables parameter efficient finetuning. Dropout and stochastic depth (drop path) regularise deep stacks [34]. For long context models, kernel level optimisations such as FlashAttention/FlashAttention2 [11, 26] are employed. Implementations use PyTorch [35] for model definitions, tokenisation, dataset streaming, and distributed training. All experiments were run with a global random seed of 42, a sequence length of 1024, a per-device batch size of 16, and gradient accumulation over 64 steps, yielding an effective batch size of 1024.

#### 3.4.2 Pre-training corpora (this work)

We construct a mixed domain corpus to preserve broad linguistic competence while also strengthening reasoning capabilities. All datasets are publicly available on the Hugging Face Hub and are detailed in Table 3.1. Sampling ratios and mixing weights are tuned to balance stability (prevent catastrophic forgetting) and downstream reasoning performance.

**Table 3.1:** Pre-training data mixture used for fine-tuning. All datasets are available on the Hugging Face Hub. Ratios refer to the subset of the original training split used.

| Dataset Name             | Ratio | Tokens (M) | Description             |
| ------------------------ | ----- | ---------- | ----------------------- |
| wikitext-103-raw-v1      | 1.0   | 103        | Encyclopedic text       |
| cnn\_dailymail (3.0.0)   | 0.2   | 43         | High-quality news       |
| storytracer/US-PD-Books  | 0.5   | 650        | Classic literature      |
| HuggingFaceTB/cosmopedia | 0.1   | 2500       | Synthetic textbooks     |
| sciq                     | 1.0   | 3          | Factual science QA      |
| codeparrot-clean-valid   | 1.0   | ~18        | Cleaned Python code     |
| roneneldan/TinyStories   | 1.0   | 525        | Simple causality stories |

#### 3.4.3 Benchmarks

We evaluate on widely used benchmarks that test general knowledge and reasoning:
* MMLU [36] (multitask knowledge and reasoning)
* ARCChallenge [37] (complex science question answering)
* HellaSwag [38] (commonsense sentence completion)
* WinoGrande [39] (commonsense pronoun resolution)
* TruthfulQA [40] (evaluating model truthfulness)

**Model Adaptation from Pre-trained Checkpoints.** The DTF and MoD architectures are instantiated by converting a pre-trained, dense TF model, Qwen2.5 0.5B [3]. This is achieved by first initialising our custom model with its specified layer structure, and then systematically copying the weights from the corresponding layers of the pre-trained model. Newly introduced components, such as the PriorFFN and the routers, are randomly initialised. This transfer-learning approach leverages the powerful, pre-existing representations of the dense TF, ideally accelerating convergence.

**Optimisation Strategy.** All models are trained using the AdamW optimiser [33] with a weight decay of 0.01. To ensure training stability and effective adaptation, we employ a differential learning rate strategy for three distinct parameter groups:
1.  **Base Model:** The weights of the original TF blocks are fine-tuned with a conservative learning rate to prevent catastrophic forgetting of pre-trained knowledge, learning rate $=1.0\times10^{-5}$.
2.  **PriorFFN:** This network is trained with a moderately higher learning rate to encourage rapid learning of its predictive function, learning rate $=1.0\times10^{-3}$.
3.  **Predictive Router:** The four learnable parameters of the router are trained with the highest learning rate, allowing the gating policy to adapt quickly to the fine-tuning data, learning rate $=1.0\times10^{-2}$.

For our parameter-efficient adaptation experiments, we employed LoRA with a rank (r) of 16, a scaling factor ($\alpha$) of 32, and a dropout of 0.05. The adapters were applied comprehensively to all linear projections within the base TF blocks (query, key, value, output, and MLP). Crucially, the newly introduced dynamic components, the Predictive Router and the PriorFFN, were exempted from LORA and were instead fully fine-tuned to ensure they had sufficient capacity to learn the routing policy. In this setup, to speed up convergence, we used a learning rate of $1.0\times10^{-4}$ for our base model, all else being equal.

A cosine decay learning rate schedule with a linear warm-up phase, covering 1% of total training steps, is used for all parameter groups.

**Implementation and Scalability.** Models are implemented in PyTorch [35]. To facilitate training at larger scales, we utilise the Accelerate library for distributed training, employing Fully Sharded Data Parallel (FSDP) with CPU offloading and 'bfloat16' mixed precision to ensure memory efficiency.

***

## 4. Results and Analysis

### 4.1 Experimental Setup

Our experiments compare architectures adapted from a pre-trained Qwen2.5-0.5B decoder-only TF [3]. The model classes are:
* **Baseline TF:** The standard, dense pre-trained model Qwen2.5-0.5B.
* **MoD:** Our re-implementation of the MoD architecture [10], serving as the engineering-driven baseline.
* **DTF:** Our novel VPR-inspired dynamic architecture.

All models were fine-tuned on a supervised pre-training dataset mixture. Performance was evaluated on a suite of standard language understanding benchmarks: ARC-C [37], HellaSwag [38], MMLU [36], TruthfulQA [40], and WinoGrande [39]. Internal model states were logged throughout training to facilitate detailed analysis. Unless otherwise stated, all scalar training curves (loss, perplexity, prior loss) are displayed after smoothing with an exponential moving average over the last 15 steps to better convey the average trend. Note also that all learned parameters of the Predictive Router are also displayed after having been averaged across the Dynamic Layers. This also applying to the prior loss of the Decision Layers.

### 4.2 Primary Architecture Comparison: DTF vs. MoD

The first set of experiments directly compares the learning dynamics and generalisation performance of the DTF and MoD architectures. Both models were configured with a routing capacity of $\gamma=0.5$, processing 50% of tokens in their respective Dynamic / MOD layers. This corresponds to a significant reduction in the theoretical FLOPs per forward pass compared to the dense baseline.

**Training and Validation Dynamics.** Figure 4.1 illustrates the comparative training dynamics. The smoothed training loss and perplexity curves for both Dynamic / MoD architectures are closely matched, indicating that both models learn effectively and maintain stable training profiles despite the significant reduction in computation. In our runs, DTF occasionally shows a slightly lower training loss and perplexity during the early to mid-stages.

**Figure 4.1:** Training loss and perplexity learning dynamics for DTF and MoD at $\gamma=0.5$. Curves are smoothed with an exponential moving average over the last 15 steps.

Turning to validation, Figure 4.2 shows that the DTF model achieves a consistently lower validation loss than MoD throughout most of the training. While the absolute gap is modest, this pattern is stable in our runs, and also showed up in smaller runs done in preliminary testing.

**Figure 4.2:** Validation loss (log-scale; EMA, 15 steps) for DTF and MoD at $\gamma=0.5$.

**Interpretation of Results.** The lower validation loss of DTF provides direct empirical support for our central hypothesis that surprise-based routing can offer an advantage over an importance-based heuristic at matched capacity. This observation is consistent with the intuition that a context-dependent comparison of predictive hypotheses (static vs. change) can yield slightly better calibration of which tokens warrant deeper processing. At the same time, differences are not large and should be interpreted cautiously.

**Benchmark Performance.** Table 4.1 summarises performance on the downstream benchmarks. DTF slightly outperforms MoD on three of the five tasks, which is consistent with the validation-loss trend. However, both variants are substantially below the dense baseline across most tasks at $\gamma=0.5$; this is expected in our transfer-learning setting with reduced token-level compute and limited adaptation, and we do not interpret these numbers as the capacity ceiling of dynamic models.

**Table 4.1:** Benchmark performance for 0.5B parameter models. Scores are accuracy (%). All dynamic models operate at a capacity of $\gamma=0.5$. Bold indicates the best performance among the dynamic variants.

| Model Variant     | ARC-C | HellaSwag | MMLU | TruthfulQA | WinoGrande |
| ----------------- | ----- | --------- | ---- | ---------- | ---------- |
| Number of Shots   | 25    | 10        | 5    | 0          | 5          |
| Baseline TF       | 43.7  | 52.1      | 55.9 | 40.2       | 56.3       |
| MoD               | 24.3  | 32.6      | 23.3 | 43.5       | **52.6** |
| DTF               | **25.9** | **33.3** | **24.4** | **44.5** | 50.7       |
| Random Baseline   | ~25   | 25        | 25   | ~23        | 50         |

### 4.3 Impact of Prior Network Expressivity

This study investigates how the performance of the DTF is affected by the size, and therefore the expressivity, of the PriorFFN, which serves as the architecture's internal predictive model.

**Hypothesis.** The performance of the prediction-based gating (Criterion E) is sensitive to the fidelity of the PriorFFN's predictions. A larger, more expressive PriorFFN is hypothesised to learn a more accurate predictive model, leading to more effective routing and better overall model performance, up to a point of diminishing returns.

**Methodology and Results.** To test this hypothesis, we trained four DTF models, varying the PriorFFN's intermediate size (as defined in Section 3.1.1) via an intermediate size factor hyperparameter, with values of 0.0625, 0.125, 0.25, and 0.5. As shown in Figure 4.3, the main task's training loss and perplexity curves are remarkably insensitive to the PriorFFN's size. The learning dynamics for all configurations are nearly indistinguishable, indicating that the choice of prior capacity has a negligible impact on the overall optimisation of the main language modelling task in this fine-tuning context.

[Image showing effect of prior factor on training loss and perplexity]
**Figure 4.3:** Effect of PriorFFN size factor on training dynamics. The learning curves are largely insensitive to the prior's capacity. (a) Training loss (EMA, 15 steps). (b) Training perplexity (EMA, 15 steps).

The core trade-off becomes apparent when examining the auxiliary prior loss against the final validation loss, as shown in Figure 4.4. As hypothesised, Figure 4.4(a) confirms that larger, more expressive PriorFFNs achieve a lower auxiliary loss, learning more accurate predictions of the main block's output. However, Figure 4.4(b) reveals that this improved prediction fidelity does not translate to better downstream generalisation; the model with the smallest PriorFFN $(factor=0.0625)$ achieves a marginally lower final validation loss. This suggests that a lightweight prior is not only sufficient but may also confer a slight regularisation benefit, preventing overfitting to the specific predictive patterns of the main block during fine-tuning.

[Image showing effect of prior factor on prior network auxiliary loss and validation loss]
**Figure 4.4:** Effect of PriorFFN size factor on auxiliary prior loss and validation loss. Larger priors fit the posterior more accurately (a), yet the smallest prior yields the lowest final validation loss (b). Curves are smoothed with an exponential moving average over the last 15 steps.

**Analysis of Internal Dynamics.** To understand this outcome, we analysed the internal routing mechanism. The Predictive Router adapts its learned parameters based on the prior's expressivity (Figure 4.5). Notably, from Figure 4.5(c), the prediction offset $o_{CE}$ converges to a lower value for models with larger (more accurate) priors, suggesting the router learns to rely less on this additive bias when the prior's predictions are more reliable. Conversely, the novelty multiplier $m_{CU}$ (Figure 4.5(d)) converges to a higher value for larger priors. This implies that for models with weaker priors, the novelty-based Criterion U is more influential in the gating decision, indicating a greater dependence on the raw surprise signal.

**Figure 4.5:** Evolution of the Predictive Router's parameters across prior sizes, showing distinct learned policies. (a) Expected-event temperature $\beta_{CE}$. (b) Unexpected-event temperature $\beta_{CU}$. (c) Prediction offset $o_{CE}$. (d) Novelty multiplier $m_{CU}$.

Despite these distinct learned policies, the emergent gating signals themselves remain remarkably stable across all configurations, as shown in Figure 4.6. In all four cases, the mean activation of the prediction-based signal $S_{CE}$ (pink) rises during training to dominate the novelty-based signal $S_{CU}$ (cyan). This demonstrates a consistent and principled transition from a novelty-driven routing policy early in training to a more refined, prediction-driven policy as the PriorFFN becomes more accurate. Furthermore, the final combined gate activation $G_{cont}$ (the unique colour based on the prior factor) converges to a similar level across all prior sizes, reinforcing that the routing mechanism successfully maintains a stable compute budget at the fixed capacity of $\gamma=0.5$.

[Image analyzing gating signal components by prior factor]
**Figure 4.6:** Combined gating signal components ($S_{CE}$, $S_{CU}$, and $G_{cont}$) by prior factor over training. Each panel corresponds to one prior size (top-left: 0.0625, top-right: 0.125, bottom-left: 0.25, bottom-right: 0.5). In every panel, the pink trace is $S_{CE}$, the cyan trace is $S_{CU}$, and the unique, for each prior factor size, colour trace in the top is $G_{cont}$. The model consistently shifts emphasis to the CE signal over training. Mean activations remain broadly similar across prior sizes, consistent with a stable routing budget.

**Benchmark Performance and Interpretation.** Table 4.2 shows the downstream benchmark performance. While these results allow for a nuanced comparison between DTF variants, it is important to reiterate that all dynamic configurations substantially underperform the dense baseline TF model (Table 4.1), an expected outcome given the reduced computational budget and limited adaptation. Among the DTF variants, no single prior size consistently dominates across all tasks, and the performance differences are small. This reinforces the finding from the validation loss: prior expressivity is not a critical hyperparameter in this setting. A lightweight prior is sufficient to provide an effective and regularising signal for the Predictive Router, highlighting the robustness of the DTF architecture.

**Table 4.2:** Impact of PriorFFN expressivity on benchmark performance for 0.5B DTF models. All models operate at a capacity of $\gamma=0.5$. Scores are accuracy (%). Bold indicates the best performance for each benchmark.

| Model Variant        | ARC-C      | HellaSwag | MMLU      | TruthfulQA | WinoGrande |
| -------------------- | ---------- | --------- | --------- | ---------- | ---------- |
| Number of Shots      | 25         | 10        | 5         | 0          | 5          |
| Baseline TF          | 43.7       | 52.1      | 55.9      | 40.2       | 56.3       |
| DTF (Prior=0.0625)   | 25.9       | **33.3** | **24.4** | 44.5       | 50.7       |
| DTF (Prior=0.125)    | 26.0       | 32.9      | 24.1      | 44.5       | **51.9** |
| DTF (Prior=0.25)     | 25.3       | 33.2      | 23.6      | **45.6** | 48.8       |
| DTF (Prior=0.5)      | **26.3** | 33.2      | 24.1      | 45.1       | 50.4       |
| Random Baseline      | ~25        | 25        | 25        | ~23        | 50         |

### 4.4 Ablation Study: Parameter-Efficient Adaptation

This study investigates the trade-off between performance and parameter efficiency when adapting the pre-trained base model, comparing a full fine-tuning approach with a parameter-efficient alternative.

**Hypothesis.** A highly parameter-efficient adaptation method, such as Low-Rank Adaptation (LORA) [27], can achieve competitive performance relative to full fine-tuning of the dynamic components, but with significantly fewer trainable parameters, presenting a clear efficiency-versus-accuracy trade-off.

**Methodology and Results.** We compare two adaptation strategies: Finetune, which involves training the full weights of the newly introduced dynamic components (PriorFFN and Predictive Router), and LoRA, which freezes the base model and applies low-rank adapters to its linear layers while still fully fine-tuning the dynamic components. Figure 4.7 presents the core performance results. While both methods show stable convergence, the full fine-tuning approach achieves a lower final validation loss, indicating superior generalisation. This is consistent with the PriorFFN's auxiliary loss (Figure 4.8), where the fully fine-tuned network learns a more accurate predictive model of the main block's outputs.

For our parameter-efficient experiments, we employed LoRA with a rank (r) of 16, an alpha of 32, and a dropout rate of 0.05. The LoRA adapters were applied comprehensively to the query, key, value, output, and MLP projections within the base TF blocks. The dynamic components, the Predictive Router and PriorFFN, were exempted from LORA and fully fine-tuned in both setups to ensure they had sufficient capacity to learn the routing policy.

[Image comparing Finetune and LoRA training loss and perplexity]
**Figure 4.7:** Training loss and perplexity for Full Fine-tuning vs. LoRA adaptation methods. Full fine-tuning achieves a lower final training loss generally, indicating better generalisation. Curves are smoothed with an exponential moving average over the last 15 steps. (a) Training Loss (b) Training Perplexity

[Image comparing Finetune and LoRA validation loss and prior auxiliary loss]
**Figure 4.8:** Auxiliary loss for the PriorFFN under Full Fine-tuning and LoRA. The fully fine-tuned PriorFFN learns a more accurate prediction of the main block's output. Auxiliary loss curves are smoothed with an exponential moving average over the last 15 steps. (a) Validation Loss (b) Auxiliary Prior Loss

**Analysis of Internal Dynamics.** To understand how these different adaptation methods affect the routing mechanism, we analyse the evolution of the router's internal state. Figure 4.9 shows that both methods learn stable but distinct policies for the four learnable router parameters. For example, the LORA adapted model learns a higher prediction offset $(o_{CE})$ and a sharper gating temperature $(\beta_{CE})$, suggesting it relies on a different decision boundary. Despite these policy differences, the emergent gating signals, shown in Figure 4.10, exhibit similar high-level dynamics. In both cases, the system transitions from a novelty-driven state (higher initial $S_{CU}$) to a prediction-driven one (rising $S_{CE}$), consistent with the principles of Predictive Coding, this phenomenon is even more apparent in the LoRA adapted model.

[Image showing evolution of router parameters for Finetune vs LoRA]
**Figure 4.9:** Evolution of the four learnable parameters of the DTF's Predictive Router for Full Fine-tuning vs. LoRA adaptation. Both methods learn stable but distinct routing policies.

[Image showing analysis of gating signal components for Finetune vs LoRA]
**Figure 4.10:** Analysis of gating signal components for Finetune (left) and LoRA (right) adaptation. The red/purple trace is the final gate $G_{cont}$, pink is $S_{CE}$, and cyan is $S_{CU}$. Both methods exhibit the characteristic transition from novelty-based routing (higher $S_{CU}$) to prediction-based routing (higher $S_{CE}$) as training progresses.

**Benchmark Performance and Interpretation.** The results in Table 4.3 indicate a clear trade-off. While LoRA provides a significant reduction in trainable parameters, making adaptation faster and more memory-efficient, it comes at the cost of a discernible performance degradation on several benchmarks like HellaSwag and WinoGrande. The higher validation and prior loss suggest that the low-rank approximation may constrain the PriorFFN's ability to fully capture the complexity of the TF block's function, leading to a sub-optimal, albeit stable, routing policy. For applications where maximum performance is critical, full fine-tuning of the dynamic components appears to be the superior strategy, though LoRA remains a viable and competitive option where parameter efficiency is paramount. As before, both DTF / MOD methods underperform the dense baseline, highlighting the inherent challenge of conditional computation in a limited adaptation setting.

**Table 4.3:** Benchmark performance comparison between a fully fine-tuned DTF (Prior=0.0625) and a parameter-efficient DTF adapted with LoRA. Both models operate at a capacity of $\gamma=0.5$. Scores are accuracy (%). Bold indicates the better performance for each benchmark.

| Model Variant     | ARC-C    | HellaSwag | MMLU    | TruthfulQA | WinoGrande |
| ----------------- | -------- | --------- | ------- | ---------- | ---------- |
| Number of Shots   | 25       | 10        | 5       | 0          | 5          |
| Baseline TF       | 43.7     | 52.1      | 55.9    | 40.2       | 56.3       |
| DTF (Finetune)    | 25.9     | **33.3** | **24.4**| 44.5       | **50.7** |
| DTF (LORA)        | **27.2** | 26.6      | 24.1    | **48.3** | 48.7       |
| Random Baseline   | ~25      | 25        | 25      | ~23        | 50         |

***

## 5. Discussion

This chapter interprets the empirical results of Chapter 4 in the light of the research questions stated in Chapter 1, situates them within the broader literature on conditional computation and predictive coding, and articulates the principal limitations and implications of the present work. We begin with a synthesis of the main findings and how they relate to our stated hypotheses, then analyse their agreement with the theoretical expectations VPR and DLH [14, 15], before turning to a critical evaluation of limitations, and directions for future work.

### 5.1 Summary of findings at fixed capacity

Under a fixed token capacity of $\gamma=0.5$, both DTF and MoD trained stably from a pre-trained Qwen2.5-0.5B initialisation with similar training losses and perplexities, and DTF consistently attained a lower validation loss than MoD (Chapter 4). This result provides some support for the hypothesis that a surprise-based, model-comparison criterion (DTF) can yield a modest advantage over a learned importance score (MoD) when the two are compared under matched compute and training conditions. On downstream multiple-choice benchmarks, DTF outperformed MoD on three of five tasks, broadly mirroring the validation-loss pattern; however, both variants were substantially below the dense baseline across most tasks at $\gamma=0.5$ (e.g., MMLU and ARC-C were close to random-choice baselines), which we discuss below.

Two ablations shed light on design choices. First, varying the PriorFFN size over a wide range (size factor {0.0625, 0.125, 0.25, 0.5}) showed that larger priors achieved lower auxiliary prior loss (better predictions of the block output), but did not improve validation loss; the smallest prior slightly improved generalisation. Secondly, a parameter-efficient adaptation with LoRA preserved the qualitative routing dynamics but increased validation loss relative to full fine-tuning of the dynamic components. These observations suggest that (i) a lightweight prior suffices to induce a useful surprise signal, acting as a mild regulariser, and (ii) when maximum accuracy is required, full fine-tuning of the dynamic components is preferable, whereas LoRA remains attractive where parameter efficiency is the overriding constraint.

We stress that the observed improvements in validation loss were obtained in a single, specific setting (one model scale; one capacity; one seed). We did not run statistical significance tests, so any small differences should be interpreted as indicative rather than conclusive.

### 5.2 Agreement with predictive-coding theory

A central aim of this work was to assess whether a surprise-based gating mechanism inspired by predictive-coding theory [16, 17] can be translated effectively into a token-wise TF setting. We find several qualitative points of agreement with VPR/DLH:

**CE/CU dynamics.** In VPR, the CE (expected change) criterion becomes dominant once the transition model becomes accurate, whereas CU (unexpected change) is most useful early on when the model is miscalibrated [14]. In our runs, we observe the same qualitative transition: the novelty-driven signal $S_{CU}$ is initially higher, but as the PriorFFN learns, $S_{CE}$ rises and dominates the combined gate $G_{cont}$. This shift is visible across prior sizes and adaptation methods (full fine-tuning vs. LoRA), indicating robustness of the mechanism.

**Stable routing budget at fixed capacity.** Across prior sizes, mean gating activation (and the final gate $G_{cont}$) converges to a similar level, consistent with a stable routing budget under fixed $\gamma$. This mirrors the "nested timescales" intent in VPR/DLH: once calibration improves, the model opens gates selectively where predicted change explains the evidence better than no-change, while closing gates elsewhere.

**Model-comparison interpretation.** The CE decision in DTF operationalises the same model-comparison principle as VPR/DLH: route when the change model (prior prediction) explains the new state better than the static model (original state). Our use of MSE distances as proxies for $D_{KL}$ under a whitened-Gaussian approximation [31] preserves the directionality of the decision and provides a differentiable signal that couples naturally to the PriorFFN (Section 3.1). Empirically, adding the auxiliary prior loss stabilised CE/CU calibration; removing it quickly led to drift and collapse of routing decisions.

**Layer-depth as representational granularity.** We have implicitly mapped VPR's temporal hierarchy onto layer-wise representational abstraction: lower layers are assumed to capture more local, syntactic regularities while higher layers capture more semantic, long-range structure. Although we did not vary capacity by depth in this study, the observed CE-dominant dynamics and stable budget are consistent with this mapping; a depth-wise capacity schedule remains an important avenue for future work.

Overall, while our study is limited in scope (Section 5.4), the qualitative agreement between the observed routing dynamics and VPR/DLH's theoretical expectations strengthens the case that surprise-based gating is a viable inductive bias for token-wise conditional computation.

### 5.3 Why do our models underperform the baseline?

Both DTF and MoD underperform a dense baseline by a large margin on several benchmarks at $\gamma=0.5$ (Table 4.1). This result is not unexpected in the present transfer-learning regime and we highlight several contributing factors:

**Fixed token capacity halves token-level compute.** With $\gamma=0.5$, only half the tokens receive the full block computation in each dynamic layer. In MoD this is by design [10], and in DTF selection is based on model-comparison signals. Under a short adaptation window, reduced per-token compute can harm absolute downstream accuracy even if validation loss improves slightly during fine-tuning. In other words, our study prioritised relative routing quality over absolute accuracy.

**Transfer-learning constraints and limited adaptation.** We introduced new routing parameters and PriorFFNs on top of a pre-trained dense model, and fine-tuned under a small budget (fixed seed, fixed schedule, limited data). Neither dynamic variant was trained from scratch. Prior work suggests conditional compute benefits often increase with pre-training scale and budget (e.g., Switch/MoE [12, 6, 7] and MoD [10]).

**Possible data drift / domain mismatch.** Several of our benchmark scores are near random-choice (e.g., ARC-C and MMLU), for both variants. This could indicate: (i) true failure to adapt effectively with conditional compute, (ii) undertraining at this scale, and/or (iii) mismatch between the (undisclosed) Qwen2.5 pre-training distribution and our fine-tuning mixture (Section 3.4). Because Qwen's pre-training data are not public, we cannot precisely align the adaptation corpus to the original distribution; for that reason data drift remains a plausible factor. The fact that MoD exhibits similar degradation under the same regimen supports the view that this is not specific to DTF.

**Model scale.** At 0.5B parameters, the backbone is small by modern LLM standards. It is plausible that the benefits of skipping computation accrue more at larger scales (where dense compute is redundant for many tokens) and that small models are comparatively under-parameterised to compensate for reduced token-level compute.

In sum, the observed gap to the dense baseline should not be read as a claim that dynamic models have intrinsically poorer ceilings; rather, it reflects the specific and constrained transfer-learning setting studied here. A more meaningful absolute comparison would (i) vary capacity $\gamma$ to trace out compute-accuracy curves; (ii) train a compute-matched dense control; (iii) include causal routing for DTF; and (iv) explore longer adaptation or from-scratch pre-training.

### 5.4 Limitations

We now summarise the main limitations of this work.

**Single capacity and seed.** All results were obtained at a single capacity $(\gamma=0.5)$ and a single random seed. We did not report confidence intervals or test statistical significance of small differences (e.g., DTF vs. MoD validation loss). This limits the strength of any claims about relative performance.

**One model scale and short adaptation.** We used only one model size (0.5B) and a limited fine-tuning budget. Conditional-compute benefits may scale with model size and training time [12, 6, 10], so the present results should be interpreted as a lower-bound demonstration.

**No causal router for DTF at inference.** Unlike MoD, we did not design or train a causal surrogate for DTF inference. As a result, we cannot evaluate strictly autoregressive decoding with DTF, and our multiple-choice evaluation may not fully reflect generation-time behaviour.

**Compute accounting.** We did not provide a FLOPs/tokens-per-second accounting or latency measurements for DTF vs. MoD vs. dense baselines. Although both DTF / MoD methods reduce theoretical per-step compute at $\gamma=0.5$, empirical latency depends on kernel efficiency and hardware characteristics. Future work should include measured throughput and end-to-end latency comparisons.

**Distance proxy and calibration.** We used MSE as a proxy for $D_{KL}$ under whitened-Gaussian assumptions; real hidden-state distributions will deviate from these assumptions. Although our results were stable, alternative divergences (e.g., Mahalanobis in whitened subspaces; layer-normalised cosine; or learned discriminators) might improve CE/CU calibration.

**Data drift and disclosure.** Because the Qwen2.5 pre-training data are not public, it is difficult to match the adaptation mixture to the pre-training distribution, increasing the risk of data drift. The near-random performance on some benchmarks for both DTF / MoD models suggests this may be material here; however, we cannot quantify it.

### 5.5 Implications and recommendations

Taken together, the results carry several implications for conditional computation research.

**Routing criterion matters, but effects are small at this scale.** Surprise-based DTF achieved slightly lower validation loss than MoD at matched capacity and training budget. This suggests the choice of routing criterion can matter, but the effect size was small under the present constraints, and our conclusions should be regarded as preliminary.

**Lightweight priors suffice.** Larger priors improved prior-prediction accuracy but did not improve validation loss. This is encouraging for efficient designs: a minimal PriorFFN appears adequate to produce a useful surprise signal.

**Parameter-efficient adaptation is workable but costs accuracy.** LoRA preserved the qualitative CE / CU dynamics but increased validation loss and degraded some benchmarks relative to full fine-tuning of dynamic components. Where parameter/memory budgets are tight, LoRA remains attractive; otherwise full fine-tuning is preferable.

**Absolute accuracy vs compute budget needs careful framing.** Conditional computation halves per-token processing at $\gamma=0.5$; unsurprisingly, both DTF/MOD models lost ground to the dense baseline on absolute accuracy. To make claims about "better compute-accuracy trade-offs" one must sweep $\gamma$ and compare to compute-matched dense controls.

### 5.6 Future work

We list the most impactful next steps, many of which address the limitations above.

**Capacity sweeps and compute-matched controls.** Evaluate $\gamma\in\{0.25,0.5,0.75,1.0\}$ and train a dense baseline with matched effective FLOPs (e.g., stochastic depth/depth-drop) to map compute-accuracy trade-offs fairly.

**Statistical robustness.** Run multiple seeds and report mean±sd for validation loss and benchmark accuracy; perform simple significance testing on DTF-MoD deltas.

**Causal routing for DTF.** Design and train a causal surrogate for DTF inference, analogous to MoD's auxiliary predictor [10], to enable strictly autoregressive decoding.

**Scale and training budget.** Test larger backbones and longer adaptation schedules; where possible, pre-train from scratch with conditional compute to assess scaling behaviour, as suggested by prior MoE/MOD work [12, 6, 10, 7].

**Divergence and calibration.** Replace or augment MSE with Mahalanobis (in whitened space), layer-normalised cosine, or learned discriminators to approximate $D_{KL}$ and recalibrate CE/CU thresholds; study whether this improves validation loss and downstream accuracy.

**Depth-wise capacity schedules.** Explore layer-specific capacities to align sparsity with representational abstraction (shallower layers more local, deeper layers more semantic), consistent with the predictive-coding view [16, 17, 14].

**Routing analysis and interpretability.** Instrument token-level routing by part-of-speech, rarity, or mutual information to study which tokens are gated by CE vs. CU and whether routing aligns with known linguistic/semantic cues.

**System-level metrics.** Report measured FLOPs/tok/sec, memory footprint, and end-to-end latency; implement fused kernels for Decision/Router (e.g., Triton) and measure gains under batching.

**Tasks and distributions.** Evaluate on long-context benchmarks, summarisation, retrieval-augmented QA, and code to probe settings where conditional compute should help; where possible, reduce data-drift by curating adaptation corpora closer to the pre-training distribution.

### 5.7 Concluding remarks

Within a constrained, transfer-learning setting at fixed capacity, DTF achieved slightly lower validation loss than MoD at approximately matched compute, with qualitatively consistent CE / CU dynamics that echo VPR. Absolute benchmark accuracy lagged the dense baseline, which we attribute to reduced per-token compute, short adaptation, possible data drift, and the small backbone scale. The present results therefore recommend caution: the choice of routing criterion appears to matter, but the practical benefits will likely depend on capacity, training budget, scale, and distributional alignment. Addressing these factors systematically, and providing robust statistical evidence and compute-matched baselines, constitutes the most direct path to establishing when, and by how much, surprise-based token routing can improve the compute-accuracy trade-off over heuristic importance scores in modern TF language models.

***

## 6. Conclusion

This chapter concludes the thesis by synthesising its primary findings and arguments. We first summarise the core contributions of the work, then provide direct and concise answers to the initial research questions based on the presented empirical evidence. Finally, we offer concluding remarks on the implications of this research and outline the most salient directions for future work.

### 6.1 Summary of Contributions

This thesis introduced and empirically evaluated the DTF, a conditional-computation architecture that translates event-driven, predictive routing from computational neuroscience into the token-wise domain of decoder-only TFs. The main contributions are as follows.

**Architectural design.** We proposed an alternating Decision Layer and Dynamic Layer. The Decision Layer computes a posterior state via a standard TF block and a lightweight prior state via a small PriorFFN; the Dynamic Layer implements a Predictive Router that converts the VPR criteria (expected/unexpected change) into a differentiable, capacity-constrained gating signal. This design retains a static compute graph with fixed capacity per layer.

**Principled gating mechanism.** We reformulated the VPR expected event (CE) and unexpected event (CU) criteria into a soft, differentiable, token-wise test based on distances between posterior, original, and prior states (approximating KL terms under a whitened-Gaussian assumption). This yields a context-dependent routing decision grounded in a model-comparison interpretation rather than an opaque importance score.

**Empirical comparison at fixed capacity.** At matched capacity $(\gamma=0.5)$, DTF achieved lower validation loss than a re-implemented MoD baseline [10], with similarly stable training dynamics. On a standard benchmark suite, DTF attained competitive accuracy and small improvements on multiple tasks relative to MoD.

**Ablation analyses.** Two ablations clarify design choices: (i) prior expressivity shows that larger PriorFFNs reduce the auxiliary prior loss but do not improve validation loss; a minimal PriorFFN is sufficient and slightly preferable, indicating that the predictive signal can be both lightweight and effective; (ii) parameter-efficient adaptation shows that LORA preserves the qualitative CE/CU dynamics but lags full fine-tuning in validation loss and on several benchmarks.

**Implementation and training protocol.** We provided a reproducible transfer-learning pipeline that adapts a pre-trained decoder-only model (Qwen2.5 0.5B [3]) by adding DTF/MoD components and optimising them with standard techniques (AdamW, mixed precision, activation checkpointing), enabling controlled comparisons at fixed capacity.

### 6.2 Answers to Research Questions

The empirical results from the 0.5B-parameter experiments provide the following answers to the research questions.

**1. Can a neuroscience-inspired, event-driven routing mechanism be integrated into a pre-trained TF to enable more efficient scaling while maintaining competitive performance?**

Partly. DTF integrates such a mechanism and yields a small, consistent validation-loss advantage over MoD under identical capacity and training settings, with routing dynamics $(CE/CU)$ that agree with predictive-coding expectations [16, 17, 14]. However, absolute benchmark accuracy lags the dense baseline at $\gamma=0.5$ in our short adaptation regime. We therefore characterise the result as an encouraging proof-of-concept for the routing criterion, rather than as a final statement on absolute performance.

**2. How does the empirical performance-compute trade-off of DTF compare to a standard MoD implementation?**

At matched capacity $(\gamma=0.5)$ DTF generalises marginally better, as evidenced by lower validation loss and small gains on several benchmarks. This suggests that a context-dependent, surprise-based criterion can provide a stronger inductive bias than a context-independent importance score. The effect size is modest in our setting and was not statistically tested.

**3. Is a parameter-efficient, transfer-learning approach a viable methodology for instantiating such a dynamic architecture from a pre-trained, dense TF?**

Yes. Adapting a pre-trained backbone by adding DTF components and fine-tuning only the dynamic parts is effective and tractable. Full fine-tuning of the dynamic components achieved the best validation loss and benchmark accuracy. LORA reduced trainable parameters and preserved the qualitative CE/CU dynamics but exhibited a consistent performance gap, indicating a clear efficiency-accuracy trade-off.

### 6.3 Concluding Remarks and Future Directions

Overall, the results demonstrate that surprise-based, hypothesis-testing style gating offers a principled and effective alternative to heuristic importance routing under fixed compute budgets. The mechanism trains stably, exhibits consistent transitions from novelty-driven to prediction-driven routing, and is robust to the capacity of its internal predictive model. The empirical dynamics are in qualitative agreement with VPR/DLH [14, 15]: CE becomes dominant as the prior improves, while CU is most useful early on.

At the same time, several limitations constrain the scope of our conclusions:
* We evaluated a single model scale (0.5B), a single capacity $(\gamma=0.5)$, and a single random seed; we did not average across seeds nor test statistical significance of small differences.
* We did not implement a causal surrogate router for DTF at inference (unlike MoD), which prevents strictly autoregressive evaluation for DTF.
* Both DTF/MoD models underperformed the dense baseline at $\gamma=0.5$, with some benchmark scores near random. This may reflect a combination of reduced per-token compute, short adaptation time, potential data drift between pre-training and fine-tuning distributions (Qwen2.5 pre-training data are not public), and the small backbone scale.
* We used mean-squared error as a proxy for KL divergence in the gating rule; more calibrated or learned distances may improve CE/CU calibration.
* We did not provide measured FLOPs/tokens/s or end-to-end latency; wall-clock gains depend on kernel efficiency and hardware.

These limitations point to concrete next steps. Future work should (i) sweep capacity $\gamma$ and include compute-matched dense controls to map compute-accuracy trade-offs fairly; (ii) run multiple seeds and report mean±sd with simple significance tests; (iii) design and train a causal DTF surrogate to enable strictly autoregressive decoding; (iv) test larger backbones and longer adaptation schedules, and-where possible-pre-train from scratch with conditional compute, following prior MoE/MOD scaling [12, 6, 7, 10]; (v) replace or augment MSE with Mahalanobis (in whitened space), layer-normalised cosine, or learned discriminators to approximate $D_{KL}$ and recalibrate CE/CU thresholds; (vi) explore depth-wise capacity schedules that align sparsity with representational abstraction; (vii) broaden evaluation to long-context, summarisation, retrieval-augmented QA and code; and (viii) report system-level metrics and implement fused kernels (e.g., Triton) for Decision/Router blocks.

In summary, this work establishes that a principled, neuroscience-inspired gating mechanism can be implemented efficiently in decoder-only TFs and can improve validation loss at fixed capacity relative to a leading engineering-driven baseline. The proposed architecture and analyses provide a clear basis for further technical progress on both the algorithmic and systems fronts. A more comprehensive study-covering capacity sweeps, statistical robustness, causal inference, scaling, calibrated distances, and system-level throughput-will be essential to establish if, and by how much, surprise-based token routing can improve the compute-accuracy trade-off in modern TF language models.

***

## References

[1] A. Vaswani, N. Shazeer, N. Parmar, J. Uszkoreit, L. Jones, A. N. Gomez, Ł. Kaiser, and I. Polosukhin, "Attention is all you need," in Advances in Neural Information Processing Systems, vol. 30, 2017.

[2] S. Minaee, T. Mikolov, N. Nikzad, M. Chenaghlu, R. Socher, X. Amatriain, and J. Gao, "Large Language Models: A Survey," arXiv preprint arXiv:2402.06196, 2025.

[3] Q. Team, A. Yang, B. Yang, B. Zhang, B. Hui, B. Zheng, B. Yu, C. Li, D. Liu, F. Huang, H. Wei, H. Lin, J. Yang, J. Tu, J. Zhang, J. Yang, J. Yang, J. Zhou, J. Lin, K. Dang, K. Lu, K. Bao, K. Yang, L. Yu, M. Li, M. Xue, P. Zhang, Q. Zhu, R. Men, R. Lin, T. Li, T. Tang, T. Xia, X. Ren, X. Ren, Y. Fan, Y. Su, Y. Zhang, Y. Wan, Y. Liu, Z. Cui, Z. Zhang, and Z. Qiu, "Qwen2.5 technical report," 2025.

[4] S. Tworkowski, K. Staniszewski, M. Pacek, Y. Wu, H. Michalewski, and P. Miłoś, "Focused transformer: Contrastive training for context scaling," Advances in Neural Information Processing Systems, vol. 36, 2023.

[5] N. F. Liu, K. Lin, J. Hewitt, A. Paranjape, M. Bevilacqua, F. Petroni, and P. Liang, "Lost in the middle: How language models use long contexts," Transactions of the Association for Computational Linguistics, vol. 12, pp. 157-173, 2024.

[6] W. Fedus, B. Zoph, and N. Shazeer, "Switch Transformers: Scaling to trillion parameter models with simple and efficient sparsity," Journal of Machine Learning Research, vol. 23, no. 120, pp. 1-39, 2022.

[7] A. Q. Jiang, A. Sablayrolles, A. Roux, A. Mensch, B. Savary, C. Bamford, D. S. Chaplot, D. d. l. Casas, E. B. Hanna, F. Bressand, et al., "Mixtral of experts," arXiv preprint arXiv:2401.04088, 2024.

[8] M. Dehghani, S. Gouws, O. Vinyals, J. Uszkoreit, and Ł. Kaiser, "Universal transformers," in International Conference on Learning Representations, 2018.

[9] J. Ainslie, T. Lei, M. de Jong, S. Ontañón, S. Brahma, Y. Zemlyanskiy, D. Uthus, M. Guo, J. Lee-Thorp, Y. Tay, et al., "Colt5: Faster long-range transformers with conditional computation," arXiv preprint arXiv:2303.09752, 2023.

[10] D. Raposo, S. Ritter, B. Richards, T. Lillicrap, P. C. Humphreys, and A. Santoro, "Mixture-of-Depths: Dynamically allocating compute in Transformer-based language models," arXiv preprint arXiv:2404.02258, 2024.

[11] T. Dao, D. Y. Fu, S. Ermon, A. Rudra, and C. Ré, "FlashAttention: Fast and memory-efficient exact attention with IO-awareness," in Advances in Neural Information Processing Systems, vol. 35, pp. 16344-16359, 2022.

[12] N. Shazeer, A. Mirhoseini, K. Maziarz, A. Davis, Q. Le, G. Hinton, and J. Dean, "Outrageously large neural networks: The sparsely-gated mixture-of-experts layer," in International Conference on Learning Representations, 2017.

[13] L. Itti and P. F. Baldi, "Bayesian surprise attracts human attention," in Advances in Neural Information Processing Systems, vol. 18, 2005.

[14] A. Zakharov, Q. Guo, and Z. Fountas, "Variational predictive routing with nested subjective timescales," in International Conference on Learning Representations, 2022.

[15] A. Zakharov, Q. Guo, and Z. Fountas, "Long-horizon video prediction using a dynamic latent hierarchy," arXiv preprint arXiv:2212.14376, 2023.

[16] R. P. N. Rao and D. H. Ballard, "Predictive coding in the visual cortex: a functional interpretation of some extra-classical receptive-field effects," Nature Neuroscience, vol. 2, no. 1, pp. 79-87, 1999.

[17] K. Friston, "The free-energy principle: a unified brain theory?," Nature Reviews Neuroscience, vol. 11, no. 2, pp. 127-138, 2010.

[18] Z. Fountas, M. A. Benfeghoul, A. Oomerjee, F. Christopoulou, G. Lampouras, H. Bou-Ammar, and J. Wang, "Human-like episodic memory for infinite context LLMs," arXiv preprint arXiv: 2407.09450, 2024.

[19] J. L. Elman, "Finding structure in time," Cognitive Science, vol. 14, no. 2, pp. 179-211, 1990.

[20] Y. Bengio, P. Simard, and P. Frasconi, "Learning long-term dependencies with gradient descent is difficult," IEEE Transactions on Neural Networks, vol. 5, no. 2, pp. 157-166, 1994.

[21] S. Hochreiter and J. Schmidhuber, "Long short-term memory," Neural Computation, vol. 9, no. 8, pp. 1735-1780, 1997.

[22] J. L. Ba, J. R. Kiros, and G. E. Hinton, "Layer normalization," arXiv preprint arXiv:1607.06450, 2016.

[23] B. Zhang and R. Sennrich, "Root mean square layer normalization," Advances in Neural Information Processing Systems, vol. 32, 2019.

[24] N. Shazeer, "Glu variants improve transformer," arXiv preprint arXiv:2002.05202, 2020.

[25] J. Su, M. Ahmed, Y. Lu, S. Pan, B. Wen, and Y. Liu, "RoFormer: Enhanced transformer with rotary position embedding," Neurocomputing, vol. 568, p. 127063, 2024.

[26] T. Dao, "FlashAttention-2: Faster attention with better parallelism and work partitioning," arXiv preprint arXiv:2307.08691, 2023.

[27] E. J. Hu, Y. Shen, P. Wallis, Z. Allen-Zhu, Y. Li, S. Wang, L. Wang, and W. Chen, "LoRA: Low-rank adaptation of large language models," arXiv preprint arXiv:2106.09685, 2021.

[28] J. Ainslie, J. Lee-Thorp, M. De Jong, Y. Zemlyanskiy, F. Lebrón, and S. Sanghai, "Gqa: Training generalized multi-query transformer models from multi-head checkpoints," arXiv preprint arXiv:2305.13245, 2023.

[29] D. Guo, D. Yang, H. Zhang, J. Song, R. Zhang, R. Xu, Q. Zhu, S. Ma, P. Wang, X. Bi, et al., "Deepseek-r1: Incentivizing reasoning capability in llms via reinforcement learning," arXiv preprint arXiv:2501.12948, 2025.

[30] M. Kumar, A. Goldstein, S. Michelmann, J. M. Zacks, U. Hasson, and K. A. Norman, "Bayesian surprise predicts human event segmentation in story listening," Cognitive Science, vol. 47, no. 10, p. e13343, 2023.

[31] D. J. C. MacKay, Information Theory, Inference and Learning Algorithms. Cambridge University Press, 2003.

[32] D. P. Kingma and J. Ba, "Adam: A method for stochastic optimization," arXiv preprint arXiv:1412.6980, 2014.

[33] I. Loshchilov and F. Hutter, "Decoupled weight decay regularization," arXiv preprint arXiv:1711.05101, 2017.

[34] N. Srivastava, G. Hinton, A. Krizhevsky, I. Sutsever, and R. Salakhutdinov, "Dropout: A simple way to prevent neural networks from overfitting," in Journal of Machine Learning Research, vol. 15, pp. 1929-1958, 2014.

[35] A. Paszke, S. Gross, F. Massa, A. Lerer, J. Bradbury, G. Chanan, T. Killeen, Z. Lin, N. Gimelshein, L. Antiga, et al., "Pytorch: An imperative style, high-performance deep learning library," Advances in neural information processing systems, vol. 32, 2019.

[36] D. Hendrycks, C. Burns, S. Basart, A. Zou, M. Mazeika, D. Song, and J. Steinhardt, "Measuring massive multitask language understanding," in Proceedings of the 2021 Conference on Empirical Methods in Natural Language Processing, pp. 10771-10793, 2021.

[37] P. Clark, I. Cowhey, O. Etzioni, T. Khot, A. Sabharwal, C. Schoenick, and O. Tafjord, "Think you have solved question answering? try ARC, the AI2 reasoning challenge," in Proceedings of the 2018 Conference on Empirical Methods in Natural Language Processing, pp. 157-166, 2018.

[38] R. Zellers, A. Holtzman, Y. Bisk, A. Farhadi, and Y. Choi, "HellaSwag: Can a machine really finish your sentence?," in Proceedings of the 57th Annual Meeting of the Association for Computational Linguistics, pp. 4791-4800, 2019.

[39] K. Sakaguchi, R. Le Bras, C. Bhagavatula, and Y. Choi, "WinoGrande: An adversarial winograd schema challenge at scale," Communications of the ACM, vol. 64, no. 9, pp. 99-106, 2021.

[40] S. Lin, J. Hilton, and O. Evans, "TruthfulQA: Measuring how models mimic human falsehoods," arXiv preprint arXiv:2109.07958, 2021.