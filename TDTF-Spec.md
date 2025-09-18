# Technical Specification: Temporal Dynamic Transformer (TDT)

## 1. Overview

The Temporal Dynamic Transformer (TDT) is a conditional computation architecture that gates the execution of entire Transformer (TF) blocks. It operates under a student-teacher framework with two distinct modes: a training-time "teacher" mode and an inference-time "student" mode.

- During training, a powerful, non-causal Predictive Router observes the output of a TF block to make an optimal routing decision. This decision is used as a supervisory signal.
- During inference, a lightweight, Causal Router predicts this decision using only causally available information, making its choice *before* the TF block is executed.

This document specifies the complete end-to-end logic for implementing the TDT architecture.

## 2. Notation

- $l$: The layer index of the Transformer, from $1$ to $L$.
- $t$: The token index in the sequence, from $1$ to $T$.
- $d$: The hidden dimension of the model.
- $x_t^{(l-1)} \in R^d$: The hidden state for token $t$ entering layer $l$.
- $x_t^{(l)} \in R^d$: The hidden state for token $t$ exiting layer $l$.
- $\Delta x_t^{(l)} \in R^d$: The residual update computed by the TF block at layer $l$ for token $t$.
- $\text{TF-Block}(\cdot)$: The standard Transformer block function (Attention + FFN).
- $\text{TPN}(\cdot)$: The Transition Network.
- $\tilde{R}^{(l)}(\cdot)$: The Causal Router for layer $l$.

## 3. Training Phase Architecture (The "Teacher" Model)

During training, for each TDT layer $l$, all computations are performed to generate the necessary supervisory signals. The logic flow for a single token $t$ is as follows.

### 3.1. Standard TF Block Computation

First, compute the full TF block output for the current token $t$. This provides the ground truth for the surprise calculation.

The residual update is:
$$
\Delta x_t^{(l)} = \text{TF-Block}(x_t^{(l-1)}) - x_t^{(l-1)}
$$

The final output state for the layer is:
$$
x_t^{(l)} = x_t^{(l-1)} + \Delta x_t^{(l)}
$$

### 3.2. Transition Network (TPN)

The TPN is a lightweight, bottlenecked Feed Forward Neural Net (Very similar to the Prior FFN of the DTF model). It predicts the residual update of the current token, $\Delta x_t^{(l)}$, using the *final output state* of the previous token, $x_{t-1}^{(l)}$.

$$
\Delta\hat{x}_{t}^{(l)} = \text{TPN}(x_{t-1}^{(l)})
$$

Note: For the first token of a sequence ($t=1$), the TPN input $x_{0}^{(l)}$ should be a zero vector of dimension $d$. The TPN's prediction $\Delta\hat{x}_{1}^{(l)}$ will also be computed from this zero vector.

### 3.3. Predictive Router (Non-Causal)

The Predictive Router uses the actual residual $\Delta x_{t}^{(l)}$ and the predicted residual $\Delta\hat{x}_{t}^{(l)}$ to calculate a continuous gate value $g_t^{(l)}$.

#### 3.3.1. Surprise Calculation

Calculate the static surprise ($D_{st}$) and change surprise ($D_{ch}$). $D_{st}$ measures the magnitude of the actual update. $D_{ch}$ measures the TPN's prediction error.

$$
D_{st,t}^{(l)} = \frac{1}{d} || \Delta x_{t}^{(l)} ||_2^2
$$

$$
D_{ch,t}^{(l)} = \frac{1}{d} || \Delta x_{t}^{(l)} - \Delta\hat{x}_{t}^{(l)} ||_2^2
$$

#### 3.3.2. VPR Event Criteria

Transform the surprise metrics into event criteria using four learnable scalar parameters: $o_{ce}, m_{cu}, \beta_{ce}, \beta_{cu}$.

First, compute the raw criteria values for an expected event (CE) and an unexpected event (CU).

$$
CE_{t}^{(l)} = D_{st,t}^{(l)} - (D_{ch,t}^{(l)} - \log(o_{ce} + 10^{-10}))
$$

$$
CU_{t}^{(l)} = D_{st,t}^{(l)} - (m_{cu} \cdot \text{MA}(D_{st,t}^{(l)}))
$$

The Moving Average (MA) should be calculated over a fixed window (e.g., the last 100 tokens) of the static surprise values within the current sequence and layer. For tokens early in the sequence, the MA can be computed over the available history.

#### 3.3.3. Continuous Gate Computation

Convert the raw criteria into probabilities using the learnable inverse temperatures $\beta_{ce}$ and $\beta_{cu}$ and the sigmoid function. A `Softplus` function ensures the temperatures are positive.

$$
S_{CE,t}^{(l)} = \sigma(\text{Softplus}(\beta_{ce}) \cdot CE_{t}^{(l)})
$$

$$
S_{CU,t}^{(l)} = \sigma(\text{Softplus}(\beta_{cu}) \cdot CU_{t}^{(l)})
$$

Combine these probabilities using a probabilistic OR to get the final continuous gate value $g_t^{(l)}$.

$$
g_t^{(l)} = S_{CE,t}^{(l)} + S_{CU,t}^{(l)} - (S_{CE,t}^{(l)} \cdot S_{CU,t}^{(l)})
$$

This value, $g_t^{(l)} \in [0, 1]$, is the teacher's final belief that token $t$ was a temporal event at layer $l$.

### 3.4. Causal Router Training

The Causal Router is trained in parallel.

#### 3.4.1. Target Generation

Create a binary target mask, $m^{(l)} \in \{0, 1\}^T$, by selecting the tokens with the TopK highest scores from the continuous gate vector $g^{(l)}$. The number of tokens to select, $k$, is determined by a fixed capacity hyperparameter $\gamma$.

$$
k = \lfloor\gamma \cdot T\rfloor
$$

$$
m_t^{(l)} = 1 \text{ if } g_t^{(l)} \text{ is in the TopK of } g^{(l)}, \text{ else } 0
$$

#### 3.4.2. Causal Router Prediction

The Causal Router, $\tilde{R}^{(l)}$, is a simple linear layer. It takes as input the concatenation of the *pre-computation* states $x_t^{(l-1)}$ and $x_{t-1}^{(l-1)}$.

$$
\vz_t^{(l-1)} = [x_t^{(l-1)} \mathbin\Vert x_{t-1}^{(l-1)}]
$$

For $t=1$, the input $x_0^{(l-1)}$ should be a zero vector. The router produces a logit, which is passed through a sigmoid to get a predicted probability $\hat{g}_t^{(l)}$.

$$
\hat{g}_t^{(l)} = \sigma(\tilde{R}^{(l)}(\vz_t^{(l-1)}))
$$

## 4. Inference Phase Architecture (The "Student" Model)

During inference, the TPN and Predictive Router are discarded. Only the pre-trained Causal Router $\tilde{R}^{(l)}$ is used.

### 4.1. Causal Gating Decision

For each token $t$ at layer $l$:
1. Construct the causal input: $\vz_t^{(l-1)} = [x_t^{(l-1)} \mathbin\Vert x_{t-1}^{(l-1)}]$.
2. Get the predicted probability from the Causal Router: $\hat{g}_t^{(l)} = \sigma(\tilde{R}^{(l)}(\vz_t^{(l-1)}))$.
3. Binarize this probability to get a hard decision, $m_t^{(l)} \in \{0, 1\}$. This can be done by simple thresholding (e.g., `if` $\hat{g}_t^{(l)} > 0.5$) or by maintaining a running TopK of probabilities for the generated prefix.

### 4.2. Conditional TF Block Execution

Execute the TF block based on the causal decision.

- If $m_t^{(l)} = 1$: The token is processed normally.
$$
x_t^{(l)} = \text{TF-Block}(x_t^{(l-1)})
$$

- If $m_t^{(l)} = 0$: The entire TF block is skipped. The token's state passes through the residual connection unchanged.
$$
x_t^{(l)} = x_t^{(l-1)}
$$

The token's Key and Value vectors are only added to the KV cache if $m_t^{(l)} = 1$.

## 5. Training Objectives and Losses

The model is trained by minimizing a total loss function composed of three parts:

1. The main Language Modeling loss ($\mathcal{L}_{LM}$).
2. The TPN auxiliary loss ($\mathcal{L}_{TPN}$).
3. The Causal Router auxiliary loss ($\mathcal{L}_{causal}$).

### 5.1. TPN Loss

This is a Mean Squared Error loss between the TPN's prediction and the actual residual update. A `stop_gradient` is applied to the target to prevent this loss from affecting the main TF block's weights.

$$
\mathcal{L}_{\text{TPN}} = \text{MSE}(\Delta\hat{x}_{t}^{(l)}, \text{stop\_gradient}(\Delta x_t^{(l)}))
$$

### 5.2. Causal Router Loss

This is a Binary Cross-Entropy loss between the Causal Router's prediction and the binary target mask generated by the teacher model.

$$
\mathcal{L}_{\text{causal}} = \text{BCE}(\hat{g}_{t}^{(l)}, m_{t}^{(l)})
$$

## 6. Parameter Initialization

The four learnable parameters of the Predictive Router should be initialized to reasonable values to ensure stable training startup. Suggested initial values based on the VPR paper are:
- $o_{ce} = 1.025$
- $m_{cu} = 1.1$
- $\beta_{ce} = -0.3$
- $\beta_{cu} = -0.6$

All other new modules (TPN, Causal Router) should use standard initialization procedures (e.g., Kaiming or Xavier).