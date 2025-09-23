# Technical Specification: Surprise-based Transformer with Transition network - the Subjective Timescale Transformer (STT)

## 1. Overview

The **Subjective Timescale Transformer (STT)** is a conditional-compute architecture for decoder-only Transformers (like Qwen2.5). Its goal is to dynamically allocate computational resources (i.e., executing a full Transformer block) only to tokens that are "surprising" or information-rich, while allowing predictable tokens to bypass the block via a residual connection. This saves computation during inference.

The architecture is built on three core principles:
1.  **Surprise-based Gating:** Inspired by Variational Predictive Routing (VPR), the decision to process a token is based on a model-comparison signal. A token is processed if the change in its representation is significant (unexpected) or better explained by a learned dynamics model (expected) than by a static, no-change assumption.
2.  **Teacher-Student Paradigm:** To make this process efficient and causal, STT uses a teacher-student approach:
    * **Teacher (Training):** A computationally intensive but accurate **Predictive Router** operates on the full output of a Transformer block to generate high-quality, non-causal routing decisions.
    * **Student (Inference):** A lightweight **Causal Router** is trained to mimic the Teacher's decisions using only causally available inputs. At inference time, only this efficient Student router is used.
3.  **Configurable Routing:** The system supports two main routing strategies, controlled by flags:
    * **Fixed-Capacity (Top-K):** Processes a fixed percentage of tokens per layer, ensuring a static compute graph for hardware efficiency.
    * **Variable-Capacity (Thresholding):** Processes tokens whose surprise score exceeds a threshold, allowing each layer to dynamically adjust its compute budget per sequence.

The implementation alternates standard `Qwen2DecoderLayer`s with custom `STTLayer`s, which contain the dynamic routing logic.


## 2. Core Components of an `STTLayer`

Each `STTLayer` is composed of the following modules, as implemented in `src/models/stt/model.py`:

* **TF-Block (`self.block`):** A standard `Qwen2DecoderLayer` wrapped in a `DynamicBlock` class. The `DynamicBlock` provides a `process_selected` method for applying the layer's logic to a subset of tokens.
* **Transition Network (TPN) (`self.transition_network`):** A small, pre-normalized MLP that predicts the change (residual) for the current token based on the final hidden state of the *previous* token.
* **Predictive Router (Teacher) (`self.predictive_router`):** A non-parametric module that calculates surprise scores and generates the final gating signal, `g_cont`. It is only used to generate targets during training.
* **Causal Router (Student) (`self.causal_router`):** A lightweight, learnable MLP that predicts routing decisions causally. It takes the current and previous token's *input* states as features. This is the only router used during efficient inference.

## 3. Mathematical Formulation & Logic Flow

This section details the step-by-step logic for a single `STTLayer` at depth $l$.

### 3.1 Training Pass (Teacher Mode)

During training, the primary goal is to perform a **single dense forward pass** to generate high-quality supervisory signals for the TPN and the Causal Router.

**Step 1: Dense Block Computation & Residual Calculation**
Compute the full block output $x_t^{(l)}$ and the "ground truth" residual $\Delta x_t^{(l)}$ for all tokens.
$$x_t^{(l)} = \text{TF-Block}^{(l)}(x_t^{(l-1)})$$
$$\Delta x_t^{(l)} = x_t^{(l)} - x_t^{(l-1)}$$

**Step 2: Transition Network (TPN) Prediction**
The TPN predicts the residual for token $t$ using the final state of the previous token, $x_{t-1}^{(l)}$.
$$\hat{\Delta x}_t^{(l)} = \text{TPN}^{(l)}(x_{t-1}^{(l)}) \quad (\text{where } x_0^{(l)} = \mathbf{0})$$

**Step 3: Surprise Metric Calculation**
The Predictive Router calculates the static surprise ($D_{st}$) and change surprise ($D_{ch}$).
$$D_{st,t}^{(l)} = \frac{1}{d} \left\lVert \Delta x_t^{(l)} \right\rVert_2^2$$
$$D_{ch,t}^{(l)} = \frac{1}{d} \left\lVert \Delta x_t^{(l)} - \hat{\Delta x}_t^{(l)} \right\rVert_2^2$$

**Step 4: Gating Criteria & Probabilistic Gate**
The surprise metrics are combined using learnable bias parameters ($o_{ce}^{(l)}, m_{cu}^{(l)}$) and scheduled inverse temperatures ($\beta_{ce}, \beta_{cu}$) to produce a continuous gating score $g_{t}^{(l)} \in [0, 1]$.
$$CE_t^{(l)} = D_{st,t}^{(l)} - \left(D_{ch,t}^{(l)} - \log(\text{softplus}(o_{ce}^{(l)}))\right)$$$$CU_t^{(l)} = D_{st,t}^{(l)} - \left(\text{softplus}(m_{cu}^{(l)}) \cdot \text{MA}(D_{st,t}^{(l)}) \right)$$$$g_{t}^{(l)} = \sigma(\beta_{ce} \cdot CE_t^{(l)}) + \sigma(\beta_{cu} \cdot CU_t^{(l)}) - \sigma(\beta_{ce} \cdot CE_t^{(l)}) \sigma(\beta_{cu} \cdot CU_t^{(l)})$$

**Step 5: Routing Target Generation**
A binary target mask $m_t^{(l)}$ is created based on the configured routing mode.
* **Fixed-Capacity:** $m_t^{(l)} = \text{TopK}(g_t^{(l)}, k = \lfloor \gamma T \rfloor)$
* **Variable-Capacity:** $m_t^{(l)} = \mathbb{I}(g_t^{(l)} \ge g_{th})$

**Step 6: Causal Router Training & Final Output**
The Causal Router is trained to predict the binary targets $m_t^{(l)}$ using only the layer's *input* states.
$$\tilde{r}_t^{(l)} = \tilde{R}^{(l)}([x_t^{(l-1)} \, \Vert \, x_{t-1}^{(l-1)}])$$
The final output of the `STTLayer` in training mode is the **dense block output**, $x_t^{(l)}$. The logic described in the code that performs a *second*, partial block pass (`self.block.process_selected`) should be considered an implementation detail for exploring soft gating during training, but the primary architectural specification relies on a single pass to generate signals and a single dense output to pass to the next layer.

### 3.2 Inference Pass (Student Mode)

During inference, the Teacher path is disabled. Only the efficient Causal Router is used.

**Step 1: Causal Logit Calculation**
Compute the routing logit for each token.
$$\tilde{r}_t^{(l)} = \tilde{R}^{(l)}([x_t^{(l-1)} \, \Vert \, x_{t-1}^{(l-1)}])$$

**Step 2: Routing Decision**
Make a binary routing decision based on the configured mode.
* **Fixed-Capacity:** Select the top-$k$ tokens based on their logits $\tilde{r}_t^{(l)}$.
* **Variable-Capacity:** Select tokens where the predicted probability exceeds the threshold: $\sigma(\tilde{r}_t^{(l)}) \ge g_{th}$.

**Step 3: Conditional Block Execution**
The TF-Block is only executed for the selected tokens.
$$
x_t^{(l)} = \begin{cases}
    \text{TF-Block}^{(l)}(x_t^{(l-1)}) & \text{if token } t \text{ is selected} \\
    x_t^{(l-1)} & \text{otherwise}
\end{cases}
$$
**Crucially, only selected tokens contribute their keys and values to the KV cache for this layer.**

## 4. Configuration Flags & Ablations

The behavior of STT is controlled by several parameters in `config/default.yaml`.

### 4.1 Routing Mode

* `stt.use_g_threshold_selection` (bool):
    * If `true`, the model uses **Variable-Capacity (Thresholding)** routing. The number of processed tokens per layer is dynamic.
    * If `false`, the model uses **Fixed-Capacity (Top-K)** routing. The number of processed tokens is determined by `model.capacity`.
* `stt.g_threshold` (float): The threshold value used when `use_g_threshold_selection` is `true`.
* `model.capacity` (float): The fraction ($\gamma$) of tokens to process in Top-K mode.

### 4.2 Causal Router Control

* `train_causal_router` (bool): If `true`, the Causal Router (student) is instantiated and trained. If `false`, it is not created.
* `use_causal_router_in_validation` (bool):
    * If `true` (and `train_causal_router` is true), validation and inference use the efficient **Student path**.
    * If `false`, validation and inference use the **Teacher path**. This involves a full dense block pass to compute surprise signals and is useful for evaluating the "upper bound" performance of the routing criteria, but it is more computationally expensive.

### 4.3 Router Parameter Control

* `learn_o_ce` / `learn_m_cu` (bool): If `true`, the bias parameters $o_{ce}$ and $m_{cu}$ in the gating criteria are learnable `nn.Parameter`s. If `false`, they are fixed buffers.
* `beta_schedule`: This section defines a schedule (e.g., `cosine`) to anneal the inverse temperatures $\beta_{ce}$ and $\beta_{cu}$ from a `_start` to an `_end` value over the course of training. This allows the gating decision to transition from "soft" (low beta, more exploration) to "hard" (high beta, more decisive).

## 5. Loss Functions

The total loss is a weighted sum of four components:

1.  **Language Modeling Loss ($\mathcal{L}_{\text{LM}}$):** The standard cross-entropy loss for next-token prediction.
2.  **TPN Loss ($\mathcal{L}_{\text{TPN}}$):** An MSE loss that trains the Transition Network to predict the true residual.
    $$\mathcal{L}_{\text{TPN}}^{(l)} = \text{MSE}(\hat{\Delta x}_t^{(l)}, \text{stop\_grad}(\Delta x_t^{(l)}))$$
3.  **Causal Router Loss ($\mathcal{L}_{\text{causal}}$):** A Binary Cross-Entropy loss that trains the Causal Router to mimic the Teacher's decisions.
    $$\mathcal{L}_{\text{causal}}^{(l)} = \text{BCE}(\sigma(\tilde{r}_t^{(l)}), \text{stop\_grad}(m_t^{(l)}))$$
4.  **Gating Regularization Loss ($\mathcal{L}_{g\_\text{reg}}$):** The mean of the continuous gating scores across all STT layers. This encourages a target level of sparsity and is particularly important in thresholding mode.
    $$\mathcal{L}_{g\_\text{reg}} = \text{mean}(g^{(l)} \text{ for all STT layers } l)$$

The final objective function is:
$$\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{LM}} + \lambda_{\text{TPN}} \cdot \mathcal{L}_{\text{TPN}} + \lambda_{\text{causal}} \cdot \mathcal{L}_{\text{causal}} + \lambda_{g\_\text{reg}} \cdot \mathcal{L}_{g\_\text{reg}}$$
The weights ($\lambda$) are defined in `config/default.yaml` under `stt.tpn_loss_weight`, `stt.causal_loss_weight`, and `stt.g_reg_loss_weight`.

## 6. Parameter Groups & Optimization

To facilitate stable training, the model's parameters are split into distinct groups with different learning rates, as defined in `training.optimizer.lrs`.
* `base_model`: The parameters of the underlying Qwen2.5 model. Typically uses the lowest learning rate.
* `stt_transition_network`: The weights of the TPN.
* `stt_predictive_router`: The learnable bias parameters ($o_{ce}, m_{cu}$) of the Teacher router. Typically uses the highest learning rate to adapt the gating policy quickly.
* `stt_causal_router`: The weights of the Student router.

## 7. Pseudocode for an `STTLayer` Forward Pass

```python
function forward_stt_layer(
    original_hidden, 
    is_training,
    use_causal_router,
    use_threshold_gating,
    # ... other args like betas, attention_mask, etc.
):
    # --- Part 1: Signal Generation (Teacher Path) ---
    # This part is always run if is_training=True or if the non-causal path is used for validation.
    
    if is_training or not use_causal_router:
        # Step 1: Dense pass to get ground truth
        processed_hidden = self.block(original_hidden, ...)
        actual_residual = processed_hidden - original_hidden

        # Step 2: TPN prediction
        prev_final_states = get_previous_token_states(processed_hidden)
        predicted_residual = self.transition_network(prev_final_states)
        tpn_loss = mse_loss(predicted_residual, actual_residual.detach())

        # Step 3 & 4: Calculate surprise and gating score
        D_st = calculate_static_surprise(actual_residual)
        D_ch = calculate_change_surprise(actual_residual, predicted_residual)
        g_cont = calculate_gating_score(D_st, D_ch, ...) # Uses betas, o_ce, m_cu
    
    # --- Part 2: Routing and Conditional Execution ---

    if is_training:
        # Step 5: Generate targets for the student
        if use_threshold_gating:
            binary_targets = g_cont >= g_threshold
        else: # Top-K
            binary_targets = top_k(g_cont, capacity)

        # Step 6: Train student
        causal_logits = self.causal_router(original_hidden, ...)
        causal_loss = bce_loss(causal_logits, binary_targets.detach())
        
        # NOTE: The implementation has a second pass for soft-gating experiments.
        # The primary spec returns the dense `processed_hidden`.
        # For experimental flexibility, a conditional second pass can be added here.
        # e.g., final_hidden_states = self.block.process_selected(...)
        
        final_hidden_states = processed_hidden # According to the single-pass spec
        return final_hidden_states, {tpn_loss, causal_loss}, g_cont

    else: # Inference
        if use_causal_router:
            # Efficient Student Path
            causal_logits = self.causal_router(original_hidden, ...)
            
            if use_threshold_gating:
                selected_mask = sigmoid(causal_logits) >= g_threshold
            else: # Top-K
                selected_mask = top_k_from_logits(causal_logits, capacity)
        
        else: # Fallback to Teacher Path for validation
            if use_threshold_gating:
                selected_mask = g_cont >= g_threshold
            else: # Top-K
                selected_mask = top_k(g_cont, capacity)
        
        batch_indices, token_indices = get_indices_from_mask(selected_mask)

        # Conditional execution
        final_hidden_states = self.block.process_selected(
            original_hidden, 
            batch_indices, 
            token_indices,
            use_soft_gating=False # Hard gating at inference
        )
        return final_hidden_states, {}, None
```