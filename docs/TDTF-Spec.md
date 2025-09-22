# Temporal Dynamic Transformer (TDTF) — Technical Specification

1. Overview
TDTF is a conditional-compute Transformer that gates entire Transformer blocks at token granularity using a teacher–student paradigm:

- Teacher (training-time): a non-causal Predictive Router computes surprise-based routing scores using the full TF-block residual and a Transition Network (TPN) prediction. These scores define TopK binary targets per sequence and layer.
- Student (inference-time): a lightweight Causal Router predicts those routing decisions causally, before a block executes. Selected tokens execute the TF block; others bypass it via the residual path, and do not contribute keys/values (K/V) to the layer’s cache.

Key properties
- Surprise-based gating (CE/CU) inspired by VPR: events trigger extra compute.
- Fixed-capacity TopK per layer/sequence keeps compute predictable and static-graph friendly.
- Inverse temperatures for CE and CU are scheduled across training (not learned), starting soft and ending hard.
- No default hyperparameters in code: all required parameters must be provided via configuration.

2. Notation
- Sequence length: \( T \in \mathbb{N} \)
- Batch size: \( B \in \mathbb{N} \)
- Hidden size: \( d \in \mathbb{N} \)
- Layer index: \( l \in \{1,\dots,L\} \)
- Token index: \( t \in \{1,\dots,T\} \)
- Input to layer \( l \): \( x_t^{(l-1)} \in \mathbb{R}^d \)
- Output from layer \( l \): \( x_t^{(l)} \in \mathbb{R}^d \)
- Transformer block function: TF-Block(\(\cdot\))
- Residual update (true): \( \Delta x_t^{(l)} = x_t^{(l)} - x_t^{(l-1)} \)
- Transition Network: TPN(\(\cdot\))
- Causal Router at layer \( l \): \( \tilde{R}^{(l)}(\cdot) \)
- Capacity per layer: \( \gamma \in (0,1] \) with \( k = \lfloor \gamma T \rfloor \)
- Moving-average window (per layer): \( \tau_w \in \mathbb{N} \)
- Inverse temperatures (scheduled scalars): \( \beta_{ce} > 0, \beta_{cu} > 0 \)

3. Architecture
3.1 Layering
- Stack alternates standard TF decoder layers and TDTF layers, e.g., even \( l \) are standard, odd \( l \) are TDTF.
- Pre-norm assumption (RMSNorm/LayerNorm before sublayers) is recommended.
- Each TDTF layer comprises:
  - Standard TF block (attention + MLP).
  - Transition Network (TPN): a small MLP operating on previous token’s final state.
  - Predictive Router (teacher): non-causal gating based on surprise.
  - Causal Router (student): causal gating based on pre-block inputs.

3.2 Execution order in a TDTF layer
- Teacher (training): compute the dense TF block output; derive surprise and routing signals; train student router with TopK targets. Block skipping is not used for the teacher path.
- Student (inference): compute causal scores first; decide skip/execute; run TF block only for selected tokens; unselected tokens bypass the block.

4. Teacher Mode (Training-Time)
4.1 TF block and residuals
Compute the full block and residual for each token:
- TF block:
  $$ x_t^{(l)} = \text{TF-Block}(x_t^{(l-1)}) $$
- Residual update:
  $$ \Delta x_t^{(l)} = x_t^{(l)} - x_t^{(l-1)} $$

4.2 Transition Network (TPN)
Predict the residual using the previous token’s final state at the same layer:
- Input convention:
  - For \( t=1 \): \( x_{0}^{(l)} = \mathbf{0} \in \mathbb{R}^d \)
  - For \( t>1 \): \( x_{t-1}^{(l)} \) is the previous token’s final state
- Prediction:
  $$ \hat{\Delta x}_t^{(l)} = \text{TPN}(x_{t-1}^{(l)}) $$
- TPN loss (per token, per layer):
  $$ \mathcal{L}_{TPN}^{(l)} = \frac{1}{d}\left\lVert \hat{\Delta x}_t^{(l)} - \text{stop\_grad}(\Delta x_t^{(l)}) \right\rVert_2^2 $$
Scale with \( 1/d \) implicitly by using vector MSE; average across \( t \) and across TDTF layers when combining.

4.3 Surprise metrics
For each token:
- Static surprise:
  $$ D_{st,t}^{(l)} = \frac{1}{d}\left\lVert \Delta x_t^{(l)} \right\rVert_2^2 $$
- Change surprise:
  $$ D_{ch,t}^{(l)} = \frac{1}{d}\left\lVert \Delta x_t^{(l)} - \hat{\Delta x}_t^{(l)} \right\rVert_2^2 $$

4.4 Event criteria (CE/CU)
Let \( \text{MA}(D_{st}) \) be a causal, per-sequence moving average with window \( \tau_w \), computed per layer. For early timesteps \( t < \tau_w \), compute over available history.

- Learnable bias terms (per layer): \( o_{ce} > 0 \) and \( m_{cu} > 0 \) (implemented as Softplus of raw parameters).
- Expected-event criterion:
  $$ CE_t^{(l)} = D_{st,t}^{(l)} - \left(D_{ch,t}^{(l)} - \log(o_{ce})\right) $$
- Unexpected-event criterion:
  $$ CU_t^{(l)} = D_{st,t}^{(l)} - \left(m_{cu} \cdot \text{MA}(D_{st}^{(l)})_t \right) $$

4.5 Scheduled inverse temperatures and soft gating
Inverse temperatures \( \beta_{ce}, \beta_{cu} \) are scheduled over training steps (not learned) and provided by the trainer at each forward. Apply them to produce probabilities:

- Soft probabilities:
  $$ S_{CE,t}^{(l)} = \sigma\!\left( \beta_{ce} \cdot CE_t^{(l)} \right), \quad S_{CU,t}^{(l)} = \sigma\!\left( \beta_{cu} \cdot CU_t^{(l)} \right) $$

- Continuous gate via probabilistic OR:
  $$ g_{t}^{(l)} = S_{CE,t}^{(l)} + S_{CU,t}^{(l)} - S_{CE,t}^{(l)} S_{CU,t}^{(l)} $$

4.6 TopK routing targets (teacher -> student)
Per sequence (per layer), choose the \( k = \lfloor \gamma T \rfloor \) tokens with highest \( g_t^{(l)} \). Produce the binary mask \( m_t^{(l)} \in \{0,1\} \) with exactly \( k \) ones.

4.7 Causal Router training (student)
The student router predicts \( m_t^{(l)} \) causally using only pre-block information:

- Input features (recommendation):
  $$ z_t^{(l-1)} = \left[ \text{Norm}(x_t^{(l-1)}) \,\Vert\, \text{Norm}(x_{t-1}^{(l-1)}) \right] $$
  Use RMSNorm/LayerNorm to improve scale invariance. For \( t=1 \), use \( x_0^{(l-1)} = \mathbf{0} \).

- Router logit:
  $$ \tilde{r}_t^{(l)} = \tilde{R}^{(l)}(z_t^{(l-1)}) $$
- Student loss (BCE with teacher targets):
  $$ \mathcal{L}_{causal}^{(l)} = \text{BCE}\!\left( \sigma(\tilde{r}_t^{(l)}),\; m_t^{(l)} \right) $$

4.8 Total loss (layer-wise and global)
Let \( \mathcal{D} \) be the set of TDTF layers. Combine losses as:
- Average auxiliary losses across TDTF layers to stabilize scale.
- Per-layer losses are averaged across tokens in the batch/sequence.

Global objective:
- \( \mathcal{L}_{total} = \mathcal{L}_{LM} + \lambda_{TPN} \cdot \frac{1}{|\mathcal{D}|} \sum_{l\in\mathcal{D}} \mathcal{L}_{TPN}^{(l)} + \lambda_{causal} \cdot \frac{1}{|\mathcal{D}|} \sum_{l\in\mathcal{D}} \mathcal{L}_{causal}^{(l)} \)

All weights \( \lambda_{TPN}, \lambda_{causal} \) must be provided via configuration. No defaults in code.

5. Student Mode (Inference-Time)
5.1 Autoregressive generation (T = 1 per sequence per step)
At generation step \( s \):
- Compute student router logits for each batch element’s current token.
- Routing decision before block execution:
  - Capacity-enforced TopK mode (recommended): select the top \( \lfloor \gamma B \rfloor \) examples in the batch to process at this step (hard budget per step).
  - Threshold mode: process tokens with \( \sigma(\tilde{r}) > \tau \) (non-deterministic budget).
- Execute the TF block only for selected tokens; others are bypassed:
  $$ x_t^{(l)} = \begin{cases}
      \text{TF-Block}(x_t^{(l-1)}) & \text{if selected}\\
      x_t^{(l-1)} & \text{otherwise}
  \end{cases} $$
- KV cache semantics: only selected tokens write K/V for layer \( l \). Unselected tokens are not visible as keys/values to other tokens at this layer.

5.2 Batched evaluation (T > 1)
For simplicity and correctness in evaluation scenarios with full sequences, run dense (no skipping), unless you implement a correct, per-layer per-sequence capacity mechanism with causal constraints.

6. Scheduling of inverse temperatures
Inverse temperatures \( \beta_{ce} \) and \( \beta_{cu} \) are scheduled by the training loop via a configurable schedule. They are not learnable model parameters.

6.1 Schedule definition
Define a schedule function \( f: \{0,\dots,S\} \rightarrow \mathbb{R}^+ \) for \( \beta \) with warm-up:

- Given total training steps \( S \), warm-up steps \( S_w \), and endpoints \( \beta_{\text{start}} \), \( \beta_{\text{end}} \), define progression \( r \in [0,1] \):
  $$ r = \begin{cases}
      0, & s \le S_w \\
      \frac{s - S_w}{S - S_w}, & s > S_w
  \end{cases} $$
- Linear:
  $$ \beta(s) = \beta_{\text{start}} + r\left(\beta_{\text{end}} - \beta_{\text{start}}\right) $$
- Cosine:
  $$ \beta(s) = \beta_{\text{start}} + \frac{1 - \cos(\pi r)}{2}\left(\beta_{\text{end}} - \beta_{\text{start}}\right) $$

The trainer must compute \( \beta_{ce}(s), \beta_{cu}(s) \) each step and pass them to the model forward.

6.2 Practical guidance
- Start with relatively low \( \beta \) values (soft gating); ramp up to higher values (hard gating).
- Use the same schedule family (linear/cosine) for both CE and CU or configure separately.

7. Required configuration parameters
All of the following must be provided (no defaults in code):

Model-architecture and routing
- tdtf_capacity: \( \gamma \in (0,1] \)
- ma_window: \( \tau_w \in \mathbb{N} \) (per-layer, per-sequence moving average length)
- prior_ffn_intermediate_size_factor: \( f \in (0,1] \) (e.g., 0.0625) to compute TPN width \( d_i = \max(2, \lfloor f \cdot d \rfloor) \)
- student_routing_mode: “topk” or “threshold”
- o_ce_init > 0, m_cu_init > 0 (learnable bias initializations)

Loss weighting
- tpn_loss_weight: \( \lambda_{TPN} > 0 \)
- causal_loss_weight: \( \lambda_{causal} > 0 \)

Beta schedules
- router.beta_schedule.type: “linear” or “cosine”
- router.beta_schedule.beta_ce_start, beta_ce_end: \( >0 \)
- router.beta_schedule.beta_cu_start, beta_cu_end: \( >0 \)
- router.beta_schedule.warmup_steps: \( S_w \ge 0 \)

Optional (advanced)
- Depth-wise capacity schedule \( \gamma_l \) (e.g., route every other layer or larger \( \gamma \) in deeper layers).
- Freeze base model flag (for PEFT scenarios).
- Router input normalization choice (RMSNorm/LayerNorm or identity).

8. Optimization & training
Parameter groups (recommended separation)
- Base model (Transformer blocks).
- TPN parameters.
- Predictive Router bias parameters (learnable \( o_{ce}, m_{cu} \)).
- Causal Router parameters.

Trainer responsibilities
- Compute and supply \( \beta_{ce}(s), \beta_{cu}(s) \) every step.
- Average per-layer \( \mathcal{L}_{TPN}, \mathcal{L}_{causal} \) across TDTF layers when combining.
- Optionally schedule \( \lambda_{TPN}, \lambda_{causal} \) (e.g., short warmup to zero).
- Gradient clipping, mixed precision, and other standard optimizations as desired.

9. Implementation notes
- Pre-norm architecture assumed; the router inputs are more stable if normalized (e.g., RMSNorm).
- MA computation: implement causal moving average per sequence and layer; reset at sequence boundaries (do not carry across batches).
- TopK tie-breaking: ensure deterministic tie-break or add a tiny \( \epsilon \)-noise to avoid flapping.
- Logging: track per-layer \( \text{mean}(D_{st}), \text{mean}(D_{ch}), \text{mean}(S_{CE}), \text{mean}(S_{CU}), \text{mean}(g), \) router agreement metrics, and selection ratios. Log \( \beta_{ce}, \beta_{cu} \).

10. Pseudocode
10.1 Training step (teacher–student)

Inputs: batch of sequences \( \{x_{1:T}\} \), current step \( s \), schedule params, capacity \( \gamma \), window \( \tau_w \), weights \( \lambda_{TPN}, \lambda_{causal} \).

- Compute \( \beta_{ce}, \beta_{cu} \leftarrow \) schedule(s)
- For each TDTF layer \( l \):
  1) Dense TF block:
     - \( x_t^{(l)} \leftarrow \text{TF-Block}(x_t^{(l-1)}) \) for all \( t \)
     - \( \Delta x_t^{(l)} \leftarrow x_t^{(l)} - x_t^{(l-1)} \)
  2) TPN:
     - \( \hat{\Delta x}_t^{(l)} \leftarrow \text{TPN}(x_{t-1}^{(l)}) \), with \( x_0^{(l)} = \mathbf{0} \)
     - \( \mathcal{L}_{TPN}^{(l)} \leftarrow \text{MSE}(\hat{\Delta x}_t^{(l)}, \text{stop\_grad}(\Delta x_t^{(l)})) \)
  3) Surprise and CE/CU:
     - \( D_{st,t}^{(l)}, D_{ch,t}^{(l)} \) as above
     - \( CE_t^{(l)}, CU_t^{(l)} \) via MA and learnable \( o_{ce}, m_{cu} \)
     - \( S_{CE,t}^{(l)} \leftarrow \sigma(\beta_{ce} \cdot CE_t^{(l)}) \), \( S_{CU,t}^{(l)} \leftarrow \sigma(\beta_{cu} \cdot CU_t^{(l)}) \)
     - \( g_t^{(l)} \leftarrow S_{CE,t}^{(l)} + S_{CU,t}^{(l)} - S_{CE,t}^{(l)} S_{CU,t}^{(l)} \)
  4) TopK selection per sequence:
     - \( m_t^{(l)} \leftarrow \text{TopK}(g_t^{(l)}, k=\lfloor \gamma T \rfloor) \)
  5) Student router:
     - \( \tilde{r}_t^{(l)} \leftarrow \tilde{R}^{(l)}(z_t^{(l-1)}) \)
     - \( \mathcal{L}_{causal}^{(l)} \leftarrow \text{BCE}(\sigma(\tilde{r}_t^{(l)}), m_t^{(l)}) \)

- Combine losses:
  - \( \mathcal{L}_{total} \leftarrow \mathcal{L}_{LM} + \lambda_{TPN}\cdot \frac{1}{|\mathcal{D}|}\sum_{l\in\mathcal{D}} \mathcal{L}_{TPN}^{(l)} + \lambda_{causal}\cdot \frac{1}{|\mathcal{D}|}\sum_{l\in\mathcal{D}} \mathcal{L}_{causal}^{(l)} \)

10.2 Inference step (student)
For AR decoding at step \( s \) (one new token per sequence):
- For each TDTF layer \( l \):
  1) Compute \( \tilde{r}^{(l)} \) for each batch element.
  2) Select tokens to process:
     - TopK: select \( \lfloor \gamma B \rfloor \) highest-prob elements.
     - Threshold: select where \( \sigma(\tilde{r}^{(l)}) > \tau \).
  3) Process selected examples through TF block; bypass others.
  4) Only processed tokens write K/V for layer \( l \).

11. Relationship to VPR and rationale
- CE is a model comparison (static vs change), akin to Bayesian model selection. CU guards against model-misspecification and bootstraps early learning.
- Scheduled inverse temperatures mimic VPR’s qualitative behavior: early “soft” decisions allow exploration; later “hard” gating sharpens selectivity.
- Using residuals \( \Delta x \) aligns with VPR’s emphasis on state changes rather than absolute values.

12. Failure modes and debugging
- If routing collapses to all on/off: examine \( \beta \) schedule endpoints, \( \lambda_{causal} \), and MA window; log CE/CU distributions.
- If student diverges from teacher: inspect BCE calibration, ensure TopK targets are correct and not leaking across batch or sequences; verify normalization of router inputs.
- If performance degrades with sparse compute: consider depth-wise capacity scheduling or allowing more tokens at deeper layers; monitor KV cache semantics.

13. Extensions (optional)
- Split routing for Q vs K/V per layer (e.g., allow a token to query but not be a key).
- Learned distance metrics (e.g., Mahalanobis in whitened space) for \( D_{st}, D_{ch} \).
- Mixed-mode training: after forming teacher targets, re-execute the layer in student-gated mode and backprop \( \mathcal{L}_{LM} \) through the masked path to better match inference compute.

14. Summary checklist (implementation)
- Provide all required hyperparameters via config (no code defaults).
- Schedule \( \beta_{ce}, \beta_{cu} \) and pass them to model forward every step.
- Compute CE/CU with per-sequence MA; learn \( o_{ce}, m_{cu} \).
- TopK per sequence/layer for teacher targets; TopK per batch step for student AR generation (or threshold).
- Skip full TF block and KV writes for unselected tokens.
- Average auxiliary losses across TDTF layers; weight by \( \lambda_{TPN}, \lambda_{causal} \).
- Log VPR signals, router stats, and betas; monitor agreement between teacher and student.

This specification contains the mathematical definitions, execution logic, training/inference procedures, scheduler mechanics, configuration requirements, and practical implementation notes needed to implement TDTF faithfully and reproducibly.