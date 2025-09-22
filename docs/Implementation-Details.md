# Implementation Details

## Project Aim
The primary goal of this repository is to provide a clean, simplified, and highly readable codebase for training Transformer models based on the Qwen2.5 architecture. This includes supporting both transfer learning from pre-trained Hugging Face weights (e.g., 0.5B models) and training smaller models from scratch (e.g., 50M parameters) where pre-trained weights are unavailable. A key principle is to externalize all hardcoded variables into configuration files, ensuring a clear separation of concerns and adherence to DRY (Don't Repeat Yourself) principles. The overall aim is to maintain full functionality while significantly reducing code complexity and lines compared to previous versions.

## Features Implemented

### 1. Codebase Cleanup & Refactoring
*   **Docstring Removal:** Top-level docstrings have been removed from all Python files in the `src/` directory to reduce clutter and improve readability.
*   **Configuration Centralization:**
    *   Platform-specific settings (e.g., device, precision, Flash Attention usage) have been moved from `src/training/utils.py` to `config/system.yaml`.
    *   Model dimension presets for training from scratch have been moved from `src/training/utils.py` to `config/model/scratch_configs.yaml`.
    *   The `PlatformOptimizer` class in `src/training/utils.py` has been removed, and its logic is now handled via configuration.
    *   All individual model-specific configuration YAMLs (`dtf_scratch.yaml`, `mod_transfer.yaml`, etc.) have been consolidated into a single, comprehensive `config/train.yaml` using Hydra's `defaults` mechanism, aligning with the requested "old config style."
*   **`train.py` Refactoring:**
    *   The `train_step` function has been simplified to return `outputs` and `loss` directly, removing redundant `accelerator` handling within the function.
    *   The main training loop now correctly processes `outputs` for logging and handles gradient accumulation and clipping.
    *   A duplicate `evaluate` function definition was removed.

### 2. Wandb Logging
*   **Integration:** `wandb` has been integrated into `train.py` using `accelerator.init_trackers` for comprehensive experiment tracking.
*   **Metrics Logged:**
    *   **Standard:** Training loss, validation loss, training perplexity, validation perplexity.
    *   **Auxiliary Losses:** Prior loss (for DTF), causal loss (for DTF/MoD/TDTF), and MoD auxiliary loss are now extracted from model outputs and logged.
    *   **Router Statistics:** A `process_router_stats` helper function has been implemented to flatten and log detailed router statistics (e.g., `S_CE`, `S_CU`, `G_cont` for DTF; layer-wise stats for MoD; averaged stats for TDTF) to wandb.
*   **Perplexity Tracking:** Perplexity is calculated from the loss and logged for both training and validation.

### 3. Causal Routers for MoD and DTF
*   **Enhanced Causal Routers:** The `CausalDTFRouter` and `CausalMoDRouter` implementations have been updated to align with the `TDTF-Spec.md`'s approach. They now take a concatenated input of the current token's hidden state and the previous token's hidden state (`[x_t^(l-1) || x_{t-1}^(l-1)]`) for more robust causal decision-making during inference.

### 4. Top-K Selection for Non-Causal Router Training
*   **Verification:** It has been verified that `DTFDynamicLayer` and `MoDLayer` already correctly implement top-K selection based on `self.router.capacity` during training, fulfilling this requirement.

### 5. SwiGLU-like Gating for Prior Feed-Forwards
*   **Verification:** The `DTFPriorNetwork` already utilizes `Qwen2MLP` for its feed-forward mechanism. Assuming `Qwen2MLP` (from the `transformers` library) correctly implements the SwiGLU activation as per Qwen2.5 specifications, this requirement is met by design.

### 6. Multi-GPU Training and LoRA Support
*   **Multi-GPU:** The existing `accelerate` integration in `train.py` (using `Accelerator` and `accelerator.prepare`) inherently supports multi-GPU training.
*   **LoRA Integration:**
    *   A `peft` section has been added to `config/train.yaml` to configure LoRA.
    *   `train.py` now conditionally applies LoRA to the model using `peft.LoraConfig` and `peft.get_peft_model` if `cfg.peft.enabled` is true. Trainable parameters are printed after LoRA application.

### 7. Hugging Face Model Upload
*   **`push_to_hub` Function:** A `push_to_hub` utility function has been added to `src/training/utils.py`, leveraging `huggingface_hub.HfApi` to upload models and tokenizers to the Hugging Face Hub.
*   **Configuration:** A `push_to_hub` section has been added to `config/train.yaml` to enable/disable this feature and specify repository details.
*   **Integration:** `train.py` now calls `push_to_hub` after the final model save if the feature is enabled and the process is the main one.

## Remaining Tasks / Known Issues / Debugging Points

### Debugging Session Summary (September 18, 2025)

This section summarizes the issues encountered and resolutions applied during a debugging session focused on getting `uv run python train.py --config-name laptop_10m_wikitext` to execute successfully.

**1. `ValueError: bf16 mixed precision requires PyTorch >= 1.10 and a supported device.`**
*   **Initial Cause:** The `system.precision` was set to `bf16` in `config/system/default.yaml`, which was being loaded by default. The user's macOS environment (likely Apple Silicon) does not support `bf16` for `accelerate`.
*   **Resolution:**
    *   Modified `pyproject.toml` to constrain `torch` version for `darwin` to `<2.4.0` (to avoid potential `bf16` issues with newer `torch` versions on MPS) and added `lm_eval` as a dependency.
    *   The `system` block in `config/laptop_10m_wikitext.yaml` was explicitly set to `precision: "no"`, `torch_dtype: "float32"`, `device: cpu`, and `use_flash_attention: false`.
    *   Removed `  - system: default` from the `defaults` section of `config/laptop_10m_wikitext.yaml` to ensure its local `system` block takes precedence over the `system: default` loaded via `config/train.yaml`.

**2. `TypeError: Accelerator.init_trackers() got an unexpected keyword argument 'entity'`**
*   **Cause:** The `accelerate.Accelerator.init_trackers` method does not accept an `entity` argument directly.
*   **Resolution:** Removed the `entity` argument from the `accelerator.init_trackers` call in `train.py`.

**3. `omegaconf.errors.ConfigAttributeError: Key 'scratch_config' is not in struct full_key: model.scratch_config`**
*   **Cause:** The `create_model_config` function in `src/training/utils.py` expected `cfg.model.scratch_config` when `training.mode` was `scratch`, but this key was missing from the `model` configuration. `config/model/default.yaml` did not include the `scratch_config` definitions.
*   **Resolution:** Appended the `scratch_config` block (content from `config/model/scratch_configs.yaml`) directly to `config/model/default.yaml` under the `model` key. This ensures `cfg.model.scratch_config` is correctly populated.

**4. `NameError: name 'Dict' is not defined. Did you mean: 'dict'?`**
*   **Cause:** `Dict` and `Any` were used in type hints in `src/models/dtf/causalLM.py` without being imported from the `typing` module.
*   **Resolution:** Added `Dict` and `Any` to the `typing` import statement in `src/models/dtf/causalLM.py`.

**5. `ModuleNotFoundError: No module named 'src.models.dtf.causalLMs'`**
*   **Cause:** Typo in `src/models/dtf/model.py` where `causalLMs` (plural) was used instead of `causalLM` (singular) in an import statement.
*   **Resolution:** Corrected the import statement in `src/models/dtf/model.py` from `from .causalLMs import DTFForCausalLM` to `from .causalLM import DTFForCausalLM`.

**6. `NameError: name 'Dict' is not defined. Did you mean: 'dict'?` (in `src/models/mod/causalLM.py`)**
*   **Cause:** Same as issue 4, but in `src/models/mod/causalLM.py`.
*   **Resolution:** Added `Dict` and `Any` to the `typing` import statement in `src/models/mod/causalLM.py`.

**7. `ModuleNotFoundError: No module named 'src.models.mod.causalLMs'`**
*   **Cause:** Same as issue 5, but in `src/models/mod/model.py`.
*   **Resolution:** Corrected the import statement in `src/models/mod/model.py` from `from .causalLMs import MoDForCausalLM` to `from .causalLM import MoDForCausalLM`.

**8. `ModuleNotFoundError: No module named 'src.models.standard.causalLMs'`**
*   **Cause:** Same as issue 5, but in `src/models/standard/model.py`.
*   **Resolution:** Corrected the import statement in `src/models/standard/model.py` from `from .causalLMs import StandardTransformerForCausalLM` to `from .causalLM import StandardTransformerForCausalLM`.

**Current Remaining Tasks (from original `Implementation-Details.md`):**

*   **`lm_eval` Integration:** (Status: Not yet implemented, but `lm_eval` is now a dependency).
*   **General Code Quality / DRY / Best Practices:** (Status: Ongoing).
*   **Model-Specific `modules_to_save` for LoRA:** (Status: Placeholder in `config/train.yaml`).
*   **TDTF Inference Logic:** (Status: Inefficient, needs vectorization).
*   **Data Loading for Mixed Datasets:** (Status: `src/data/datasets.py` only supports `TextDataset`).

**Next Steps for Future Debugging:**

The current state should allow the model creation to proceed further. The next error will indicate the next point of failure.
