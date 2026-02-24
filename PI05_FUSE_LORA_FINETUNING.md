# Pi05 LoRA Finetuning with Textual Reasoning (Do What You Say)

## Overview

This document describes the implementation of LoRA finetuning for the pi05 base model on the LIBERO benchmark with textual planning alignment, inspired by the "Do What You Say" paper (Wu et al., 2025). The goal is to produce a model that:

1. **Performs well on LIBERO** by finetuning on LIBERO demonstration data
2. **Preserves textual planning** by incorporating a text cross-entropy loss on reasoning tokens alongside the action diffusion loss

The implementation lives inside the `openpi` codebase and references the `actalign` codebase ("Do What You Say" training pipeline).

---

## Architecture: Pi0Fuse

### Key Design Decision

The original pi05 codebase (`openpi`) does not include textual planning loss. The "Do What You Say" codebase (`actalign`) implements textual planning but targets the pi0 architecture. Our **Pi0Fuse** model bridges these by:

- Using pi05-compatible architecture (AdaRMSNorm in action expert, discretized state in text prompt) to load pi05 base weights
- Adding joint text reasoning loss + action diffusion loss from the "Do What You Say" recipe

### Model Class: `Pi0Fuse` (`src/openpi/models/pi0_fuse.py`)

**Config**: `Pi0FuseConfig` extends `BaseModelConfig` with:
- `pi05=True`: Enables AdaRMSNorm in action expert (matching pi05 weight structure)
- `diffusion_loss_coeff=1.0`: Weight for action loss in combined loss
- `paligemma_variant="gemma_2b_lora"` / `action_expert_variant="gemma_300m_lora"`: LoRA variants

**Architecture** (matching pi05 for weight loading):
- `PaliGemma.llm`: Dual-expert Gemma module (PaliGemma 2B + Action Expert 300M)
- `PaliGemma.img`: SigLIP vision encoder
- `action_in_proj`, `action_out_proj`: Action embedding/de-embedding
- `time_mlp_in`, `time_mlp_out`: Pi05-style AdaRMSNorm time conditioning (not `state_proj` + `action_time_mlp` used in pi0)

**Loss Computation** (`compute_loss`):
```
L = L_text + diffusion_loss_coeff * L_action
```

Where:
- **L_text** (Cross-Entropy): Computed on reasoning tokens. Uses `token_loss_mask` to select which tokens contribute. Only samples with `diffusion_loss_mask=False` contribute text loss (reasoning samples).
- **L_action** (Flow Matching MSE): Computed on denoised action predictions. Masked by `diffusion_loss_mask` - only samples with `diffusion_loss_mask=True` contribute action loss.

This mixed-loss formulation ensures:
- Samples with reasoning annotations train the model's text generation capability
- Samples without reasoning (action-only) train the action generation capability
- The model learns to interleave reasoning and action generation

### Inference Methods

- `prefill()`: Process image+text prefix, decide whether to reason or act
- `reason()`: Autoregressive generation of reasoning tokens
- `act()`: Diffusion denoising to generate actions
- `sample_actions()`: End-to-end action generation (prefill + denoise)

---

## Textual Planning Alignment

### Token Protocol (from "Do What You Say")

The tokenization uses special tokens to delineate sequence regions:

| Token | ID | Purpose |
|-------|-----|---------|
| `END_OF_PREFIX_TOKEN` | 257022 | Marks end of instruction/context prefix |
| `BEGIN_OF_REASONING` | 257020 | Marks start of reasoning output |
| `BEGIN_OF_ACTION` | 257021 | Marks start of action generation |
| `PALIGEMMA_EOS_TOKEN` | 1 | End of text sequence |

### Tokenization: `FusePaligemmaTokenizer` (`src/openpi/models/tokenizer.py`)

For each sample, the tokenizer produces:

**Reasoning mode** (when `thought` has 2 elements: prefix + suffix):
```
[BOS] <instruction_prefix> ; State: <discretized_state> [END_OF_PREFIX] [BEGIN_OF_REASONING] <reasoning_text> [EOS] [PAD...]
```
- Prefix tokens: bidirectional attention (`ar_mask=0`)
- Suffix tokens: causal attention (`ar_mask=1`)
- `diffusion_loss_mask=False`: no action loss, only text CE loss

**Action mode** (when `thought` has 1 element: prefix only):
```
[BOS] <instruction_prefix> ; State: <discretized_state> [END_OF_PREFIX] [BEGIN_OF_ACTION] [PAD...]
```
- `diffusion_loss_mask=True`: action loss applies, no text CE loss

### Loss Masking

| Field | Shape | Purpose |
|-------|-------|---------|
| `tokenized_prompt_mask` | `[B, L]` | Valid token positions (padding=False) |
| `token_ar_mask` | `[B, L]` | 0=bidirectional, 1=causal attention |
| `token_loss_mask` | `[B, L]` | Which tokens contribute to text CE loss |
| `diffusion_loss_mask` | `[B]` | Whether action diffusion loss applies per sample |

### Outdated Reasoning Handling

The dataset includes samples with "outdated" reasoning (reasoning that no longer matches the current state). The dataset loader probabilistically selects:
- **Normal reasoning**: prefix=instruction, suffix=current reasoning
- **Outdated reasoning**: prefix=old reasoning, suffix=new reasoning (with `think_with_outdated_thought=True`)
- **Act with outdated thought**: prefix=instruction, action mode (with `act_with_outdated_thought=True`)

This teaches the model to:
1. Generate updated reasoning when the current reasoning is outdated
2. Continue acting even when reasoning hasn't been updated

---

## Dataset Preparation

### Source: LIBERO-R Datasets

The dataset comes from the "Do What You Say" paper, available at:
- HuggingFace: `nvidia/libero-r-datasets` (contains `libero-10-r/`, `libero-100-r/`, `libero-100-basket-r/`)
- LeRobot format repos: `yilin-wu/libero-10`, `yilin-wu/libero-100` (used by actalign)

### Dataset Loader: `LiberoReasonDataset` (`src/openpi/policies/libero_reason_dataset.py`)

Extends `LeRobotDataset` to:
1. Load the LeRobot-formatted LIBERO data (images, states, actions)
2. Load reasoning annotations from `cot_simple.json`
3. For each sample, determine the reasoning context (prefix/suffix/outdated)
4. Return a dict with `image_1`, `image_wrist_1`, `state`, `actions`, `thought`, and reasoning flags

### Reasoning Annotations (`cot_simple.json`)

The JSON file maps episode indices to reasoning segments:
```json
{
  "0": {
    "segments": [
      {
        "content": "Instruction: put the black bowl on the plate",
        "start_step": 0,
        "end_step": 50,
        "updated_content": "Move gripper above the black bowl"
      },
      ...
    ]
  },
  ...
}
```

### Setup Instructions

1. **Download the LeRobot dataset**:
   ```bash
   # The dataset should be in LeRobot format. Use yilin-wu repos or convert nvidia data.
   # For libero-10 (testing):
   huggingface-cli download yilin-wu/libero-10 --repo-type dataset
   # For libero-100 (full training):
   huggingface-cli download yilin-wu/libero-100 --repo-type dataset
   ```

2. **Download the reasoning annotations**:
   ```bash
   # From nvidia/libero-r-datasets
   huggingface-cli download nvidia/libero-r-datasets libero-10-r/cot_simple.json --repo-type dataset
   huggingface-cli download nvidia/libero-r-datasets libero-100-r/cot_simple.json --repo-type dataset
   ```

3. **Place the cot_simple.json** in the right location:
   ```bash
   # Place at: <lerobot_home>/<repo_id>/cot_simple.json
   # Example for libero-10:
   mkdir -p ~/.cache/lerobot/yilin-wu/libero-10/
   cp /path/to/downloaded/cot_simple.json ~/.cache/lerobot/yilin-wu/libero-10/
   ```

4. **Set reasoning_json_path** in the training config:
   ```bash
   # Pass via CLI when running train.py:
   python scripts/train.py pi05_libero_reason_lora \
     --reasoning_json_path /path/to/lerobot/yilin-wu/libero-100/cot_simple.json
   ```
   Or modify the config directly in `config.py`.

---

## Training Configuration

### Config: `pi05_libero_reason_lora` (`src/openpi/training/config.py`)

| Parameter | Value | Notes |
|-----------|-------|-------|
| **Model** | `Pi0FuseConfig` | pi05-compatible with reasoning loss |
| `paligemma_variant` | `gemma_2b_lora` | LoRA on PaliGemma backbone |
| `action_expert_variant` | `gemma_300m_lora` | LoRA on action expert |
| `action_dim` | 32 | pi05 action dimension |
| `action_horizon` | 16 | Action chunk size |
| `max_token_len` | 415 | Max text sequence length |
| `diffusion_loss_coeff` | 1.0 | Action loss weight |
| **Data** | `LeRobotLiberoReasonDataConfig` | |
| `repo_id` | `yilin-wu/libero-100` | LeRobot LIBERO dataset |
| `use_reasoning` | True | Enable reasoning annotations |
| `use_wrist_image` | True | Include wrist camera |
| `use_outdated_reasoning` | True | Train on outdated reasoning |
| **Weights** | `pi05_base` | Loaded from GCS |
| **Freeze** | All LLM params except LoRA | Only LoRA adapters train |
| `ema_decay` | None | No EMA for LoRA |
| `batch_size` | 4 | Fits on A100 80GB |
| `num_train_steps` | 30,000 | |
| `lr_schedule` | Cosine decay | 5e-5 peak, 5e-6 min |

### Also Available: `pi05_libero_10_reason_lora`

Smaller config for testing on LIBERO-10 (10 tasks, ~10x less data):
- `num_train_steps=10,000`
- `repo_id=yilin-wu/libero-10`

### Running Training

```bash
# Activate virtual environment
source /path/to/mujoco_playground/.venv/bin/activate

# For LIBERO-100 (full training):
XLA_PYTHON_CLIENT_MEM_FRACTION=0.95 python scripts/train.py pi05_libero_reason_lora --exp-name=pi05_libero_reason_lora

# For LIBERO-10 (testing):
XLA_PYTHON_CLIENT_MEM_FRACTION=0.95 python scripts/train.py pi05_libero_10_reason_lora --exp-name=pi05_libero_10_reason_lora
```

---

## Data Pipeline

### Transform Chain (applied per sample)

1. **No Repack** (LiberoReasonDataset already returns correct format)
2. **`LiberoReasonInputs`** (`src/openpi/policies/libero_policy.py`):
   - Maps dataset keys to model format: `image_1` -> `image/base_0_rgb`, `image_wrist_1` -> `image/left_wrist_0_rgb`
   - Passes through `thought`, `act_with_outdated_thought`, `think_with_outdated_thought`
3. **`Normalize`** (with empty norm_stats: no normalization by default)
4. **`ResizeImages`**: Resizes to 224x224
5. **`FuseTokenizePrompt`** (`src/openpi/transforms.py`):
   - Consumes `thought` field, produces `tokenized_prompt`, `token_ar_mask`, `token_loss_mask`, `diffusion_loss_mask`
   - If `discrete_state_input=True`, discretizes state into 256 bins and appends to prompt
6. **`PadStatesAndActions`**: Zero-pads state (7->32) and actions (7->32)

### Data Flow Diagram

```
LiberoReasonDataset.__getitem__()
  |
  v
{image_1, image_wrist_1, state, actions, thought, flags}
  |
  v  LiberoReasonInputs
{image:{base_0_rgb, left_wrist_0_rgb, right_wrist_0_rgb}, image_mask, state, actions, thought, flags}
  |
  v  Normalize (no-op with empty stats)
  |
  v  ResizeImages (224x224)
  |
  v  FuseTokenizePrompt
{image, image_mask, state, actions, tokenized_prompt, tokenized_prompt_mask, token_ar_mask, token_loss_mask, diffusion_loss_mask}
  |
  v  PadStatesAndActions (dim=32)
  |
  v  Collation -> FuseObservation.from_dict()
```

---

## Files Modified/Created

### New Files

| File | Purpose |
|------|---------|
| `src/openpi/models/pi0_fuse.py` | Pi0Fuse model with joint text+action loss |
| `src/openpi/policies/libero_reason_dataset.py` | Dataset loader for LIBERO with reasoning |
| `test_fuse_pipeline.py` | End-to-end test for the pipeline |
| `PI05_FUSE_LORA_FINETUNING.md` | This documentation |

### Modified Files

| File | Changes |
|------|---------|
| `src/openpi/models/model.py` | Added `PI0_FUSE` to `ModelType`, `FuseObservation` dataclass, updated `preprocess_observation` to preserve `FuseObservation` |
| `src/openpi/models/tokenizer.py` | Added special tokens (`BEGIN_OF_REASONING`, `BEGIN_OF_ACTION`, `END_OF_PREFIX_TOKEN`), `FusePaligemmaTokenizer` class |
| `src/openpi/transforms.py` | Added `FuseTokenizePrompt` transform |
| `src/openpi/policies/libero_policy.py` | Added `LiberoReasonInputs` transform |
| `src/openpi/training/config.py` | Added `LiberoReasonDataConfig`, `LeRobotLiberoReasonDataConfig`, `PI0_FUSE` handling in `ModelTransformFactory`, training configs |
| `src/openpi/training/data_loader.py` | Added `LiberoReasonDataset` support in `create_torch_dataset`, `FuseObservation` support in `DataLoaderImpl` |

---

## Key Differences from Actalign

| Aspect | Actalign (Do What You Say) | Our Implementation |
|--------|---------------------------|-------------------|
| Base model | pi0 base weights | pi05 base weights |
| Finetuning | Full finetuning | LoRA finetuning |
| Action expert | No AdaRMSNorm | AdaRMSNorm (pi05-style) |
| State handling | Separate state_proj | Discretized state in text prompt |
| Gemma method | `embedder_decode` | `deembed` (openpi naming) |
| Data format | Custom LeRobot repos | Same repos (yilin-wu/*) |
| Config system | Custom `LiberoReasonTrainConfig` | Standard `TrainConfig` + `LeRobotLiberoReasonDataConfig` |

---

## Normalization

Currently, the pipeline runs **without normalization** (`norm_stats={}`). For production training, you should:

1. Compute norm stats using the existing openpi script:
   ```bash
   python scripts/compute_norm_stats.py --config-name=pi05_libero_reason_lora
   ```
2. The computed stats will be stored in `assets/pi05_libero_reason_lora/`
3. Subsequent training runs will automatically load and apply them

---

## Testing

### End-to-End Test

Run the comprehensive test suite:
```bash
source /path/to/mujoco_playground/.venv/bin/activate
cd pace/openpi
python test_fuse_pipeline.py
```

This tests:
1. Model creation with dummy config
2. Transform pipeline with synthetic data
3. Forward pass (compute_loss) correctness
4. Gradient computation and non-zero gradients
5. LoRA freeze filter structure
6. `preprocess_observation` preserves `FuseObservation` fields
7. Full training step simulation (3 steps with decreasing loss)

### Expected Output
```
ALL TESTS PASSED!
Loss trend: 13.6481 -> 13.6248 -> 13.5793
```

---

## Memory Considerations

On a single A100 80GB: so far tested max batch size: 16

LoRA finetuning comfortably fits on a single A100 80GB. For full finetuning, memory would be significantly higher (~70+ GB).
