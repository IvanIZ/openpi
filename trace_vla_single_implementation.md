# TraceVLA-single: a MoE-free ablation of `trace_vla_moe`

This document describes the `trace_vla_single` training pipeline — an ablation of the
combined-MoE TraceVLA model (`trace_vla_moe`) that **drops the hard-routed Mixture-of-
Experts from both the trace generator and the action generator** and replaces each with
a single dense head. It records the full data flow, every change made relative to
`trace_vla_moe`, the warm-start logic, and the launch / norm-stat commands.

The guiding principle was the task spec: **keep everything identical to `trace_vla_moe`
except for dropping the MoE on the trace and action generators.** The dataset,
annotations, transforms, conditioning, losses, masks, and all data augmentation
(anchor-age augmentation, scene/overlay dropout, trace perturbation, trace overlay,
image augmentation) are inherited verbatim. The per-skill **completion / progress head is
deliberately kept** as a skill-routed per-skill MLP — it is *not* part of the ablation.

---

## 1. Launch commands

Both commands run inside the project venv used for all training
(`mujoco_playground/.venv`, activated by `pace/.pace_python.sh`).

**Compute normalization stats** (run once before training; writes
`assets/trace_vla_single/yilin-wu/libero-100/norm_stats.json`):

```bash
python pace/openpi/scripts/compute_norm_stats.py --config-name trace_vla_single
```

**Train** (full finetune, warm-started from `pi05_base`):

```bash
python pace/openpi/scripts/train_trace_vla_single.py trace_vla_single --exp-name trace_vla_single
```

This mirrors the `trace_vla_moe` launch
(`python pace/openpi/scripts/train_trace_vla_moe.py trace_vla_moe --exp-name trace_vla_moe`),
with the dedicated `train_trace_vla_single.py` entry point and the `trace_vla_single` config.

> Note on norm stats: `trace_vla_single` uses the **same dataset and the same
> state/action processing** as `trace_vla_moe`, so the resulting norm stats are numerically
> identical to `trace_vla_moe`'s. They are written to a config-name-specific path
> (`assets/<config_name>/<repo_id>/`, see [config.py:1051-1053](src/openpi/training/config.py#L1051-L1053)
> and [compute_norm_stats.py:111-113](scripts/compute_norm_stats.py#L111-L113)), so the
> `trace_vla_single` stats must be present at `assets/trace_vla_single/yilin-wu/libero-100/`.
> Equivalently you may copy an existing `trace_vla_moe` norm-stats file there.

---

## 2. What the ablation changes (and what it does *not*)

`trace_vla_moe` is a 3-stream Gemma trunk
([pi0_trace_vla_moe.py](src/openpi/models/pi0_trace_vla_moe.py)):

```
stream 0 : PaliGemma 2B VLM     (dense FFN)
stream 1 : action expert        (5-expert hard-routed MoE, trace_moe_gemma_300m: width 1024, mlp 4096)
stream 2 : trace  expert        (5-expert hard-routed MoE, trace_moe_small:      width 512,  mlp 2048)
```

`trace_vla_single` keeps the exact same wiring but makes **both non-VLM streams single
dense FFNs** — i.e. each MoE collapses to one expert of its original shape:

```
stream 0 : PaliGemma 2B VLM     (dense FFN)                       [unchanged]
stream 1 : action expert        (single dense FFN, gemma_300m:       width 1024, mlp 4096)
stream 2 : trace  expert        (single dense FFN, gemma_trace_small: width 512,  mlp 2048)  [lightweight, as in trace_vla_moe]
```

| Property | `trace_vla_moe` | `trace_vla_single` |
|---|---|---|
| Action head | 5-expert HardMoE (`trace_moe_gemma_300m`) | **single dense** (`gemma_300m`) |
| Trace head | 5-expert HardMoE (`trace_moe_small`, lightweight) | **single dense** (`gemma_trace_small`, lightweight) |
| Trace head is lighter than action head | ✓ (width 512 vs 1024) | ✓ (width 512 vs 1024) — preserved |
| Routing on trace / action streams | hard one-hot over 5 skill experts | **none** (no routing) |
| Completion / progress head | per-skill MLP, K=5, skill-routed | **per-skill MLP, K=5, skill-routed** — kept identical |
| Progress head is an MoE-style per-skill head | ✓ | ✓ — preserved (only routed component) |
| Dataset / annotations / transforms | LIBERO-100 + skill/trace annotations | identical |
| Conditioning, losses, masks, augmentations | (see §6) | identical |
| LoRA variant | `trace_vla_moe_lora` exists | none (full FT only, per spec) |

**Why a single dense FFN is exactly "one expert with routing dropped":** each MoE expert is
a SwiGLU FFN of shape `3 × width × mlp_dim` (see `traceVLA_moe_design.md` §1.2). The action
MoE's expert shape is `gemma_300m` (width 1024, mlp 4096); the trace MoE's expert shape is
`trace_moe_small` (width 512, mlp 2048). Dropping the MoE means using a single FFN of that
same shape — which is what `gemma_300m` / `gemma_trace_small` provide.

The trunk requires **no edit** to support this: `TraceBlock` already auto-dispatches the
per-stream FFN by `num_local_experts` — it builds a `HardMoeBlock` only when
`num_local_experts > 1`, and a dense `lora.FeedForward` otherwise
([gemmoe_trace.py:361-374](src/openpi/models/gemmoe_trace.py#L361-L374)). Setting both
non-VLM streams to `num_local_experts == 1` therefore yields dense `mlp_1` / `mlp_2` FFNs
with no further changes.

---

## 3. Files added / changed

### 3.1 New dense trace-expert variant — `src/openpi/models/gemmoe.py`

A single new dense variant `gemma_trace_small`
([gemmoe.py:101-115](src/openpi/models/gemmoe.py#L101-L115); registered in the `Variant`
literal at [gemmoe.py:70](src/openpi/models/gemmoe.py#L70)):

```
width=512, depth=18, mlp_dim=2048, num_heads=8, num_kv_heads=1, head_dim=256   (num_local_experts=1)
```

This is the MoE-free counterpart of `trace_moe_small` (the shrunk trace MoE in
`trace_vla_moe`, [gemmoe_trace.py:139-154](src/openpi/models/gemmoe_trace.py#L139-L154)):
identical `width`/`mlp_dim`, but a single dense FFN. `depth`, `num_heads`, `num_kv_heads`,
`head_dim` are locked by the joint-attention asserts against `gemma_2b`
([gemmoe_trace.py:202-204](src/openpi/models/gemmoe_trace.py#L202-L204),
[gemmoe_trace.py:402](src/openpi/models/gemmoe_trace.py#L402)). The action stream reuses the
existing dense `gemma_300m` ([gemmoe.py:90-99](src/openpi/models/gemmoe.py#L90-L99)) — no new
variant needed there.

### 3.2 New model config — `src/openpi/models/pi0_trace_vla_single_config.py`

`Pi0TraceVLASingleConfig` mirrors `Pi0TraceVLAMoeConfig`
([pi0_trace_vla_moe_config.py](src/openpi/models/pi0_trace_vla_moe_config.py)) with three
differences:

- Both expert variants are now **dense** and typed `_gemmoe.Variant`:
  `action_expert_variant = "gemma_300m"`, `trace_expert_variant = "gemma_trace_small"`
  ([pi0_trace_vla_single_config.py:78-81](src/openpi/models/pi0_trace_vla_single_config.py#L78-L81)).
- The `num_action_experts` / `num_trace_experts` fields are removed (no stream MoE) and
  replaced by a single `num_completion_experts: int = 5`
  ([pi0_trace_vla_single_config.py:90-93](src/openpi/models/pi0_trace_vla_single_config.py#L90-L93))
  that sizes the per-skill completion head — the only routed component.
- Everything else (action/trace horizons, `max_token_len`, `append_target_anchor`, loss
  coeffs, Fourier freqs, completion dims, `image_source_hw`, `inputs_spec`, `get_freeze_filter`)
  is copied verbatim from `Pi0TraceVLAMoeConfig`. `image_source_hw` is kept for exact parity
  (LIBERO sets it `None`, a no-op).

### 3.3 New model — `src/openpi/models/pi0_trace_vla_single.py`

`Pi0TraceVLASingle` ([pi0_trace_vla_single.py:105](src/openpi/models/pi0_trace_vla_single.py#L105))
is a near-verbatim copy of `Pi0TraceVLAMoe` with only the MoE removed. Concretely:

1. **Config sourcing** — both expert configs are pulled from `gemmoe`'s dense variants
   instead of `gemmoe_trace`'s MoE variants
   ([pi0_trace_vla_single.py:127-128](src/openpi/models/pi0_trace_vla_single.py#L127-L128)):
   ```python
   action_expert_config = _gemma.get_config(config.action_expert_variant)   # dense gemma_300m
   trace_expert_config  = _gemma.get_config(config.trace_expert_variant)    # dense gemma_trace_small
   ```
2. **Dense asserts** — both non-VLM streams must be dense (`num_local_experts <= 1`)
   ([pi0_trace_vla_single.py:132-142](src/openpi/models/pi0_trace_vla_single.py#L132-L142)),
   replacing `trace_vla_moe`'s "must have K experts" asserts. The trunk
   (`TraceModule`/`TraceBlock`) is unchanged and dispatches both streams to dense FFN.
3. **No stream routing** — the skill one-hot `hard_combine_weights` is no longer consumed by
   any FFN, so every `self.PaliGemma.llm(...)` call passes a tiny inert `(B, 1, 1)`
   placeholder from `_dummy_combine_weights`
   ([pi0_trace_vla_single.py:333-344](src/openpi/models/pi0_trace_vla_single.py#L333-L344)),
   used in `_forward_planning` ([:419](src/openpi/models/pi0_trace_vla_single.py#L419)),
   `_forward_execution` ([:494](src/openpi/models/pi0_trace_vla_single.py#L494)), and every
   sampling method ([:609, :628, :686, :708, :754, :800, :831](src/openpi/models/pi0_trace_vla_single.py#L609)).
   `trace_vla_moe`'s `_combine_weights` helper (the one-hot builder) is dropped entirely.
4. **Completion head kept (routed)** — the per-skill MLP completion head is identical to
   `trace_vla_moe`'s ([pi0_trace_vla_single.py:307-327](src/openpi/models/pi0_trace_vla_single.py#L307-L327)),
   sized and routed by `num_completion_experts` instead of `num_action_experts`
   ([:196](src/openpi/models/pi0_trace_vla_single.py#L196),
   [:326](src/openpi/models/pi0_trace_vla_single.py#L326)). The dataset `atomic_token`
   (∈ {0..4}, from `trace_utils.skill_to_expert_id`) still selects the per-skill MLP via a
   one-hot — this is the *only* routing left in the model, as the spec allows.

Everything else — the embedders, the action/trace I/O projections + time MLPs, the
semantic-target Fourier MLP, the AdaRMS conditioning, the row-0 EE inpainting clamp, the
appended target-anchor row, the three losses, and all five public methods — is byte-for-byte
the `trace_vla_moe` code with `combine_weights` → `dummy_weights`. The public method surface
(`compute_loss`, `sample_actions`, `sample_actions_and_completion`, `predict_completion`,
`sample_trace`) is unchanged, so the inference wrapper needs no edits (see §7).

### 3.4 New data factory + TrainConfig — `src/openpi/training/config.py`

- `LeRobotTraceVLASingleDataConfig`
  ([config.py:780](src/openpi/training/config.py#L780)) is a 3-line clone of
  `LeRobotTraceVLAMoeDataConfig` ([config.py:731](src/openpi/training/config.py#L731)) — same
  dataset, same repack/data/model transforms — differing only in the runtime type check, which
  binds to `Pi0TraceVLASingleConfig` ([config.py:797](src/openpi/training/config.py#L797)). It
  produces a `LiberoTraceDataConfig` identical in shape to `trace_vla_moe`'s, so
  `create_torch_dataset` ([data_loader.py:149-154](src/openpi/training/data_loader.py#L149-L154))
  and `compute_norm_stats` work unchanged.
- `trace_vla_single` TrainConfig ([config.py:2476](src/openpi/training/config.py#L2476)) is a
  copy of `trace_vla_moe` ([config.py:2305](src/openpi/training/config.py#L2305)) with only:
  - `model = Pi0TraceVLASingleConfig(action_expert_variant="gemma_300m",
    trace_expert_variant="gemma_trace_small", num_completion_experts=5, ...)`
    ([config.py:2479-2489](src/openpi/training/config.py#L2479-L2489));
  - `data = LeRobotTraceVLASingleDataConfig(repo_id="yilin-wu/libero-100", ...)` pointing at the
    **same** LIBERO-100 dataset and the **same** annotation files
    (`data/libero-100/skill_annotations.json`, `data/libero-100/skill_target_traces.json`).

  It is a full finetune (no `freeze_filter` ⇒ default `nnx.Nothing`,
  [config.py:1006](src/openpi/training/config.py#L1006)), warm-starts from `pi05_base`, and
  keeps `trace_vla_moe`'s optimizer / LR schedule / EMA / horizons / `max_token_len` and all
  `LiberoTraceDataConfig` augmentation defaults (anchor-age `h_train_max=15`,
  `scene_dropout_rate=0.15`, `overlay_dropout_rate=0.10`, `trace_perturb_max_sigma=0.03`,
  overlay color/thickness, `trace_horizon=20`).

### 3.5 New training script — `scripts/train_trace_vla_single.py`

A mirror of `scripts/train_trace_vla_moe.py`. The **only** difference is the pi05_base weight
remap (see §4); the data loader, train step, optimizer, EMA, checkpointing, and main loop are
identical. The data-loader type check binds to `LeRobotTraceVLASingleDataConfig`
([train_trace_vla_single.py:249](scripts/train_trace_vla_single.py#L249)).

---

## 4. Weight remap from `pi05_base`

`Pi0TraceVLASingle` warm-starts from the same `pi05_base` checkpoint as `trace_vla_moe`
(`gs://openpi-assets/checkpoints/pi05_base/params`). The remap
(`_load_and_filter_weights_single`,
[train_trace_vla_single.py:96-115](scripts/train_trace_vla_single.py#L96-L115)) is the
*simplest* of the trace family — a thin wrapper around `loader.load(params_shape)`:

- **Stream 0 (PaliGemma 2B VLM):** loaded as-is.
- **Stream 1 (action expert, dense `gemma_300m`):** pi05_base's dense FFN `mlp_1`
  (`gating_einsum` + `linear`) and the stream-1 attention / norm weights match the model's
  stream-1 keys **directly** — the single variant has `mlp_1`, not `moe_1/expert_*`, so **no MoE
  fan-out is needed** (contrast `train_trace_vla_moe._load_and_filter_weights_moe`
  [:135-151](scripts/train_trace_vla_moe.py#L135-L151), which fans `mlp_1` into 5 experts). The
  dense FFN parameter layout (`gating_einsum` of shape `(L, 2, in, hidden)` + `linear` of shape
  `(L, hidden, in)`, GELU gate) is exactly `lora.FeedForward`'s
  ([lora.py:96-106](src/openpi/models/lora.py#L96-L106)), the block `TraceBlock` instantiates
  for a dense stream — so the warm-start is element-wise exact (this is the same path the
  original `Pi0TraceVLA`'s dense action stream uses,
  [train_trace_vla.py:122-141](scripts/train_trace_vla.py#L122-L141)).
- **Stream 2 (trace expert, dense `gemma_trace_small`, width 512):** **randomly initialized.**
  Its width-512 attention/FFN shapes match nothing in pi05_base, so it is left at fresh random
  init — exactly like `trace_vla_moe`'s trace MoE (`trace_vla_moe` notes its trace stream is
  randomly initialized for the same reason, [config.py:2300-2302](src/openpi/training/config.py#L2300-L2302)).
  Unlike the original `Pi0TraceVLA` we **do not** copy stream-1 weights into stream-2
  (contrast [train_trace_vla.py:94-116](scripts/train_trace_vla.py#L94-L116)), since the
  attention projections differ in `width`.
- **Completion head, time MLPs, action/trace I/O projections, semantic-target Fourier MLP:**
  not present in pi05_base ⇒ random init.

Mechanically, `CheckpointWeightLoader.load`
([weight_loaders.py:50-54](src/openpi/training/weight_loaders.py#L50-L54)) runs `_merge_params`
with `missing_regex=".*lora.*"`: it (a) keeps exactly the loaded keys that exist in the model's
reference state — stream 0 + stream 1 (incl. dense `mlp_1`) + matching norms — and (b) back-fills
missing LoRA adapters (none here, full FT). Every other model key (stream 2, heads) is absent
from the returned dict and therefore stays at random init when `update_params`
([train_trace_vla_single.py:118-127](scripts/train_trace_vla_single.py#L118-L127)) overlays the
partial dict onto the freshly-initialized state. This is why no custom fan-out / copy logic is
needed: `loader.load` already produces precisely the intended partial warm-start.

---

## 5. End-to-end data flow

```
train_trace_vla_single.py  trace_vla_single
  └─ main()                                              scripts/train_trace_vla_single.py
     └─ _create_trace_data_loader()                      scripts/train_trace_vla_single.py:245
        ├─ isinstance(config.data, LeRobotTraceVLASingleDataConfig)  ✓   (:249)
        ├─ data_config = config.data.create(...)         -> LiberoTraceDataConfig
        │    (LeRobotTraceVLASingleDataConfig.create     config.py:793)
        │     repack = Group()  (empty)
        │     data_transforms.inputs  = [LiberoTraceInputs]
        │     data_transforms.outputs = [LiberoTraceOutputs]
        │     model_transforms.inputs = [TraceResizeImages(224,224),
        │                                TraceTokenizePrompt(PaligemmaTokenizer(200)),
        │                                PadStatesAndActions(32)]
        ├─ dataset = LiberoTraceDataset(data_config, action_horizon=10)
        │    (same loader, annotations, 256x256 LIBERO frames as trace_vla_moe)
        ├─ transform_dataset(dataset, data_config, skip_norm_stats=False)
        └─ TorchDataLoader(..., num_workers=config.num_workers)

LiberoTraceDataset.__getitem__ -> dict: observation/{image,wrist_image,overlay_image},
   state(8), actions(10,7), atomic_token, semantic_target_xy, current_ee_xy,
   future_trace_xy(20,2), has_trace, has_overlay, progress, ...
   (identical to trace_vla_moe — same anchor-age aug, scene/overlay dropout,
    trace perturbation, overlay rendering)

LiberoTraceInputs -> TraceResizeImages(224) -> Normalize -> TraceTokenizePrompt
   -> PadStatesAndActions(32) -> TraceObservation.from_dict -> Pi0TraceVLASingle.compute_loss
```

The model forward (`Pi0TraceVLASingle`):

- `compute_loss` ([pi0_trace_vla_single.py:509](src/openpi/models/pi0_trace_vla_single.py#L509))
  runs `preprocess_trace_observation(..., image_source_hw=self.config.image_source_hw)`
  ([:528](src/openpi/models/pi0_trace_vla_single.py#L528)) — same train-time geometric+color
  augmentation chain as `trace_vla_moe` (a no-op letterbox for square LIBERO frames), then the
  trace planning + action execution forwards.
- `_forward_planning` ([:348](src/openpi/models/pi0_trace_vla_single.py#L348)): clean image
  prefix + trace suffix; the **single dense trace FFN** (stream 2) is active; AdaRMS on
  time + Fourier(semantic target); row-0 EE inpainting clamp (+ appended target-anchor row);
  trace flow-matching target; `hard_combine_weights=dummy_weights`.
- `_forward_execution` ([:439](src/openpi/models/pi0_trace_vla_single.py#L439)): overlay image
  prefix + action suffix; the **single dense action FFN** (stream 1) is active; AdaRMS on time;
  action flow-matching target; `hard_combine_weights=dummy_weights`; then the **per-skill
  completion head** routed by `atomic_token`.
- Losses = action FM + trace FM + completion regression, with the identical `has_trace`/
  loss-mask handling as `trace_vla_moe`
  ([pi0_trace_vla_single.py:531-557](src/openpi/models/pi0_trace_vla_single.py#L531-L557)).

---

## 6. Behaviors inherited unchanged from `trace_vla_moe`

All of the following are reused verbatim (no single-variant-specific code):

- Dataset, repo (`yilin-wu/libero-100`), and **both annotation files**
  (`skill_annotations.json`, `skill_target_traces.json`) — identical to `trace_vla_moe`.
- Anchor-age augmentation for receding-horizon training; scene dropout; overlay dropout; smooth
  low-frequency trace perturbation; trace overlay rendering (cyan polyline, same
  color/thickness/endpoints) — all in `LiberoTraceDataset` / `trace_utils`, unchanged.
- Train-time geometric + color image augmentation applied jointly to base/overlay/keypoints
  (`preprocess_trace_observation`) — same chain, same `image_source_hw=None` (square LIBERO).
- Trace flow-matching with EE row-0 inpainting clamp + appended semantic-target anchor; action
  flow-matching on the overlay image; per-skill completion regression — same math, same masks,
  same loss weights (`action_loss_coeff=1.0`, `trace_loss_coeff=1.0`, `completion_loss_coeff=0.1`).
- Prompt construction (`"Plan: ... Current step: K. <skill_text>"`) and tokenization.
- Optimizer (AdamW, grad-clip 1.0), LR schedule (cosine, warmup 10k, peak 5e-5), EMA 0.999,
  `num_train_steps=120_000`, `batch_size=64`, `max_token_len=200`, horizons (`action_horizon=10`,
  `trace_horizon=20`) — all copied from `trace_vla_moe`.

The completion head is intentionally **kept routed** (per-skill MLP, `num_completion_experts=5`)
because progress prediction is explicitly out of scope for this ablation.

---

## 7. Inference

No change is required to the inference script
([py_script/libero_traceVLA_test.py](../../py_script/libero_traceVLA_test.py)). It dispatches on
`--model-name` via `_config.get_config(...)` and wraps the model with the generic
`create_trained_trace_vla_policy`
([policy_config.py:289](src/openpi/policies/policy_config.py#L289)), which only requires the
methods `sample_actions_and_completion`, `sample_trace`, and `predict_completion` — all present
in `Pi0TraceVLASingle` with identical signatures to `trace_vla_moe`. The input formatting
(`make_planning_obs` / `make_execution_obs`, `atomic_token` from `skill_to_expert_id`, the
overlay rendering, the EE-projection inpainting signal) is unchanged. To evaluate, run with:

```bash
python py_script/libero_traceVLA_test.py \
    --model-name trace_vla_single \
    --checkpoint-dir /work/hdd/bgtb/$USER/checkpoints/trace_vla_single/trace_vla_single/<step>
```

(The `atomic_token` still drives the **completion head's** per-skill routing; it no longer
affects the trace/action heads, which are now MoE-free — consistent with this ablation.)

---

## 8. Validation performed (sandbox, CPU, no full model)

Lightweight checks run in the project venv (no checkpoint load, no SigLIP/gemma-2b build, no
real inference), all passing:

1. `gemma_trace_small` resolves to a dense `(width=512, mlp_dim=2048, depth=18)` config with
   `num_local_experts==1` and the gemma_2b head-shape contract; `gemma_300m` likewise dense.
2. A tiny 3-stream `TraceModule` with all-dense configs builds dense `mlp_1`/`mlp_2` and **no**
   `moe_*`/`expert_*` params, and forwards correctly with one stream `None` per pass while the
   `(B,1,1)` `hard_combine_weights` placeholder is ignored — and the contrast case
   (`num_local_experts>1`) does build `moe_*`/`expert_*`, confirming the dispatch keys solely on
   expert count.
3. The completion-head per-skill einsums produce `(B,)` logits from a `(B, K)` per-skill output.
4. `get_config("trace_vla_single")` registers exactly once and constructs with the expected
   fields (dense variants, `num_completion_experts=5`, horizons, `append_target_anchor=True`,
   full-FT freeze = `nnx.Nothing`, correct repo + annotation paths, `pi05_base` warm-start).
5. `LeRobotTraceVLASingleDataConfig.create(...)` produces a `LiberoTraceDataConfig` with the
   correct transforms and correctly type-checks (accepts `Pi0TraceVLASingleConfig`, rejects
   `Pi0TraceVLAMoeConfig`).
6. The `Pi0TraceVLASingle` model module imports cleanly and exposes all required public methods.

Full model instantiation, checkpoint loading, and rollout were intentionally **not** run here
(GPU-cluster job); they are left to be launched with the commands in §1.
