"""Config for the Pi0TraceVLASingle model (MoE-free ablation of Pi0TraceVLAMoe).

``trace_vla_single`` is the ablation that removes the hard-routed MoE from **both**
non-VLM streams of ``Pi0TraceVLAMoe`` while keeping everything else identical:

  - Stream 0: PaliGemma 2B VLM (dense FFN; always full FT here).
  - Stream 1: Action expert — a **single dense** FFN (``gemma_300m``: width=1024,
              mlp_dim=4096, depth=18). This is exactly one expert of
              ``trace_vla_moe``'s 5-expert action MoE (``trace_moe_gemma_300m``),
              with the MoE routing dropped. Warm-started from pi05_base's ``mlp_1``.
  - Stream 2: Trace expert — a **single dense, lightweight** FFN
              (``gemma_trace_small``: width=512, mlp_dim=2048, depth=18). This is
              exactly one expert of ``trace_vla_moe``'s shrunk trace MoE
              (``trace_moe_small``), with the MoE routing dropped. Randomly
              initialized (its shape does not match pi05_base's action FFN).

What is **kept** as a (skill-routed) per-skill head — i.e. NOT part of the ablation:

  - The completion / progress-prediction MLP head stays a per-skill MLP routed by
    the dataset ``atomic_token`` (∈ {0..4}), exactly as in ``Pi0TraceVLAMoe``.
    ``num_completion_experts`` (=5, the LIBERO skill atoms) sizes this head; it is
    the only place routing survives. The trace/action streams carry no routing.

Everything else — conditioning (semantic-target Fourier + AdaRMS, EE row-0
inpainting clamp, appended target-anchor row), the overlay-image action input,
the dataset, the transforms, the three losses (action FM + trace FM + completion
regression), and all training tricks (anchor-age augmentation, scene/overlay
dropout, trace perturbation, image augmentation) — is inherited **unchanged**
from ``Pi0TraceVLAMoe``.
"""
from __future__ import annotations

import dataclasses
from typing import TYPE_CHECKING

import flax.nnx as nnx
import jax
import jax.numpy as jnp
from typing_extensions import override

from openpi.models import model as _model
from openpi.models import trace_observation as _trace_obs
import openpi.models.gemmoe as _gemmoe
from openpi.shared import array_typing as at
import openpi.shared.nnx_utils as nnx_utils

if TYPE_CHECKING:
    from openpi.models.pi0_trace_vla_single import Pi0TraceVLASingle


@dataclasses.dataclass(frozen=True)
class Pi0TraceVLASingleConfig(_model.BaseModelConfig):
    """Configuration for the MoE-free (single-head) trace-augmented VLA model.

    Three streams:
      - PaliGemma 2B (full FT).
      - Action expert: single dense FFN (``gemma_300m``); always full FT.
      - Trace expert:  single dense lightweight FFN (``gemma_trace_small``);
        always full FT and randomly initialized.

    No MoE / hard routing on either non-VLM stream. The only routing left is the
    per-skill completion head, sized by ``num_completion_experts``.
    """

    dtype: str = "bfloat16"
    paligemma_variant: _gemmoe.Variant = "gemma_2b"
    # Action expert is a single dense FFN (one expert of trace_vla_moe's action MoE).
    action_expert_variant: _gemmoe.Variant = "gemma_300m"
    # Trace expert is a single dense, lightweight FFN (one expert of trace_vla_moe's trace MoE).
    trace_expert_variant: _gemmoe.Variant = "gemma_trace_small"

    # Action chunk shape.
    action_dim: int = 32
    action_horizon: int = 10
    max_token_len: int = 200
    # We always run pi05-style (state token in the prompt, time via adaRMS).
    pi05: bool = True
    discrete_state_input: bool = False  # match AtomicVLA/TraceVLA on libero

    # Original camera frame (height, width) before ``resize_with_pad`` letterboxes it
    # to the 224x224 model input. Only used by the train-time geometric augmentation in
    # ``preprocess_trace_observation`` to keep the image-space trace/keypoint targets
    # aligned with the letterboxed content. ``None`` (default) = square source / no
    # letterbox, i.e. the LIBERO behaviour (this ablation is LIBERO-only). Kept for
    # exact parity with ``Pi0TraceVLAMoeConfig``.
    image_source_hw: tuple[int, int] | None = None

    # Trace head shape: (N, 2). N is the number of waypoints.
    trace_horizon: int = 20
    trace_dim: int = 2

    # Number of skill-specific experts in the **completion head** (the only routed
    # component left after dropping the trace/action MoEs). Pinned to 5 (LIBERO skill
    # atoms), matching ``trace_vla_moe``'s per-skill completion head.
    num_completion_experts: int = 5

    # When True, the trace stream is extended by one extra token whose value is
    # inpainting-clamped to the semantic target ``p_tgt`` (the same mechanism
    # already used for the current-EE clamp at row 0). Mirrors ``Pi0TraceVLAMoe``.
    append_target_anchor: bool = True

    # Loss weights.
    trace_loss_coeff: float = 1.0
    action_loss_coeff: float = 1.0
    completion_loss_coeff: float = 0.1

    # Fourier-encoding for AdaRMS conditioning on the semantic target point.
    fourier_num_freqs: int = 8

    # Completion head: shared compression dim and per-skill hidden dim.
    completion_shared_dim: int = 256
    completion_per_skill_hidden: int = 64

    @property
    @override
    def model_type(self) -> _model.ModelType:
        # Reuse the PI05 model_type for transform routing.
        return _model.ModelType.PI05

    @override
    def create(self, rng: at.KeyArrayLike) -> "Pi0TraceVLASingle":
        from openpi.models.pi0_trace_vla_single import Pi0TraceVLASingle  # noqa: PLC0415

        return Pi0TraceVLASingle(self, rngs=nnx.Rngs(rng))

    @override
    def inputs_spec(self, *, batch_size: int = 1) -> tuple[_trace_obs.TraceObservation, _model.Actions]:
        # Same TraceObservation schema as TraceVLA / TraceVLAMoe — same dataset and
        # transforms, just a different (MoE-free) architecture.
        image_spec = jax.ShapeDtypeStruct([batch_size, *_model.IMAGE_RESOLUTION, 3], jnp.float32)
        image_mask_spec = jax.ShapeDtypeStruct([batch_size], jnp.bool_)
        with at.disable_typechecking():
            observation_spec = _trace_obs.TraceObservation(
                images={
                    "base_0_rgb": image_spec,
                    "left_wrist_0_rgb": image_spec,
                    "right_wrist_0_rgb": image_spec,
                },
                image_masks={
                    "base_0_rgb": image_mask_spec,
                    "left_wrist_0_rgb": image_mask_spec,
                    "right_wrist_0_rgb": image_mask_spec,
                },
                state=jax.ShapeDtypeStruct([batch_size, self.action_dim], jnp.float32),
                tokenized_prompt=jax.ShapeDtypeStruct([batch_size, self.max_token_len], jnp.int32),
                tokenized_prompt_mask=jax.ShapeDtypeStruct([batch_size, self.max_token_len], jnp.bool_),
                token_ar_mask=jax.ShapeDtypeStruct([batch_size, self.max_token_len], jnp.int32),
                token_loss_mask=jax.ShapeDtypeStruct([batch_size, self.max_token_len], jnp.bool_),
                atomic_token=jax.ShapeDtypeStruct([batch_size], jnp.float32),
                semantic_target_xy=jax.ShapeDtypeStruct([batch_size, 2], jnp.float32),
                current_ee_xy=jax.ShapeDtypeStruct([batch_size, 2], jnp.float32),
                has_trace=jax.ShapeDtypeStruct([batch_size], jnp.bool_),
                has_overlay=jax.ShapeDtypeStruct([batch_size], jnp.bool_),
                progress=jax.ShapeDtypeStruct([batch_size], jnp.float32),
                diffusion_loss_mask=jax.ShapeDtypeStruct([batch_size], jnp.bool_),
                future_trace_xy=jax.ShapeDtypeStruct(
                    [batch_size, self.trace_horizon, self.trace_dim], jnp.float32
                ),
                overlay_images={
                    "base_0_rgb": image_spec,
                },
                overlay_image_masks={
                    "base_0_rgb": image_mask_spec,
                },
            )
        action_spec = jax.ShapeDtypeStruct([batch_size, self.action_horizon, self.action_dim], jnp.float32)
        return observation_spec, action_spec

    def get_freeze_filter(self) -> nnx.filterlib.Filter:
        """Freeze filter for the single (MoE-free) variant.

        Stream layout in the param tree (identical path structure to ``trace_vla_moe``,
        except the per-stream FFNs are dense ``mlp_1`` / ``mlp_2`` rather than
        ``moe_1`` / ``moe_2``):
          - paligemma (stream 0): ``llm/.../*_0`` paths (no suffix; ``_name(name, 0) == name``).
          - action expert (stream 1): ``llm/.../*_1`` paths (dense: ``mlp_1``).
          - trace expert  (stream 2): ``llm/.../*_2`` paths (dense: ``mlp_2``).

        ``trace_vla_single`` is full FT everywhere, so this returns ``nnx.Nothing``.
        The LoRA branch is kept for parity with the rest of the trace family (the
        ``_1`` / ``_2`` subtree regexes match both dense and MoE FFNs identically),
        but no LoRA variant is currently defined for the single ablation.
        """
        filters = []
        has_lora = False

        all_llm = nnx_utils.PathRegex(".*llm.*")
        action_expert_subtree = nnx_utils.PathRegex(".*llm.*(_1).*")
        trace_expert_subtree = nnx_utils.PathRegex(".*llm.*(_2).*")

        if "lora" in self.paligemma_variant:
            # Freeze stream 0 (paligemma) but leave both expert subtrees fully trainable.
            filters.append(all_llm)
            filters.append(nnx.Not(action_expert_subtree))
            filters.append(nnx.Not(trace_expert_subtree))
            has_lora = True

        if has_lora:
            # Keep LoRA adapters trainable inside the frozen subtree.
            filters.append(nnx.Not(nnx_utils.PathRegex(".*lora.*")))

        if not filters:
            return nnx.Nothing
        return nnx.All(*filters)
