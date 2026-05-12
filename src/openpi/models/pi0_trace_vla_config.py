"""Config for the Pi0TraceVLA model."""
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
import openpi.models.gemmoe_trace as _gemmoe_trace
from openpi.shared import array_typing as at
import openpi.shared.nnx_utils as nnx_utils

if TYPE_CHECKING:
    from openpi.models.pi0_trace_vla import Pi0TraceVLA


@dataclasses.dataclass(frozen=True)
class Pi0TraceVLAConfig(_model.BaseModelConfig):
    """Configuration for the trace-augmented VLA model.

    Three streams:
      - PaliGemma 2B (LoRA-able)
      - Action expert gemma_300m (LoRA-able)
      - Trace expert (5-experts hard-routed MoE; full FT)

    The model jointly trains an action flow-matching head, a trace flow-matching head
    (over normalized 2-D pixel coords), and a per-skill MLP completion-progress head.
    """

    dtype: str = "bfloat16"
    paligemma_variant: _gemmoe.Variant = "gemma_2b"
    action_expert_variant: _gemmoe.Variant = "gemma_300m"
    trace_expert_variant: _gemmoe_trace.Variant = "trace_moe_gemma_300m"

    # Action chunk shape.
    action_dim: int = 32
    action_horizon: int = 10
    max_token_len: int = 200
    # We always run pi05-style (state token in the prompt, time via adaRMS).
    pi05: bool = True
    discrete_state_input: bool = False  # AtomicVLA uses False for libero; keep consistent.

    # Trace head shape: (N, 2). N is the number of waypoints.
    trace_horizon: int = 20
    trace_dim: int = 2
    target_dim: int = 2
    # Number of skill-specific experts in the trace MoE.
    num_trace_experts: int = 5
    share_trace_embedder: bool = False  # True to share an embedder between embedding trace tokens for the
                                        # trace generator, and the action generator.

    # When True, the trace stream is extended by one extra token whose value is
    # inpainting-clamped to the semantic target ``p_tgt`` (the same mechanism
    # already used for the current-EE clamp at row 0). This gives the trace
    # generator a hard, spatial anchor for the target — complementary to the
    # AdaRMS modulation pathway. The supervised flow-matching target for the
    # extra row is constructed by appending ``p_tgt`` to the dataset's
    # ``future_trace_xy``; the flow loss is masked at this extra row (just as
    # it is at row 0).
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
        # Reuse existing PI05 model_type for transform routing; the new model has
        # its own training script so this is mostly cosmetic.
        return _model.ModelType.PI05

    @override
    def create(self, rng: at.KeyArrayLike) -> "Pi0TraceVLA3D":
        print("Creating trace vla with trace dimension", self.trace_dim)
        if self.trace_dim == 2:
            from openpi.models.pi0_trace_vla import Pi0TraceVLA  # noqa: PLC0415
            return Pi0TraceVLA(self, rngs=nnx.Rngs(rng))
        else:
            from openpi.models.pi0_trace_vla import Pi0TraceVLA3D # noqa: PLC0415
            return Pi0TraceVLA3D(self, rngs=nnx.Rngs(rng))

    @override
    def inputs_spec(self, *, batch_size: int = 1) -> tuple[_trace_obs.TraceObservation, _model.Actions]:
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
                semantic_target_xy=jax.ShapeDtypeStruct([batch_size, self.target_dim], jnp.float32),
                current_ee_xy=jax.ShapeDtypeStruct([batch_size, self.trace_dim], jnp.float32),
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
        """Same logic as Pi0AtomicConfig but considers our 3-stream layer naming.

        Streams in our model:
          - paligemma : VLM, llm.../*_0 paths
          - action exp: llm.../*_1 paths
          - trace exp : llm.../*_2 paths (HardMoE - we DO NOT LoRA-freeze this stream)

        For LoRA finetune (our `trace_vla_lora`):
          - VLM (paligemma) is LoRA-frozen (variant ``gemma_2b_lora``)
          - Action expert is LoRA-frozen (variant ``gemma_300m_lora``)
          - Trace expert is *fully trainable* — its FFN is the new MoE we add and we want
            to learn it from scratch / pi05 init.
        """
        filters = []
        has_lora = False

        # Paligemma stream lives at llm/.../*_0 (no suffix).
        # We'll match it via "_0" path since _name(0) returns the bare name. See lora paths.
        # Use a broad "all llm" filter and exclude the action-expert/trace-expert subtrees.
        all_llm = nnx_utils.PathRegex(".*llm.*")
        action_expert_subtree = nnx_utils.PathRegex(".*llm.*(_1).*")
        trace_expert_subtree = nnx_utils.PathRegex(".*llm.*(_2).*")

        if "lora" in self.paligemma_variant:
            # Freeze VLM (stream 0 weights) but exclude streams 1 and 2.
            filters.append(all_llm)
            filters.append(nnx.Not(action_expert_subtree))
            filters.append(nnx.Not(trace_expert_subtree))
            # We may still LoRA-freeze the action expert independently; combine.
            if "lora" in self.action_expert_variant:
                # Re-add the action_expert subtree as frozen (the union of "VLM" and "action_expert" regions).
                filters[0] = nnx.Any(all_llm, action_expert_subtree)
                # Already exclude trace expert
                filters = [filters[0], nnx.Not(trace_expert_subtree)]
            has_lora = True
        elif "lora" in self.action_expert_variant:
            filters.append(action_expert_subtree)
            has_lora = True

        if has_lora:
            # If any LoRA is used, exclude all LoRA params from the freeze filter
            # so that LoRA params remain trainable.
            filters.append(nnx.Not(nnx_utils.PathRegex(".*lora.*")))

        if not filters:
            return nnx.Nothing
        return nnx.All(*filters)
