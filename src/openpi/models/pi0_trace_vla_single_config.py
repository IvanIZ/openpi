"""Config for the Pi0TraceVLASingle model (MoE-free / single dense head ablation).

The "single" variant of TraceVLA uses a single dense FFN on **both** non-VLM
streams (no hard-routed MoE on either):

  - Stream 0: PaliGemma 2B VLM (dense FFN, optional LoRA).
  - Stream 1: Action expert — single dense FFN; full FT. Defaults to
              ``gemma_300m`` (width=1024, mlp_dim=4096, depth=18), the same shape
              as one expert of ``trace_vla_moe``'s action MoE, so it warm-starts
              from ``pi05_base``'s dense action FFN.
  - Stream 2: Trace expert  — single dense lightweight FFN; full FT. Defaults to
              ``gemma_trace_small`` (width=512, mlp_dim=2048, depth=18), the same
              shape as one expert of ``trace_moe_small``. Randomly initialized
              (no matching dense FFN in ``pi05_base``).

This is the controlled MoE-free ablation of ``Pi0TraceVLAMoe``: identical in
every other respect (dataset, ``atomic_token`` skill conditioning, transforms,
losses, and training tricks), it only drops the per-skill MoE on the action and
trace generators. The per-skill MLP completion head is KEPT and stays
skill-routed over ``num_completion_experts`` (=5 LIBERO skill atoms).
"""
from __future__ import annotations

import dataclasses
from typing import TYPE_CHECKING

import flax.nnx as nnx
from typing_extensions import override

from openpi.models import model as _model
from openpi.models import pi0_config as _pi0_config
from openpi.models import trace_observation as _trace_obs
import openpi.models.gemmoe as _gemmoe
from openpi.shared import array_typing as at

if TYPE_CHECKING:
    from openpi.models.pi0_trace_vla_single import Pi0TraceVLASingle


@dataclasses.dataclass(frozen=True)
class Pi0TraceVLASingleConfig(_model.BaseModelConfig):
    """Configuration for the MoE-free (single dense head) trace-augmented VLA model.

    Three streams:
      - PaliGemma 2B (LoRA-able).
      - Action expert: single dense FFN (``gemma_300m``); always full FT.
      - Trace expert:  single dense lightweight FFN (``gemma_trace_small``);
        always full FT and randomly initialized.

    No MoE / hard routing on either non-VLM stream. The only routed component left
    is the per-skill completion head, sized by ``num_completion_experts``.
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
    # letterbox, i.e. the LIBERO behaviour. Kept for exact parity with ``Pi0TraceVLAMoeConfig``.
    image_source_hw: tuple[int, int] | None = None

    # Trace head shape: (N, 2). N is the number of waypoints.
    trace_horizon: int = 20
    trace_dim: int = 2
    # Number of skill-specific experts in the **completion head** — the only routed
    # component left after dropping the action/trace MoEs. Pinned to 5 (LIBERO skill
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
        return _trace_obs.trace_inputs_spec(self, batch_size=batch_size)

    def get_freeze_filter(self) -> nnx.filterlib.Filter:
        """Freeze filter for the single (MoE-free) variant.

        Stream layout in the param tree:
          - paligemma (stream 0): ``llm/.../*_0`` paths (no suffix).
          - action expert (stream 1): ``llm/.../*_1`` paths (dense: ``mlp_1``).
          - trace expert  (stream 2): ``llm/.../*_2`` paths (dense: ``mlp_2``).

        For a ``..._lora`` variant (paligemma LoRA, both dense experts full FT):
            freeze = (paligemma subtree) AND NOT (LoRA params).
        For full FT everywhere: no freeze. Same handling as ``Pi0TraceVLAMoeConfig``.
        """
        return _pi0_config.llm_freeze_filter(
            self.paligemma_variant, self.action_expert_variant, expert_suffixes=("_1", "_2")
        )
