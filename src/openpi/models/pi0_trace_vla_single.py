"""Pi0TraceVLASingle: MoE-free (single dense head) ablation of ``Pi0TraceVLAMoe``.

Three-stream Gemma trunk:

  - Stream 0: PaliGemma 2B VLM (dense FFN, optional LoRA).
  - Stream 1: action expert (single dense FFN, full FT). Default ``gemma_300m``
              (width=1024, mlp_dim=4096, depth=18) — the same shape as *one*
              expert of ``trace_vla_moe``'s action MoE, so it warm-starts from
              ``pi05_base``'s dense action FFN.
  - Stream 2: trace  expert (single dense lightweight FFN, full FT). Default
              ``gemma_trace_small`` (width=512, mlp_dim=2048, depth=18) — the
              same shape as one expert of ``trace_moe_small``. Randomly
              initialized (no matching dense FFN in ``pi05_base``).

This is the controlled ablation of ``Pi0TraceVLAMoe`` that isolates the benefit
of the hard-routed MoE on the two non-VLM streams: everything else (dataset,
annotations, transforms, conditioning, the three losses, and all training tricks)
is identical to ``trace_vla_moe`` — both non-VLM streams simply drop the K-expert
MoE in favor of a single dense FFN. The per-skill **completion head** is KEPT and
stays skill-routed over ``num_completion_experts`` skills; it is not part of the
ablation.

Because both expert streams are dense (``num_local_experts == 1``), the base's
``action_is_moe`` / ``trace_is_moe`` flags are both False, so every forward pass
feeds a placeholder ``combine_weights`` that the dense FFNs ignore (the
``HardMoeBlock`` is never instantiated). The skill one-hot is still consumed by
the completion head via ``self.num_skills``.
"""
from __future__ import annotations

from typing_extensions import override

import openpi.models.gemmoe as _gemma
from openpi.models.pi0_trace_vla_base import TraceVLABase


# ---------------------------------------------------------------------------
# Pi0TraceVLASingle model
# ---------------------------------------------------------------------------

class Pi0TraceVLASingle(TraceVLABase):
    """Trace-augmented VLA with a single dense FFN on both the action and trace heads.

    Inherits everything from ``TraceVLABase`` (trunk/head construction, embeds, forward passes,
    loss, and sampling); the only override is ``_build_expert_configs`` — both expert streams are
    single dense FFNs (no MoE), so neither consumes the routing one-hot. The completion head still
    routes over ``num_completion_experts`` skills (``self.num_skills``).
    """

    @override
    def _build_expert_configs(self, config):
        """Both the action and the trace stream are single dense FFNs (``gemmoe`` dense configs).

        Unlike ``Pi0TraceVLAMoe``, ``self.num_skills`` is *decoupled* from the expert streams: the
        streams carry exactly one expert each (dense), while the completion head stays skill-routed
        over ``config.num_completion_experts`` skills. The base reads ``self.num_skills`` only for
        the completion head and the (here unused) placeholder combine weights, so the dense FFNs
        ignore routing entirely.
        """
        self.num_skills = int(config.num_completion_experts)
        paligemma_config = _gemma.get_config(config.paligemma_variant)
        # Both expert streams are single dense FFNs (gemmoe dense configs, num_local_experts == 1).
        action_expert_config = _gemma.get_config(config.action_expert_variant)
        trace_expert_config = _gemma.get_config(config.trace_expert_variant)

        if int(getattr(action_expert_config, "num_local_experts", 1)) != 1:
            raise ValueError(
                f"action_expert_variant must be a dense (single-expert) config for the single "
                f"variant; got num_local_experts={action_expert_config.num_local_experts}."
            )
        if int(getattr(trace_expert_config, "num_local_experts", 1)) != 1:
            raise ValueError(
                f"trace_expert_variant must be a dense (single-expert) config for the single "
                f"variant; got num_local_experts={trace_expert_config.num_local_experts}."
            )
        return paligemma_config, action_expert_config, trace_expert_config
