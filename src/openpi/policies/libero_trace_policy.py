"""Input/output transforms for the TraceVLA model on LIBERO.

The dataset (``LiberoTraceDataset``) emits a dict with extra keys for the trace-VLA
pipeline. This module provides the input transform that maps that dict into the
shape the model expects (Observation-style dict with `image`, `image_mask`,
`overlay_image`, `overlay_image_mask`, `state`, etc., plus the trace-side scalars).
"""
from __future__ import annotations

import dataclasses

import einops
import numpy as np

from openpi import transforms
from openpi.models import model as _model
from openpi.models import tokenizer as _tokenizer
from openpi_client import image_tools


def _parse_image(image) -> np.ndarray:
    image = np.asarray(image)
    if np.issubdtype(image.dtype, np.floating):
        image = (255 * image).astype(np.uint8)
    if image.ndim == 3 and image.shape[0] == 3:
        image = einops.rearrange(image, "c h w -> h w c")
    return image


@dataclasses.dataclass(frozen=True)
class LiberoTraceInputs(transforms.DataTransformFn):
    """Pack a LiberoTraceDataset sample into the dict expected by the model.

    Mirrors :class:`openpi.policies.libero_policy.LiberoInputs` but additionally
    forwards trace-related fields (``semantic_target_xy``, ``current_ee_xy``,
    ``future_trace_xy``, ``has_trace``, ``has_overlay``, ``progress``,
    ``atomic_token``) and the overlay image dict.
    """

    model_type: _model.ModelType = _model.ModelType.PI05

    def __call__(self, data: dict) -> dict:
        base_image = _parse_image(data["observation/image"])
        wrist_image = _parse_image(data["observation/wrist_image"])

        inputs = {
            "state": data["observation/state"],
            "image": {
                "base_0_rgb": base_image,
                "left_wrist_0_rgb": wrist_image,
                "right_wrist_0_rgb": np.zeros_like(base_image),
            },
            "image_mask": {
                "base_0_rgb": np.True_,
                "left_wrist_0_rgb": np.True_,
                "right_wrist_0_rgb": np.False_,
            },
        }

        # Overlay base image (used for execution-mode forward pass).
        if "observation/overlay_image" in data:
            overlay_image = _parse_image(data["observation/overlay_image"])
            inputs["overlay_image"] = {"base_0_rgb": overlay_image}
            inputs["overlay_image_mask"] = {"base_0_rgb": np.True_}

        # Actions (training only).
        if "actions" in data:
            inputs["actions"] = data["actions"]

        # Trace-side fields.
        if "atomic_token" in data:
            inputs["atomic_token"] = np.asarray(data["atomic_token"], dtype=np.float32)
        if "semantic_target_xy" in data:
            inputs["semantic_target_xy"] = np.asarray(data["semantic_target_xy"], dtype=np.float32)
        if "current_ee_xy" in data:
            inputs["current_ee_xy"] = np.asarray(data["current_ee_xy"], dtype=np.float32)
        if "future_trace_xy" in data:
            inputs["future_trace_xy"] = np.asarray(data["future_trace_xy"], dtype=np.float32)
        if "has_trace" in data:
            inputs["has_trace"] = np.asarray(bool(data["has_trace"]))
        if "has_overlay" in data:
            inputs["has_overlay"] = np.asarray(bool(data["has_overlay"]))
        if "progress" in data:
            inputs["progress"] = np.asarray(data["progress"], dtype=np.float32)
        if "diffusion_loss_mask" in data:
            inputs["diffusion_loss_mask"] = np.asarray(bool(data["diffusion_loss_mask"]))

        # Forward the prompt (task instruction) for tokenization downstream.
        if "prompt" in data:
            inputs["prompt"] = data["prompt"]

        # Forward the skill's name and full parameterized text so `TraceTokenizePrompt`
        # can use them. The VLM is supposed to see the externally-selected skill (with
        # its arguments) instead of / on top of the raw LIBERO task instruction.
        if "skill_name" in data:
            inputs["skill_name"] = data["skill_name"]
        if "skill_text" in data:
            inputs["skill_text"] = data["skill_text"]

        return inputs


@dataclasses.dataclass(frozen=True)
class TraceResizeImages(transforms.DataTransformFn):
    """Resize both ``image`` and ``overlay_image`` dicts to the model resolution."""

    height: int
    width: int

    def __call__(self, data: dict) -> dict:
        if "image" in data:
            data["image"] = {k: image_tools.resize_with_pad(v, self.height, self.width) for k, v in data["image"].items()}
        if "overlay_image" in data:
            data["overlay_image"] = {k: image_tools.resize_with_pad(v, self.height, self.width) for k, v in data["overlay_image"].items()}
        return data


@dataclasses.dataclass(frozen=True)
class LiberoTraceOutputs(transforms.DataTransformFn):
    """Trim padded action dimensions back to the dataset's 7-dim action."""

    def __call__(self, data: dict) -> dict:
        if "actions" in data:
            data["actions"] = np.asarray(data["actions"][..., :7])
        return data


@dataclasses.dataclass(frozen=True)
class TraceTokenizePrompt(transforms.DataTransformFn):
    """Tokenize the prompt for the TraceVLA model.

    The TraceVLA design feeds the externally-selected *skill* (with parameters) into
    the VLA as its language input — the original task instruction is consumed by the
    high-level off-the-shelf VLM and is *not* needed by the VLA. We therefore use
    the parameterized skill expression as the prompt itself (e.g.
    ``PICKUP_FROM(white mug, table)``). If a parameterized form is unavailable for
    a given frame, we fall back to the stripped skill name, then to the raw task
    instruction (defensive default for frames that have no annotated skill segment).

    Tokens are written into the standard ``tokenized_prompt`` / ``tokenized_prompt_mask``
    fields. The skill is also hard-routed to the trace MoE via ``atomic_token``.
    """

    tokenizer: _tokenizer.PaligemmaTokenizer
    discrete_state_input: bool = False

    @staticmethod
    def _to_str(x) -> str:
        if isinstance(x, str):
            return x
        if isinstance(x, np.ndarray):
            try:
                return str(x.item())
            except Exception:
                return str(x)
        try:
            return x.item() if hasattr(x, "item") else str(x)
        except Exception:
            return str(x)

    def __call__(self, data: dict) -> dict:
        prompt = self._to_str(data.pop("prompt", ""))
        # Pop (not get) so these strings do not survive into the batched dict — the torch
        # default collate cannot stack variable-length strings; we no longer need them
        # downstream once the prompt is tokenized.
        skill_name = self._to_str(data.pop("skill_name", ""))
        skill_text = self._to_str(data.pop("skill_text", ""))

        # Per the TraceVLA design, the VLM sees the externally-selected skill, not the
        # raw task instruction. Prefer the parameterized expression; fall back gracefully.
        if skill_text:
            full_prompt = skill_text
        elif skill_name:
            full_prompt = skill_name
        else:
            full_prompt = prompt

        state = None
        if self.discrete_state_input:
            state = data.get("state")

        tokens, token_mask = self.tokenizer.tokenize(full_prompt, state=state)
        # Build trivial AR mask + loss mask (we don't predict any text):
        # - AR mask: 0 (prefix) for all valid tokens. We don't need autoregressive predictions.
        # - Loss mask: all False (no text loss in the TraceVLA model).
        token_ar_mask = np.zeros_like(tokens, dtype=np.int32)
        token_loss_mask = np.zeros_like(tokens, dtype=np.bool_)

        return {
            **data,
            "tokenized_prompt": np.asarray(tokens, dtype=np.int32),
            "tokenized_prompt_mask": np.asarray(token_mask, dtype=np.bool_),
            "token_ar_mask": token_ar_mask,
            "token_loss_mask": token_loss_mask,
        }
