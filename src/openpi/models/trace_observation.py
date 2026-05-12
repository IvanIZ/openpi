"""TraceObservation: extends Observation for the TraceVLA model.

Carries the extra per-sample fields produced by ``LiberoTraceDataset``:

  - ``atomic_token``           : float scalar in {0..K-1} (skill -> hard-routed expert id)
  - ``semantic_target_xy``     : float[2] normalized [0,1] semantic target pixel
  - ``current_ee_xy``          : float[2] normalized [0,1] EE pixel at this frame
  - ``future_trace_xy``        : float[N,2] normalized [0,1] resampled GT trace
                                  (only used as supervision target during training; passed
                                  in the actions slot of the model; here we just carry a
                                  placeholder shape so to_dict round-trips cleanly)
  - ``has_trace``              : bool[1]   True iff the trace is valid for this frame
  - ``has_overlay``            : bool[1]   True iff the base image already carries an overlay
                                  (set by the data loader's anchor-age augmentation)
  - ``diffusion_loss_mask``    : bool[1]   carried over for symmetry with AtomicObservation

Existing :class:`openpi.models.model.Observation` is *not* modified - this is a fresh subclass
that we register through a separate code path.
"""
from __future__ import annotations

import dataclasses
from typing import Generic, TypeVar

import augmax
import jax
import jax.numpy as jnp
import numpy as np
import torch
from flax import struct

from openpi.models import model as _model
from openpi.shared import image_tools as _img_tools
import openpi.shared.array_typing as at

ArrayT = TypeVar("ArrayT", bound=jax.Array | torch.Tensor | np.ndarray)


@at.typecheck
@struct.dataclass
class TraceObservation(_model.Observation, Generic[ArrayT]):
    """Observation for the TraceVLA model."""

    atomic_token: at.Float[ArrayT, "*b"] | None = None
    semantic_target_xy: at.Float[ArrayT, "*b k"] | None = None
    current_ee_xy: at.Float[ArrayT, "*b k"] | None = None
    has_trace: at.Bool[ArrayT, "*b"] | None = None
    has_overlay: at.Bool[ArrayT, "*b"] | None = None
    progress: at.Float[ArrayT, "*b"] | None = None
    diffusion_loss_mask: at.Bool[ArrayT, "*b"] | None = None

    # Resampled GT trace target for the trace flow-matching loss.
    future_trace_xy: at.Float[ArrayT, "*b n k"] | None = None

    # Overlay images (only `base_0_rgb` typically): used for execution-mode forward pass.
    # Wrist images are reused from `images`.
    overlay_images: dict[str, at.Float[ArrayT, "*b h w c"]] | None = None
    overlay_image_masks: dict[str, at.Bool[ArrayT, "*b"]] | None = None

    @classmethod
    def from_dict(cls, data: at.PyTree[ArrayT]) -> "TraceObservation[ArrayT]":
        if ("tokenized_prompt" in data) != ("tokenized_prompt_mask" in data):
            raise ValueError("tokenized_prompt and tokenized_prompt_mask must be provided together.")
        # Image normalization (uint8 -> [-1, 1] float32) — same convention as Observation.from_dict.
        for key in data["image"]:
            if data["image"][key].dtype == np.uint8:
                data["image"][key] = data["image"][key].astype(np.float32) / 255.0 * 2.0 - 1.0
            elif hasattr(data["image"][key], "dtype") and data["image"][key].dtype == torch.uint8:
                data["image"][key] = data["image"][key].to(torch.float32).permute(0, 3, 1, 2) / 255.0 * 2.0 - 1.0
        # Same conversion for overlay images, if present.
        overlay_images = data.get("overlay_image")
        if overlay_images is not None:
            for key in overlay_images:
                if overlay_images[key].dtype == np.uint8:
                    overlay_images[key] = overlay_images[key].astype(np.float32) / 255.0 * 2.0 - 1.0
                elif hasattr(overlay_images[key], "dtype") and overlay_images[key].dtype == torch.uint8:
                    overlay_images[key] = overlay_images[key].to(torch.float32).permute(0, 3, 1, 2) / 255.0 * 2.0 - 1.0
        return cls(
            images=data["image"],
            image_masks=data["image_mask"],
            state=data["state"],
            tokenized_prompt=data.get("tokenized_prompt"),
            tokenized_prompt_mask=data.get("tokenized_prompt_mask"),
            token_ar_mask=data.get("token_ar_mask"),
            token_loss_mask=data.get("token_loss_mask"),
            atomic_token=data.get("atomic_token"),
            semantic_target_xy=data.get("semantic_target_xy"),
            current_ee_xy=data.get("current_ee_xy"),
            has_trace=data.get("has_trace"),
            has_overlay=data.get("has_overlay"),
            progress=data.get("progress"),
            diffusion_loss_mask=data.get("diffusion_loss_mask"),
            future_trace_xy=data.get("future_trace_xy"),
            overlay_images=overlay_images,
            overlay_image_masks=data.get("overlay_image_mask"),
        )


_BASE_IMAGE_KEY = "base_0_rgb"


def preprocess_trace_observation(
    rng: at.KeyArrayLike | None,
    observation: TraceObservation,
    *,
    train: bool = False,
    image_keys=_model.IMAGE_KEYS,
    image_resolution: tuple[int, int] = _model.IMAGE_RESOLUTION,
) -> TraceObservation:
    """Resize and augment images for the TraceVLA model.

    Crucially, when ``train=True`` we apply the *same* random geometric transform
    (random crop + rotate, plus color jitter) to all of:

      - the base camera image (``base_0_rgb``),
      - the overlay version of the base image (``overlay_images['base_0_rgb']``),
      - the semantic-target keypoint (``semantic_target_xy``),
      - the current end-effector keypoint (``current_ee_xy``),
      - every waypoint of the supervised future trace (``future_trace_xy``).

    This keeps the image-space conditioning labels and the trace flow-matching target
    in correspondence with the visual content the VLM actually sees, fixing the
    misalignment that ``_model.preprocess_observation`` (image-only) would otherwise
    introduce. Wrist images receive a *separate* color-only chain (no geometric
    transform), matching ``_model.preprocess_observation``'s wrist policy.

    Out-of-bounds keypoints (e.g. caused by the random crop pulling a near-edge
    point past the boundary) are clamped back into the unit square. With 5%
    crop margin and ±5° rotation this is rare for LIBERO-style data because the
    workspace points sit well inside the camera frame.

    When ``train=False`` we only resize images to the target resolution; no augmax
    is applied. Overlay images and trace keypoints pass through unchanged.
    """
    H, W = image_resolution
    if not set(image_keys).issubset(observation.images):
        raise ValueError(f"images dict missing keys: expected {image_keys}, got {list(observation.images)}")

    # ---- Step 1: bring all images to the target resolution. ----
    raw_images: dict = {}
    for key in image_keys:
        img = observation.images[key]
        if img.shape[1:3] != image_resolution:
            img = _img_tools.resize_with_pad(img, *image_resolution)
        raw_images[key] = img

    raw_overlay: dict | None = None
    if observation.overlay_images is not None:
        raw_overlay = {}
        for key, img in observation.overlay_images.items():
            if img.shape[1:3] != image_resolution:
                img = _img_tools.resize_with_pad(img, *image_resolution)
            raw_overlay[key] = img

    sem_full = observation.semantic_target_xy
    cur_ee_full = observation.current_ee_xy
    future_trace_full = observation.future_trace_xy

    # ---- Step 2: training-time augmentation. ----
    if train and rng is not None:
        # Convert to [0, 1] for augmax, matching `_model.preprocess_observation`.
        for key in raw_images:
            raw_images[key] = raw_images[key] / 2.0 + 0.5
        if raw_overlay is not None:
            for key in raw_overlay:
                raw_overlay[key] = raw_overlay[key] / 2.0 + 0.5

        # Per-batch RNGs for vmap. Using the same `rng` here means base, overlay, and
        # keypoints share identical transform parameters per batch element. Wrist
        # ColorJitter also derives from the same `rng` split, mirroring the
        # behavior of `_model.preprocess_observation` (color jitter is consistent
        # across cameras within a sample).
        B = raw_images[_BASE_IMAGE_KEY].shape[0]
        sub_rngs = jax.random.split(rng, B)

        sem_xy = sem_full[..., :2]
        cur_ee_xy = cur_ee_full[..., :2]
        future_trace_xy = future_trace_full[..., :2]
        scale = jnp.asarray([W - 1, H - 1], dtype=jnp.float32)

        # ---- 2a. Geometric + color chain on base + overlay + keypoints ----
        has_overlay_base = raw_overlay is not None and _BASE_IMAGE_KEY in raw_overlay

        has_keypoints = True
        if has_keypoints:
            geom_input_types = [augmax.InputType.IMAGE]                   # base
            if has_overlay_base:
                geom_input_types.append(augmax.InputType.IMAGE)           # overlay
            geom_input_types.extend([augmax.InputType.KEYPOINTS] * 3)     # sem, ee, future_trace

            geom_chain = augmax.Chain(
                augmax.RandomCrop(int(W * 0.95), int(H * 0.95)),
                augmax.Resize(W, H),
                augmax.Rotate((-5, 5)),
                augmax.ColorJitter(brightness=0.3, contrast=0.4, saturation=0.5),
                input_types=geom_input_types,
            )

            # Convert keypoints to pixel coordinates in [0, W-1] x [0, H-1].
            sem_px = (sem_xy * scale)[:, None, :]              # (B, 1, K)
            ee_px = (cur_ee_xy * scale)[:, None, :]            # (B, 1, K)
            ft_px = future_trace_xy * scale                    # (B, N, K)

            inputs = [raw_images[_BASE_IMAGE_KEY]]
            if has_overlay_base:
                inputs.append(raw_overlay[_BASE_IMAGE_KEY])
            inputs.extend([sem_px, ee_px, ft_px])

            def _apply_geom(rng_b, *args):
                return geom_chain(rng_b, list(args))

            outs = jax.vmap(_apply_geom)(sub_rngs, *inputs)

            idx = 0
            raw_images[_BASE_IMAGE_KEY] = outs[idx]; idx += 1
            if has_overlay_base:
                raw_overlay[_BASE_IMAGE_KEY] = outs[idx]; idx += 1
            sem_aug = outs[idx]; idx += 1
            ee_aug = outs[idx]; idx += 1
            ft_aug = outs[idx]

            # Back to normalized [0, 1], clamped to image bounds.
            sem_full = sem_full.at[..., :2].set(jnp.clip(sem_aug[:, 0, :] / scale, 0.0, 1.0))
            cur_ee_full = cur_ee_full.at[..., :2].set(jnp.clip(ee_aug[:, 0, :] / scale, 0.0, 1.0))
            future_trace_full = future_trace_full.at[..., :2].set(jnp.clip(ft_aug / scale, 0.0, 1.0))
        else:
            # No keypoints provided — fall back to a plain image-only geom chain.
            base_chain = augmax.Chain(
                augmax.RandomCrop(int(W * 0.95), int(H * 0.95)),
                augmax.Resize(W, H),
                augmax.Rotate((-5, 5)),
                augmax.ColorJitter(brightness=0.3, contrast=0.4, saturation=0.5),
                input_types=[augmax.InputType.IMAGE],
            )
            raw_images[_BASE_IMAGE_KEY] = jax.vmap(
                lambda rng_b, img: base_chain(rng_b, [img])[0]
            )(sub_rngs, raw_images[_BASE_IMAGE_KEY])
            if has_overlay_base:
                raw_overlay[_BASE_IMAGE_KEY] = jax.vmap(
                    lambda rng_b, img: base_chain(rng_b, [img])[0]
                )(sub_rngs, raw_overlay[_BASE_IMAGE_KEY])

        # ---- 2b. Color-only chain on wrist images (and any other non-base, non-wrist keys) ----
        wrist_chain = augmax.Chain(
            augmax.ColorJitter(brightness=0.3, contrast=0.4, saturation=0.5),
            input_types=[augmax.InputType.IMAGE],
        )
        for key in image_keys:
            if key == _BASE_IMAGE_KEY:
                continue
            raw_images[key] = jax.vmap(
                lambda rng_b, img: wrist_chain(rng_b, [img])[0]
            )(sub_rngs, raw_images[key])

        # Convert images back to [-1, 1].
        for key in raw_images:
            raw_images[key] = raw_images[key] * 2.0 - 1.0
        if raw_overlay is not None:
            for key in raw_overlay:
                raw_overlay[key] = raw_overlay[key] * 2.0 - 1.0

    # ---- Step 3: build masks (default ones if missing). ----
    batch_shape = observation.state.shape[:-1]
    out_image_masks: dict = {}
    for key in image_keys:
        if observation.image_masks is not None and key in observation.image_masks:
            out_image_masks[key] = jnp.asarray(observation.image_masks[key])
        else:
            out_image_masks[key] = jnp.ones(batch_shape, dtype=jnp.bool_)

    overlay_image_masks = observation.overlay_image_masks if raw_overlay is not None else None
    overlay_images = raw_overlay  # may be None

    return TraceObservation(
        images=raw_images,
        image_masks=out_image_masks,
        state=observation.state,
        tokenized_prompt=observation.tokenized_prompt,
        tokenized_prompt_mask=observation.tokenized_prompt_mask,
        token_ar_mask=observation.token_ar_mask,
        token_loss_mask=observation.token_loss_mask,
        atomic_token=observation.atomic_token,
        semantic_target_xy=sem_full,
        current_ee_xy=cur_ee_full,
        has_trace=observation.has_trace,
        has_overlay=observation.has_overlay,
        progress=observation.progress,
        diffusion_loss_mask=observation.diffusion_loss_mask,
        future_trace_xy=future_trace_full,
        overlay_images=overlay_images,
        overlay_image_masks=overlay_image_masks,
    )
