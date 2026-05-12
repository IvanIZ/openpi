"""Visualize per-frame inputs/targets as the model sees them during training.

Loads the trace dataset, runs a single frame through the same image transforms used
at training time (``LiberoTraceInputs`` + ``TraceResizeImages`` + the augmax random
crop / rotate / color jitter inside ``preprocess_trace_observation``), then writes
either the *execution-mode* overlay image (default) or the *planning-mode* image
plus its supervised trace target + EE/semantic-target dots (``--plan-mode``) to a
PNG. Press Enter to advance one ``--step-interval`` within the current episode;
press ``q`` (or Ctrl-C) to quit. When the current episode runs out of frames the
script jumps to step 0 of the next episode.

Modes:

  - **Default (execution mode)**: saves ``obs.overlay_images['base_0_rgb']`` — the
    input the action expert sees in execution-mode training. The polyline is drawn
    dataset-side by ``draw_polyline_overlay`` (anchor-aged trace, current overlay
    color/thickness/endpoint settings); we do *not* re-draw it here. The saved
    PNG already reflects two training-time augmentations applied dataset-side:

      * If the per-frame **overlay dropout** fired (default rate 10%) the saved
        PNG will be the *clean base image* with no polyline at all — the action
        head learns to act without the trace cue on that frame.
      * The polyline you see has already been bent by the per-sample **smooth
        low-frequency trace perturbation** (default
        ``trace_perturb_max_sigma=0.03``). That is the actual signal the action
        head trains on; the un-perturbed reference is *not* recoverable from the
        rendered image and is not emitted by the dataset, so we cannot draw it
        as an overlay here without modifying the training pipeline. To compare
        perturbed vs un-perturbed visually, run the script twice with the same
        ``--seed`` and toggle ``trace_perturb_max_sigma`` between 0.0 and the
        training value (e.g. via a temporary edit to the ``LiberoTraceDataConfig``
        in ``training/config.py``).

  - **``--plan-mode``**: saves ``obs.images['base_0_rgb']`` — the clean image the
    trace generator sees in planning-mode training — and *visually* overlays the
    supervised target trace (``obs.future_trace_xy``, anchored at ``t_now``) plus
    two filled disks: green for the current EE keypoint and red for the semantic
    target keypoint. All three overlays are drawn from the *post-augmax*
    keypoints, so they line up with the cropped/rotated image. The on-screen
    annotations are visualization-only — at training time these signals enter the
    model as conditioning (Fourier+MLP for the semantic target, an inpainting
    clamp on row 0 for the current EE) rather than as drawn pixels. Note: the
    trace perturbation does NOT touch ``future_trace_xy`` (planning supervision
    target is the true EE trace), so plan-mode visualizations are unaffected by
    the new perturbation augmentation.

Examples:
    # Default execution mode, full preprocessing (matches what the model sees).
    python pace/openpi/scripts/visualize_trace.py \\
        --config-name trace_vla_lora --step-interval 10 --output vis.png

    # Planning mode — clean base image + supervised trace target + EE/sem dots.
    python pace/openpi/scripts/visualize_trace.py --plan-mode

    # Cleaner visualization: skip the augmax random crop / rotate / color jitter.
    python pace/openpi/scripts/visualize_trace.py --remove-preprocess
"""
from __future__ import annotations

import argparse
from pathlib import Path

import jax
import numpy as np
from PIL import Image

import openpi.models.trace_observation as _trace_obs
import openpi.models.trace_utils as _trace_utils
import openpi.policies.libero_trace_dataset as _libero_trace
import openpi.policies.libero_trace_policy as _libero_trace_policy
import openpi.training.config as _config


_BASE_KEY = "base_0_rgb"
_IMAGE_HW = (224, 224)  # = openpi.models.model.IMAGE_RESOLUTION

# Plan-mode keypoint dot styling (visualization-only; not in training).
_EE_DOT_COLOR = (0, 255, 0)         # current EE — green
_SEM_TARGET_DOT_COLOR = (255, 0, 0) # semantic target — red
_DOT_RADIUS = 4.0


def _build_observation(sample, inputs_xform, resize_xform):
    """Run a per-frame dataset sample through the training-time image transforms
    (LiberoTraceInputs + TraceResizeImages), add a batch dim, then build the
    TraceObservation that compute_loss receives. uint8 -> float [-1, 1] conversion
    happens inside TraceObservation.from_dict.
    """
    # 1. Reorganize into the model's expected schema (image dict, overlay dict, ...).
    data = inputs_xform(dict(sample))

    # 2. Resize all images to model resolution.
    data = resize_xform(data)

    # 3. Add a leading batch dim of 1. Skip strings — they're only used by
    #    TraceTokenizePrompt, which we don't run here, and they would not be
    #    representable as numpy arrays for downstream consumers anyway.
    batched: dict = {}
    for k, v in data.items():
        if isinstance(v, str):
            continue
        if isinstance(v, dict):
            batched[k] = {kk: np.asarray(vv)[None, ...] for kk, vv in v.items()}
        else:
            batched[k] = np.asarray(v)[None, ...]

    # 4. TraceObservation.from_dict (numpy uint8 BHWC -> float32 [-1, 1] BHWC).
    return _trace_obs.TraceObservation.from_dict(batched)


def _to_uint8_hwc(image_field):
    """Convert a (1, H, W, 3) JAX/numpy array in [-1, 1] back to (H, W, 3) uint8."""
    a = np.asarray(image_field)
    if a.ndim == 4:
        a = a[0]
    a = (a + 1.0) * 0.5
    return np.clip(a * 255.0, 0.0, 255.0).astype(np.uint8)


def _format_skill(sample) -> str:
    skill_text = str(sample.get("skill_text", ""))
    skill_name = str(sample.get("skill_name", ""))
    if skill_text and skill_text != skill_name:
        return f"{skill_name} :: {skill_text}"
    return skill_name or "<no skill>"


def _draw_dot(image_hwc_uint8, xy_norm, color_rgb, radius=_DOT_RADIUS):
    """Stamp a filled antialiased disk on the image at normalized [0, 1] coordinates.
    Mutates the image in place. Used only for plan-mode keypoint visualization.
    """
    h, w = image_hwc_uint8.shape[:2]
    cx = float(xy_norm[0]) * (w - 1)
    cy = float(xy_norm[1]) * (h - 1)
    _trace_utils._filled_disk(image_hwc_uint8, cx, cy, radius, color_rgb)


def _render_execution_mode(obs):
    """Default visualization: the overlay image (base + dataset-drawn polyline) that
    the action expert sees in execution-mode training. Falls back to the clean base
    if (very rarely) overlay_images is missing.
    """
    if obs.overlay_images is not None and _BASE_KEY in obs.overlay_images:
        return _to_uint8_hwc(obs.overlay_images[_BASE_KEY])
    return _to_uint8_hwc(obs.images[_BASE_KEY])


def _render_plan_mode(obs, sample, data_config):
    """Plan-mode visualization: the clean base image (planning-mode input) with three
    visualization-only overlays drawn on top:

      - the supervised target trace ``obs.future_trace_xy[0]`` (anchored at t_now,
        rendered using the same draw_polyline_overlay style as training's overlay),
      - a green dot at ``obs.current_ee_xy[0]``,
      - a red dot at ``obs.semantic_target_xy[0]``.

    All keypoints are post-augmax (co-transformed with the image), so they align with
    the cropped/rotated frame. None of these annotations are in the actual training
    input — at training time the EE pixel enters via the inpainting clamp on row 0
    and the semantic target enters via Fourier+MLP AdaRMS conditioning.

    On frames with ``has_trace=False`` the trace target / EE / semantic-target
    fields are zeros; we skip drawing in that case so the saved image is just the
    plain (preprocessed) base.
    """
    vis = _to_uint8_hwc(obs.images[_BASE_KEY])

    if not bool(sample["has_trace"]):
        return vis

    # 1. Supervised target trace polyline (matches training overlay style).
    future_trace = np.asarray(obs.future_trace_xy)[0]  # (N, 2) in [0, 1]
    vis = _trace_utils.draw_polyline_overlay(
        vis,
        future_trace,
        color=tuple(int(c) for c in data_config.overlay_color),
        line_thickness=int(data_config.overlay_thickness),
        endpoint_radius=float(data_config.overlay_endpoint_radius),
    )

    # 2. Current EE keypoint (green dot). By design future_trace[0] == current_ee
    #    after the inpainting-clamp invariant, so this dot lands on the trace's start.
    ee = np.asarray(obs.current_ee_xy)[0]
    _draw_dot(vis, ee, _EE_DOT_COLOR)

    # 3. Semantic target keypoint (red dot).
    sem = np.asarray(obs.semantic_target_xy)[0]
    _draw_dot(vis, sem, _SEM_TARGET_DOT_COLOR)

    return vis


def main():
    parser = argparse.ArgumentParser(description="Visualize trace overlay frames as seen by the TraceVLA model.")
    parser.add_argument("--config-name", default="trace_vla_lora",
                        help="TrainConfig name (default: trace_vla_lora). Use trace_vla for the full-FT config.")
    parser.add_argument("--step-interval", type=int, default=10,
                        help="Number of frames between visualizations within an episode (default 10).")
    parser.add_argument("--remove-preprocess", action="store_true",
                        help="Skip the augmax random crop / rotate / color jitter. Default off — keep them so "
                             "the saved PNG matches what the model actually sees in execution-mode training.")
    parser.add_argument("--plan-mode", action="store_true",
                        help="Visualize the planning-mode input instead of the execution-mode overlay. Saves the "
                             "clean base image (post-preprocess) with the supervised target trace, EE keypoint "
                             "(green dot), and semantic target keypoint (red dot) drawn on top. The dataset-side "
                             "execution-mode overlay path is unaffected.")
    parser.add_argument("--output", default="vis.png", help="Output PNG path (overwritten each step).")
    parser.add_argument("--seed", type=int, default=0, help="JAX RNG seed for the augmax preprocessing.")
    parser.add_argument("--start-ep", type=int, default=0, help="First episode index to visualize.")
    parser.add_argument("--start-step", type=int, default=0, help="First frame within --start-ep.")
    args = parser.parse_args()

    # --- Build dataset + transforms exactly as training does ---
    config = _config.get_config(args.config_name)
    data_config = config.data.create(config.assets_dirs, config.model)
    dataset = _libero_trace.LiberoTraceDataset(data_config, action_horizon=config.model.action_horizon)

    inputs_xform = _libero_trace_policy.LiberoTraceInputs(model_type=config.model.model_type)
    resize_xform = _libero_trace_policy.TraceResizeImages(*_IMAGE_HW)

    rng = jax.random.key(args.seed)

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    n_episodes = len(dataset.episode_starts)
    if n_episodes == 0:
        print("Dataset has no episodes.")
        return

    ep = max(0, args.start_ep)
    step = max(0, args.start_step)

    print(
        f"Loaded {args.config_name}: {n_episodes} episodes, "
        f"{int(dataset.episode_ends[-1])} total frames. "
        f"mode={'plan' if args.plan_mode else 'execution'}, "
        f"step interval={args.step_interval}, "
        f"preprocess={'OFF (--remove-preprocess)' if args.remove_preprocess else 'ON'}."
    )

    while ep < n_episodes:
        ep_start = int(dataset.episode_starts[ep])
        ep_end = int(dataset.episode_ends[ep])
        ep_len = ep_end - ep_start

        # Past the end of this episode → roll over to step 0 of the next one.
        if step >= ep_len:
            ep += 1
            step = 0
            continue

        global_idx = ep_start + step
        sample = dataset[global_idx]

        obs = _build_observation(sample, inputs_xform, resize_xform)

        if not args.remove_preprocess:
            sub_rng, rng = jax.random.split(rng)
            obs = _trace_obs.preprocess_trace_observation(sub_rng, obs, train=True)

        # Branch on mode. The plan-mode renderer reads obs.images (the planning-mode
        # input) and adds visualization-only keypoint/trace overlays; the execution
        # renderer reads obs.overlay_images (the action-expert input). Neither branch
        # mutates the other's data.
        if args.plan_mode:
            vis = _render_plan_mode(obs, sample, data_config)
        else:
            vis = _render_execution_mode(obs)

        Image.fromarray(vis).save(out_path)

        mode_label = "plan" if args.plan_mode else "exec"
        print(
            f"[mode={mode_label} ep={ep:>4d} step={step:>4d} idx={global_idx:>6d}] "
            f"skill={_format_skill(sample)}  "
            f"has_trace={bool(sample['has_trace'])}  "
            f"has_overlay={bool(sample['has_overlay'])}  "
            f"progress={float(sample['progress']):.3f}  "
            f"-> {out_path}",
            flush=True,
        )

        try:
            user = input("[Enter to advance, q to quit] ").strip().lower()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if user in ("q", "quit", "exit"):
            break

        step += args.step_interval


if __name__ == "__main__":
    main()
