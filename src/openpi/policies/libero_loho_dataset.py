"""Dataset loader for the LoHo-Manip executor (trace-conditioned pi05) on LIBERO.

This is a deliberately slim sibling of :class:`LiberoTraceDataset`
(``libero_trace_dataset.py``) for reproducing the LoHo-Manip executor
(arXiv 2604.21924): a plain pi05 policy conditioned on (a) the ground-truth
current-subtask end-effector trace rendered *into* the base image and (b) the
current subtask text as the prompt. See ``~/workspace/mujoco_test/LoHo-Manip_setup.md``
(D7/D8) for the design decisions.

Differences from ``LiberoTraceDataset``:

  - Single base-image output: the trace overlay is rendered directly into
    ``observation/image``. There is no clean/overlay image pair because the
    LoHo executor has no separate planning (trace-generation) pass.
  - No trace-supervision fields (``future_trace_xy``, ``semantic_target_xy``,
    ``current_ee_xy``), no MoE routing token (``atomic_token``), no
    ``progress`` label — the LoHo executor trains with the flow-matching
    action loss only.
  - Prompt is the *current subtask text only* (e.g. "PICKUP_FROM(white mug,
    table)"), matching the LoHo-Manip executor conditioning; the plan text is
    held by the high-level manager, never shown to the executor.
  - NO anchor-age augmentation: the overlay trace is always rendered fresh
    from the current timestep ``t`` (LoHo-Manip specifies none; setup doc D8).
  - Retained augmentations (setup doc D8, explicit deviation from the paper):
    scene dropout (overlay re-rendered on a zero canvas), overlay dropout
    (clean image, no trace), and smooth low-frequency trace perturbation.

Kept identical to ``LiberoTraceDataset``: skill-segment lookup, arc-length
trace resampling to ``trace_horizon`` waypoints, overlay rendering style,
8-dim state construction, and skill-end action chunk clipping/padding via
:func:`pad_skill_horizon_actions`.
"""
from __future__ import annotations

import json
import logging
import os

import numpy as np
import torch
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset

from openpi.models import trace_utils
from openpi.policies.libero_reason_dataset import (
    _resolve_dataset_root,
    pad_skill_horizon_actions,
)
from openpi.policies.libero_trace_dataset import (
    _ensure_hwc_uint8,
    _index_episodes,
    _segment_index_for_step,
)


class LiberoLoHoDataset(LeRobotDataset):
    """LIBERO dataset yielding trace-overlaid pi05 samples for the LoHo executor."""

    def __init__(self, data_config, action_horizon: int):
        root = _resolve_dataset_root(
            data_config.repo_id, data_config.skill_annotations_path
        )
        super().__init__(
            data_config.repo_id,
            root=root,
            revision="main",
        )
        print("Using LiberoLoHoDataset (LoHo-Manip executor)")
        self.data_config = data_config
        self.action_horizon = action_horizon
        self.action_down_sample_steps = int(getattr(data_config, "action_down_sample_steps", 1))
        self.use_wrist_image = bool(getattr(data_config, "use_wrist_image", True))
        self.is_computing_norm_stats = bool(getattr(data_config, "is_computing_norm_stats", False))

        # Trace rendering config (same waypoint interface as trace_vla: 20 pts).
        self.trace_horizon = int(data_config.trace_horizon)
        self.trace_resample = str(data_config.trace_resample_method)

        # Augmentations (setup doc D8): scene dropout / overlay dropout / perturbation.
        self.scene_dropout_rate = float(data_config.scene_dropout_rate)
        self.overlay_dropout_rate = float(data_config.overlay_dropout_rate)
        self.trace_perturb_max_sigma = float(data_config.trace_perturb_max_sigma)
        self.trace_perturb_num_freqs = int(data_config.trace_perturb_num_freqs)

        # Overlay rendering style (identical to trace_vla).
        self.overlay_color = tuple(int(c) for c in data_config.overlay_color)
        self.overlay_thickness = int(data_config.overlay_thickness)
        self.overlay_endpoint_radius = float(data_config.overlay_endpoint_radius)

        # State and actions arrays (identical to LiberoTraceDataset).
        states = torch.stack(self.hf_dataset["state"]).numpy().astype(np.float32)
        self.low_dim_features = {
            "eef_pos": states[:, :3],
            "eef_rot_axis_angle": states[:, 3:6],
            "gripper_control": states[:, 6:],
        }
        self.actions = torch.stack(self.hf_dataset["actions"]).numpy().astype(np.float32)

        episode_indices = np.array(self.hf_dataset["episode_index"])
        unique_episodes = np.unique(episode_indices)
        episode_masks = episode_indices[:, None] == unique_episodes[None, :]
        episode_ends = np.where(episode_masks)[0][np.cumsum(episode_masks.sum(0)) - 1] + 1
        episode_starts = np.concatenate([[0], episode_ends[:-1]])
        self.episode_starts = episode_starts
        self.episode_ends = episode_ends

        # Load skill + trace annotations (same files as trace_vla training).
        skill_path = os.path.expanduser(str(data_config.skill_annotations_path))
        trace_path = os.path.expanduser(str(data_config.trace_annotations_path))
        if not os.path.isfile(skill_path):
            raise FileNotFoundError(f"skill_annotations_path not found: {skill_path}")
        if not os.path.isfile(trace_path):
            raise FileNotFoundError(f"trace_annotations_path not found: {trace_path}")
        with open(skill_path) as f:
            self.skills_by_episode = _index_episodes(json.load(f))
        with open(trace_path) as f:
            self.traces_by_episode = _index_episodes(json.load(f))

        first_ep = next(iter(self.traces_by_episode.values()))
        self.trace_image_w = int(first_ep.get("image_width", 256))
        self.trace_image_h = int(first_ep.get("image_height", 256))

        self.indices = list(range(len(self.hf_dataset)))
        self.rdm = np.random.RandomState(int(getattr(data_config, "seed", 42)))

        logging.info(
            "[LiberoLoHoDataset] %d frames across %d episodes; trace coord space %dx%d",
            len(self.indices),
            len(unique_episodes),
            self.trace_image_w,
            self.trace_image_h,
        )

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, idx: int) -> dict:
        idx = self.indices[idx]
        item = self.hf_dataset[idx]
        ep_idx = int(item["episode_index"].item())
        # Video-backed datasets decode from mp4; image-backed LIBERO is a no-op here
        # (mirrors LiberoTraceDataset.__getitem__).
        if len(self.meta.video_keys) > 0:
            current_ts = item["timestamp"].item()
            query_timestamps = self._get_query_timestamps(current_ts, query_indices=None)
            video_frames = self._query_videos(query_timestamps, ep_idx)
            item = {**video_frames, **item}
        start_idx = int(self.episode_starts[ep_idx])
        end_idx = int(self.episode_ends[ep_idx])
        episode_step = idx - start_idx

        ep_skill = self.skills_by_episode.get(ep_idx)
        ep_trace = self.traces_by_episode.get(ep_idx)
        segments = ep_skill["segments"] if ep_skill is not None else []

        skill_text = ""
        seg_idx = -1
        if segments:
            try:
                seg_idx = _segment_index_for_step(segments, episode_step)
            except ValueError:
                seg_idx = -1

        seg = None
        trace_seg = None
        if seg_idx >= 0:
            seg = segments[seg_idx]
            skill_text = str(seg.get("skill", "")).strip()
            if ep_trace is not None:
                for ts in ep_trace.get("target_traces", []):
                    if int(ts.get("skill_index", -1)) == seg_idx:
                        trace_seg = ts
                        break

        seg_start = int(seg["start_step"]) if seg is not None else 0
        seg_end_raw = int(seg["end_step"]) if seg is not None else episode_step + 1
        seg_end = seg_end_raw if seg_end_raw != -1 else end_idx - start_idx
        seg_end = min(seg_end, end_idx - start_idx)

        # ---- Fresh (current-timestep) subtask trace, resampled to trace_horizon ----
        has_trace = False
        overlay_trace_xy_norm = None
        if trace_seg is not None and trace_seg.get("end_effector_trace", {}).get("status") == "OK":
            ee_full = np.asarray(trace_seg["end_effector_trace"]["trace"], dtype=np.float32)  # (T_seg, 2)
            if ee_full.shape[0] >= (seg_end - seg_start) and seg_end > seg_start:
                t_now_in_seg = max(0, min(episode_step - seg_start, ee_full.shape[0] - 1))
                inv_w = 1.0 / max(self.trace_image_w - 1, 1)
                inv_h = 1.0 / max(self.trace_image_h - 1, 1)
                residual = ee_full[t_now_in_seg:]
                if residual.shape[0] < 1:
                    residual = ee_full[t_now_in_seg:t_now_in_seg + 1]
                residual_norm = np.stack(
                    [residual[:, 0] * inv_w, residual[:, 1] * inv_h], axis=1
                ).astype(np.float32)
                overlay_trace_xy_norm = trace_utils.resample_trace(
                    residual_norm, n_out=self.trace_horizon, method=self.trace_resample
                ).astype(np.float32)
                has_trace = True

        # ---- Images ----
        base_image = item["image"]
        base_image_np = base_image.numpy() if isinstance(base_image, torch.Tensor) else np.asarray(base_image)
        base_image_np = _ensure_hwc_uint8(base_image_np)

        wrist_image_np = None
        if self.use_wrist_image:
            w = item.get("wrist_image", base_image)
            wrist_image_np = w.numpy() if isinstance(w, torch.Tensor) else np.asarray(w)
            wrist_image_np = _ensure_hwc_uint8(wrist_image_np)

        # Trace perturbation (D8): simulate imperfect manager-predicted traces.
        if (
            has_trace
            and not self.is_computing_norm_stats
            and self.trace_perturb_max_sigma > 0.0
        ):
            overlay_trace_xy_norm = trace_utils.smooth_low_freq_perturb(
                overlay_trace_xy_norm,
                self.rdm,
                max_sigma=self.trace_perturb_max_sigma,
                num_freqs=self.trace_perturb_num_freqs,
            ).astype(np.float32)

        # Render the trace into the (single) base image.
        if has_trace:
            model_image_np = trace_utils.draw_polyline_overlay(
                base_image_np,
                overlay_trace_xy_norm,
                color=self.overlay_color,
                line_thickness=self.overlay_thickness,
                endpoint_radius=self.overlay_endpoint_radius,
            )
        else:
            model_image_np = base_image_np.copy()

        if has_trace and not self.is_computing_norm_stats:
            # Scene dropout (D8): trace on a zero canvas — executor must rely on the
            # trace + wrist view alone.
            if self.scene_dropout_rate > 0.0 and self.rdm.rand() < self.scene_dropout_rate:
                model_image_np = trace_utils.draw_polyline_overlay(
                    np.zeros_like(base_image_np),
                    overlay_trace_xy_norm,
                    color=self.overlay_color,
                    line_thickness=self.overlay_thickness,
                    endpoint_radius=self.overlay_endpoint_radius,
                )
            # Overlay dropout (D8): clean image, no trace — executor must stay
            # functional without the trace cue. Overrides scene dropout if both fire
            # (same precedence as LiberoTraceDataset).
            if self.overlay_dropout_rate > 0.0 and self.rdm.rand() < self.overlay_dropout_rate:
                model_image_np = base_image_np.copy()

        # ---- 8-dim state ----
        state_vec = np.concatenate(
            [
                self.low_dim_features["eef_pos"][idx].flatten(),
                self.low_dim_features["eef_rot_axis_angle"][idx].flatten(),
                self.low_dim_features["gripper_control"][idx].flatten(),
            ],
            axis=-1,
        ).astype(np.float32)

        # ---- Action chunk clipped at the subtask boundary, zero-padded ----
        seg_end_idx_global = start_idx + seg_end
        slice_end = min(
            seg_end_idx_global, idx + (self.action_horizon - 1) * self.action_down_sample_steps + 1
        )
        slice_end = max(slice_end, idx + 1)
        actions_chunk = self.actions[idx:slice_end:self.action_down_sample_steps]
        if actions_chunk.shape[0] == 0:
            actions_chunk = self.actions[idx:idx + 1]
        final_actions = pad_skill_horizon_actions(actions_chunk, self.action_horizon)

        # ---- Return: a plain pi05 sample ----
        return_dict = {
            "observation/image": model_image_np,
            "observation/wrist_image": wrist_image_np if wrist_image_np is not None else model_image_np,
            "observation/state": torch.from_numpy(state_vec),
            "actions": torch.from_numpy(final_actions.astype(np.float32)),
        }
        # Prompt = current subtask text only (LoHo-Manip executor conditioning).
        # Fall back to the episode task instruction when no segment matched.
        if skill_text:
            return_dict["prompt"] = skill_text
        elif "task_index" in item:
            try:
                return_dict["prompt"] = self.meta.tasks[int(item["task_index"].item())]
            except Exception:
                return_dict["prompt"] = ""
        for key in ["timestamp", "frame_index", "episode_index", "index", "task_index"]:
            if key in item:
                return_dict[key] = item[key]
        return return_dict
