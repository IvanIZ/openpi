"""Dataset loader for LIBERO with skill segments + trace target/EE annotations.

Differs from ``LiberoSkillReasonDataset`` in that:

  - It returns no textual reasoning. The model has no language head.
  - It returns extra fields needed by the trace generator and completion head:
      ``semantic_target_xy`` (normalized [0, 1] coords), ``current_ee_xy``,
      ``future_trace_xy`` (resampled, normalized), ``has_trace``, ``progress``,
      ``has_overlay``, ``observation/overlay_image`` (base image with GT trace
      drawn on top).

  - It implements skill-end action zeroing via :func:`pad_skill_horizon_actions`
    from ``libero_reason_dataset``: when the action horizon spills past the
    current skill segment, the tail of the chunk is padded with zero pose
    deltas + the gripper command of the latest in-skill action.

  - It implements anchor-age augmentation for receding-horizon training of
    both the trace generator and the action head. Concretely, for each
    sampled frame ``t*`` inside a skill segment, we sample
    ``a ~ U{0, ..., H_train_max - 1}`` and use ``t_anchor = max(start_step, t* - a)``
    as the anchor. The trace-loss target is the GT trace from
    ``[t_anchor, end_step)`` resampled to ``trace_horizon`` waypoints (arc-length
    by default). The overlay image is rendered using *that* same anchor-aged
    trace, so the action head sees the same "freshness" distribution.

  - Scene dropout (planning + execution) at low rate (default 0.15).

NB: nothing in this module modifies the existing data pipeline; the dataset
class is selected explicitly by ``train_trace_vla.py``.
"""
from __future__ import annotations

import copy
import json
import logging
import os
import pathlib
import re

import numpy as np
import torch
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset

from openpi.models import trace_utils
from openpi.policies.libero_reason_dataset import (
    _resolve_dataset_root,
    pad_skill_horizon_actions,
)


_SKILL_NAME_RE = re.compile(r"^\s*([A-Za-z_]+)\s*(?:\(.*\))?\s*$")


def _strip_skill_parameters(skill: str) -> str:
    match = _SKILL_NAME_RE.match(skill)
    return match.group(1).upper() if match else skill.strip().upper()


def _normalize_segment_end(end_step: int | float) -> int:
    return int(1e9) if end_step == -1 else int(end_step)


def _segment_index_for_step(segments: list[dict], step: int) -> int:
    for i, seg in enumerate(segments):
        if int(seg["start_step"]) <= step < _normalize_segment_end(seg["end_step"]):
            return i
    raise ValueError(f"No segment contains step {step}")


def _safe_load_json(path: str) -> dict:
    with open(path, "r") as f:
        return json.load(f)


def _index_episodes(loaded: dict) -> dict[int, dict]:
    out: dict[int, dict] = {}
    for k, v in loaded.items():
        if isinstance(k, str) and k.isdigit():
            out[int(k)] = v
        elif isinstance(k, int):
            out[k] = v
    return out


class LiberoTraceDataset(LeRobotDataset):
    """LIBERO dataset with skill + trace annotations for the TraceVLA model."""

    def __init__(self, data_config, action_horizon: int):
        # Resolve root the same way LiberoReasonDataset does.
        root = _resolve_dataset_root(
            data_config.repo_id, data_config.skill_annotations_path
        )
        super().__init__(
            data_config.repo_id,
            root=root,
            revision="main",
        )
        print("Using LiberoTraceDataset (TraceVLA)")
        self.data_config = data_config
        self.action_horizon = action_horizon
        self.action_down_sample_steps = int(getattr(data_config, "action_down_sample_steps", 1))
        self.use_wrist_image = bool(getattr(data_config, "use_wrist_image", True))
        self.is_computing_norm_stats = bool(getattr(data_config, "is_computing_norm_stats", False))

        # Trace head config.
        self.trace_horizon = int(data_config.trace_horizon)
        self.trace_resample = str(data_config.trace_resample_method)
        self.h_train_max = int(data_config.h_train_max)
        self.scene_dropout_rate = float(data_config.scene_dropout_rate)

        # New regularization: overlay dropout (replace overlay w/ clean image) and
        # smooth low-frequency trace perturbation applied to the overlay trace only.
        self.overlay_dropout_rate = float(getattr(data_config, "overlay_dropout_rate", 0.10))
        self.trace_perturb_max_sigma = float(getattr(data_config, "trace_perturb_max_sigma", 0.03))
        self.trace_perturb_num_freqs = int(getattr(data_config, "trace_perturb_num_freqs", 3))

        # Overlay rendering config.
        self.overlay_color = tuple(int(c) for c in data_config.overlay_color)
        self.overlay_thickness = int(data_config.overlay_thickness)
        self.overlay_endpoint_radius = float(data_config.overlay_endpoint_radius)

        # State and actions arrays.
        self.low_dim_keys = ["eef_pos", "eef_rot_axis_angle", "gripper_control"]
        self.low_dim_features: dict[str, np.ndarray] = {}
        states = torch.stack(self.hf_dataset["state"]).numpy().astype(np.float32)
        self.low_dim_features["eef_pos"] = states[:, :3]
        self.low_dim_features["eef_rot_axis_angle"] = states[:, 3:6]
        self.low_dim_features["gripper_control"] = states[:, 6:]
        self.actions = torch.stack(self.hf_dataset["actions"]).numpy().astype(np.float32)

        episode_indices = np.array(self.hf_dataset["episode_index"])
        unique_episodes = np.unique(episode_indices)
        episode_masks = episode_indices[:, None] == unique_episodes[None, :]
        episode_ends = np.where(episode_masks)[0][np.cumsum(episode_masks.sum(0)) - 1] + 1
        episode_starts = np.concatenate([[0], episode_ends[:-1]])
        self.episode_starts = episode_starts
        self.episode_ends = episode_ends

        # Load skill annotations and trace annotations.
        skill_path = os.path.expanduser(str(data_config.skill_annotations_path))
        if not os.path.isfile(skill_path):
            raise FileNotFoundError(f"skill_annotations_path not found: {skill_path}")

        self.skills_by_episode = _index_episodes(_safe_load_json(skill_path))

        # Read the (constant) image w/h that the trace coordinates live in.
        first_ep = next(iter(self.skills_by_episode.values()))
        self.trace_image_w = int(first_ep.get("image_width", 256))
        self.trace_image_h = int(first_ep.get("image_height", 256))

        self.indices = list(range(len(self.hf_dataset)))
        self.rdm = np.random.RandomState(int(getattr(data_config, "seed", 42)))

        logging.info(
            "[LiberoTraceDataset] %d frames across %d episodes; trace coord space %dx%d",
            len(self.indices),
            len(unique_episodes),
            self.trace_image_w,
            self.trace_image_h,
        )

    def __len__(self) -> int:
        return len(self.indices)

    # ------------------------------------------------------------------
    # Per-sample fetch
    # ------------------------------------------------------------------
    def __getitem__(self, idx: int) -> dict:
        idx = self.indices[idx]
        item = self.hf_dataset[idx]
        ep_idx = int(item["episode_index"].item())
        start_idx = int(self.episode_starts[ep_idx])
        end_idx = int(self.episode_ends[ep_idx])
        episode_step = idx - start_idx

        # Lookup skill segments + traces for this episode.
        ep_data = self.skills_by_episode.get(ep_idx)
        segments = ep_data["segments"]

        # Default placeholders (in case trace annotations are missing for this frame).
        has_trace = False
        skill_name = ""
        skill_text = ""  # full raw expression including parameters, e.g. "PICKUP_FROM(white mug, table)"
        skill_id = 0
        # `plan_text` is the per-episode plan string from skill_annotations.json
        # (e.g. "1. PICKUP_FROM(...) 2. PLACE_ON(...) ..."). `skill_step_num` is the
        # 1-based position of the current skill within that plan, so the VLM prompt
        # can read "Plan: ...  Current step: 2. PICKUP_FROM(...)".
        plan_text = ""
        skill_step_num = 0
        sem_target_xy_norm = np.zeros((2,), dtype=np.float32)
        cur_ee_xy_norm = np.zeros((2,), dtype=np.float32)
        # Trace target for the trace generator: always built from `t_now -> seg_end`.
        future_trace_xy_norm = np.zeros((self.trace_horizon, 2), dtype=np.float32)
        # Trace used for the action-head overlay: built from `t_anchor -> seg_end`. With
        # `a > 0` this represents the "stale plan" that the action head must learn to
        # consume during receding-horizon execution. When `a == 0` it equals the trace target.
        overlay_trace_xy_norm = np.zeros((self.trace_horizon, 2), dtype=np.float32)
        progress = 0.0
        has_overlay_flag = False

        # Identify the current skill segment.
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
            skill_raw = str(seg.get("skill", "")).strip()
            skill_text = skill_raw  # keep the full parameterized expression for the VLM prompt
            skill_name = _strip_skill_parameters(skill_raw)
            skill_id = trace_utils.skill_to_expert_id(skill_raw)
            # Plan text + 1-based step number for the combined prompt.
            skill_step_num = seg_idx + 1
            if ep_data is not None:
                _plan = ep_data.get("plan", "")
                if isinstance(_plan, str):
                    plan_text = _plan.strip()
                # Find the matching trace_seg by skill_index OR start/end step.
                for ts in ep_data.get("target_traces", []):
                    if int(ts.get("skill_index", -1)) == seg_idx:
                        trace_seg = ts
                        break

        # Construct trace target and EE if available.
        seg_start = int(seg["start_step"]) if seg is not None else 0
        seg_end_raw = int(seg["end_step"]) if seg is not None else episode_step + 1
        seg_end = seg_end_raw if seg_end_raw != -1 else end_idx - start_idx
        seg_end = min(seg_end, end_idx - start_idx)

        # Anchor-age augmentation: a ~ U{0, ..., H_train_max-1}, t_anchor = max(seg_start, t*-a).
        a = int(self.rdm.randint(0, self.h_train_max)) if self.h_train_max > 1 else 0
        t_anchor = max(seg_start, episode_step - a)

        if trace_seg is not None and trace_seg.get("end_effector_trace", {}).get("status") == "OK":
            ee_full = np.asarray(trace_seg["end_effector_trace"]["trace"], dtype=np.float32)  # (T_seg, 2)
            sem = trace_seg.get("semantic_target", {})
            if sem.get("status") == "OK":
                sem_pt_pixel = np.asarray(sem.get("point", [0, 0]), dtype=np.float32)  # [x, y]
                sem_target_xy_norm = np.array(
                    [
                        sem_pt_pixel[0] / max(self.trace_image_w - 1, 1),
                        sem_pt_pixel[1] / max(self.trace_image_h - 1, 1),
                    ],
                    dtype=np.float32,
                )
            # Validate ee_full length matches segment length; if not, fall back to "no trace".
            if ee_full.shape[0] >= (seg_end - seg_start) and seg_end > seg_start:
                # Index relative to segment start.
                t_now_in_seg = episode_step - seg_start
                t_anchor_in_seg = t_anchor - seg_start
                t_now_in_seg = max(0, min(t_now_in_seg, ee_full.shape[0] - 1))
                t_anchor_in_seg = max(0, min(t_anchor_in_seg, ee_full.shape[0] - 1))

                cur_ee_pixel = ee_full[t_now_in_seg]
                cur_ee_xy_norm = np.array(
                    [
                        cur_ee_pixel[0] / max(self.trace_image_w - 1, 1),
                        cur_ee_pixel[1] / max(self.trace_image_h - 1, 1),
                    ],
                    dtype=np.float32,
                )

                inv_w = 1.0 / max(self.trace_image_w - 1, 1)
                inv_h = 1.0 / max(self.trace_image_h - 1, 1)

                def _normalize_polyline(arr: np.ndarray) -> np.ndarray:
                    return np.stack([arr[:, 0] * inv_w, arr[:, 1] * inv_h], axis=1).astype(np.float32)

                # ---------------- Trace generator supervision target ----------------
                # The user's design states the trace generator MUST always be supervised on a
                # trace that starts at the *current* EE position (matching the inpainting
                # clamp at row 0 of the diffused trace). So the supervised target slice is
                # `[t_now_in_seg, seg_end_in_seg)`, regardless of the sampled anchor age `a`.
                seg_residual_target = ee_full[t_now_in_seg:]
                if seg_residual_target.shape[0] < 1:
                    seg_residual_target = ee_full[t_now_in_seg:t_now_in_seg + 1]
                seg_residual_target_norm = _normalize_polyline(seg_residual_target)
                future_trace_xy_norm = trace_utils.resample_trace(
                    seg_residual_target_norm, n_out=self.trace_horizon, method=self.trace_resample
                ).astype(np.float32)

                # ---------------- Overlay trace for the action head ----------------
                # The action head is trained on overlays that may correspond to "old plans"
                # (so it learns to consume traces produced by the trace generator a few steps
                # ago, as in receding-horizon execution). We build this from `t_anchor`,
                # which equals `t_now` when `a == 0` and is older otherwise. This is the only
                # consumer of the anchor age now.
                seg_residual_overlay = ee_full[t_anchor_in_seg:]
                if seg_residual_overlay.shape[0] < 1:
                    seg_residual_overlay = ee_full[t_anchor_in_seg:t_anchor_in_seg + 1]
                seg_residual_overlay_norm = _normalize_polyline(seg_residual_overlay)
                overlay_trace_xy_norm = trace_utils.resample_trace(
                    seg_residual_overlay_norm, n_out=self.trace_horizon, method=self.trace_resample
                ).astype(np.float32)

                # Progress in [0, 1].
                seg_len = max(1, seg_end - seg_start)
                progress = float(np.clip((episode_step - seg_start) / seg_len, 0.0, 1.0))
                has_trace = True

        # ----- Image + overlay rendering -----
        base_image = item["image"]  # tensor or array (H, W, 3) or (3, H, W)
        if isinstance(base_image, torch.Tensor):
            base_image_np = base_image.numpy()
        else:
            base_image_np = np.asarray(base_image)
        base_image_np = _ensure_hwc_uint8(base_image_np)
        
        # Preserve a clean copy of the base image, taken BEFORE any scene-dropout
        base_image_clean_np = base_image_np.copy()

        wrist_image_np = None
        if self.use_wrist_image:
            w = item.get("wrist_image", base_image)
            wrist_image_np = w.numpy() if isinstance(w, torch.Tensor) else np.asarray(w)
            wrist_image_np = _ensure_hwc_uint8(wrist_image_np)

        # Smooth low-frequency perturbation on the *overlay* trace only. The supervised
        # ``future_trace_xy_norm`` is left untouched: the trace generator must still
        # learn the true trace, but the action head must learn to tolerate imperfect
        # predicted traces (the inference-time failure mode). Per-sample sigma is
        # drawn from ``[0, trace_perturb_max_sigma]`` inside the helper.
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

        # Overlay base image with the GT (anchor-aged, possibly perturbed) trace. The
        # overlay uses ``overlay_trace_xy_norm`` (built from t_anchor), which can be
        # stale by up to ``h_train_max`` frames. The trace generator's supervision
        # target ``future_trace_xy_norm`` always starts at the current EE — see the
        # "Trace generator supervision target" block above.
        if has_trace:
            overlay_image_np = trace_utils.draw_polyline_overlay(
                base_image_np,
                overlay_trace_xy_norm,
                color=self.overlay_color,
                line_thickness=self.overlay_thickness,
                endpoint_radius=self.overlay_endpoint_radius,
            )
            has_overlay_flag = True
        else:
            overlay_image_np = base_image_np.copy()

        # Scene dropout (only when training) — implemented dataset-side for clarity.
        # - Planning input image: with probability `scene_dropout_rate`, zero out the base image.
        # - Execution input image: with probability `scene_dropout_rate`, replace with overlay-only image
        #   (zero canvas + overlay), keeping the trace visible.
        # We always also keep `overlay_image` for execution. Here we apply the dropout per-sample.
        if self.scene_dropout_rate > 0.0 and not self.is_computing_norm_stats:
            if self.rdm.rand() < self.scene_dropout_rate:
                base_image_np = np.zeros_like(base_image_np)  # planning-side dropout
            if has_trace and self.rdm.rand() < self.scene_dropout_rate:
                # Replace overlay-on-image with overlay-on-zeros to strip scene cues. Same
                # anchor-aged trace as the regular overlay path above.
                overlay_image_np = trace_utils.draw_polyline_overlay(
                    np.zeros_like(base_image_np),
                    overlay_trace_xy_norm,
                    color=self.overlay_color,
                    line_thickness=self.overlay_thickness,
                    endpoint_radius=self.overlay_endpoint_radius,
                )

        # Overlay dropout (dual of the planning-side scene dropout above): with
        # probability ``overlay_dropout_rate``, replace the overlay image with the
        # *clean* base image (no trace polyline). Forces the action head to act
        # without the trace cue occasionally. We use ``base_image_clean_np`` (the
        # pre-scene-dropout copy) so the action head gets a real RGB scene to
        # reason from rather than a zero canvas. Independent draw from the scene-
        # dropout draws above; this overrides the overlay if both fire.
        if (
            has_trace
            and not self.is_computing_norm_stats
            and self.overlay_dropout_rate > 0.0
            and self.rdm.rand() < self.overlay_dropout_rate
        ):
            overlay_image_np = base_image_clean_np.copy()

        # ----- Build state vector (8-dim, like LiberoSkillReasonDataset) -----
        state_vec = np.concatenate(
            [
                self.low_dim_features["eef_pos"][idx].flatten(),
                self.low_dim_features["eef_rot_axis_angle"][idx].flatten(),
                self.low_dim_features["gripper_control"][idx].flatten(),
            ],
            axis=-1,
        ).astype(np.float32)

        # ----- Action chunk with skill-end zero-padding -----
        # Clip slice to skill end (apply skill-horizon truncation).
        seg_end_idx_global = start_idx + seg_end
        slice_end = min(
            seg_end_idx_global, idx + (self.action_horizon - 1) * self.action_down_sample_steps + 1
        )
        slice_end = max(slice_end, idx + 1)
        actions_chunk = self.actions[idx:slice_end:self.action_down_sample_steps]
        if actions_chunk.shape[0] == 0:
            actions_chunk = self.actions[idx:idx + 1]
        action_is_pad_count = self.action_horizon - actions_chunk.shape[0]
        action_is_pad = torch.tensor(
            [False] * actions_chunk.shape[0] + [True] * action_is_pad_count, dtype=torch.bool
        )
        final_actions = pad_skill_horizon_actions(actions_chunk, self.action_horizon)

        # ----- Build the return dict -----
        return_dict = {
            "observation/image": base_image_np,
            "observation/wrist_image": wrist_image_np if wrist_image_np is not None else base_image_np,
            "observation/overlay_image": overlay_image_np,
            "observation/state": torch.from_numpy(state_vec),
            "actions": torch.from_numpy(final_actions.astype(np.float32)),
            "action_is_pad": action_is_pad,
            "atomic_token": float(skill_id),
            "skill_name": skill_name,
            "skill_text": skill_text,
            # Per-episode plan string + 1-based current-skill index; combined into
            # the VLM prompt downstream by ``TraceTokenizePrompt``.
            "plan_text": plan_text,
            "skill_step_num": int(skill_step_num),
            "semantic_target_xy": sem_target_xy_norm.astype(np.float32),
            "current_ee_xy": cur_ee_xy_norm.astype(np.float32),
            "future_trace_xy": future_trace_xy_norm.astype(np.float32),
            "has_trace": bool(has_trace),
            "has_overlay": bool(has_overlay_flag),
            "progress": float(progress),
            "diffusion_loss_mask": True,  # action loss always applies for chunks we sampled
        }
        # Use the dataset task as the prompt (instruction).
        if "task_index" in item:
            try:
                task_text = self.meta.tasks[int(item["task_index"].item())]
            except Exception:
                task_text = ""
            return_dict["prompt"] = task_text
        for key in ["timestamp", "frame_index", "episode_index", "index", "task_index"]:
            return_dict[key] = item[key]
        return return_dict

def _ensure_hwc_uint8(image: np.ndarray) -> np.ndarray:
    """Coerce image to (H, W, 3) uint8 layout."""
    img = np.asarray(image)
    if img.ndim == 3 and img.shape[0] == 3 and img.shape[-1] != 3:
        img = np.transpose(img, (1, 2, 0))
    if np.issubdtype(img.dtype, np.floating):
        img = np.clip(img * 255.0, 0, 255).astype(np.uint8)
    elif img.dtype != np.uint8:
        img = img.astype(np.uint8)
    return img


class LiberoTrace3DDataset(LiberoTraceDataset):
    def __init__(self, data_config, action_horizon: int):
        print("Using LiberoTrace3DDataset (TraceVLA)")
        super().__init__(data_config, action_horizon)
    
    def __getitem__(self, idx: int) -> dict:
        idx = self.indices[idx]
        item = self.hf_dataset[idx]
        ep_idx = int(item["episode_index"].item())
        start_idx = int(self.episode_starts[ep_idx])
        end_idx = int(self.episode_ends[ep_idx])
        episode_step = idx - start_idx

        # Lookup skill segments + traces for this episode.
        ep_data = self.skills_by_episode.get(ep_idx)
        segments = ep_data["segments"]

        # Trace target for the trace generator: always built from `t_now -> seg_end`.
        # Trace used for the action-head overlay: built from `t_anchor -> seg_end`. With
        # `a > 0` this represents the "stale plan" that the action head must learn to
        # consume during receding-horizon execution. When `a == 0` it equals the trace target.
        overlay_trace_xy_norm = np.zeros((self.trace_horizon, 2), dtype=np.float32)
        progress = 0.0
        has_overlay_flag = False

        # Identify the current skill segment.
        seg_idx = _segment_index_for_step(segments, episode_step)

        seg = segments[seg_idx]
        skill_raw = seg["skill"].strip()
        skill_text = skill_raw  # full raw expression including parameters, e.g. "PICKUP_FROM(white mug, table)"
        skill_name = _strip_skill_parameters(skill_raw)
        skill_id = trace_utils.skill_to_expert_id(skill_raw)

        # `plan_text` is the per-episode plan string from skill_annotations.json
        # (e.g. "1. PICKUP_FROM(...) 2. PLACE_ON(...) ..."). `skill_step_num` is the
        # 1-based position of the current skill within that plan, so the VLM prompt
        # can read "Plan: ...  Current step: 2. PICKUP_FROM(...)".
        plan_text = ep_data["plan"].strip()
        skill_step_num = seg_idx + 1

        trace_seg = None
        # Find the matching trace_seg by skill_index OR start/end step.
        for ts in ep_data.get("target_traces", []):
            # Can this just be an index lookup? Why not? - JCP
            if int(ts.get("skill_index", -1)) == seg_idx:
                trace_seg = ts
                break
        assert trace_seg is not None

        # Construct trace target and EE if available.
        seg_start = int(seg["start_step"]) if seg is not None else 0
        seg_end_raw = int(seg["end_step"]) if seg is not None else episode_step + 1
        seg_end = seg_end_raw if seg_end_raw != -1 else end_idx - start_idx
        seg_end = min(seg_end, end_idx - start_idx)

        # Anchor-age augmentation: a ~ U{0, ..., H_train_max-1}, t_anchor = max(seg_start, t*-a).
        a = int(self.rdm.randint(0, self.h_train_max)) if self.h_train_max > 1 else 0
        t_anchor = max(seg_start, episode_step - a)


        # Semantic target: pixel coordinate + depth.
        sem = trace_seg["semantic_target"]
        sem_pt_pixel = sem["point"]  # [x, y]
        sem_target_xyd_normalized = np.array(
            [
                sem_pt_pixel[0] / self.trace_image_w - 1,
                sem_pt_pixel[1] / self.trace_image_h - 1,
                sem["depth"]
            ],
            dtype=np.float32,
        )

        # Nx3
        ee_full = np.asarray(trace_seg["end_effector_trace"]["source_world_positions"], dtype=np.float32)  # (T_seg, 2)
        assert (ee_full.shape[0] >= (seg_end - seg_start) and seg_end > seg_start)

        # Index relative to segment start.
        t_now_in_seg = episode_step - seg_start
        t_anchor_in_seg = t_anchor - seg_start
        t_now_in_seg = max(0, min(t_now_in_seg, ee_full.shape[0] - 1))
        t_anchor_in_seg = max(0, min(t_anchor_in_seg, ee_full.shape[0] - 1))

        world_to_cam_px = ep_data['ee_projection_camera']['world_to_camera_transform']
        world_to_cam = np.linalg.inv(ep_data['ee_projection_camera']['extrinsic_matrix'])


        inv_w = 1.0 / max(self.trace_image_w - 1, 1)
        inv_h = 1.0 / max(self.trace_image_h - 1, 1)

        def ee_trace_to_xyd(trace, n=self.trace_horizon, method=self.trace_resample):
            """
            Convert 3d world positions into (x, y, depth) camera traces.
            x, y are normalized in image space (0-1). Depth is unnormalized, in meters.
            """
            trace_3d = trace_utils.resample_trace(
                trace, n_out=n, method=method
            ).astype(np.float32)
            trace_homogenous = np.concatenate([trace_3d, np.ones((n, 1))], axis=-1)
            trace_in_cam_frame = (world_to_cam @ trace_homogenous.T).T
            trace_projected = (world_to_cam_px @ trace_homogenous.T).T
            trace_xyd = np.zeros((n, 3))
            # Divide by z coordinate to get back pixels
            trace_xyd[:, :2] = trace_3d[:, :2] / trace_3d[:, 2:3]
            trace_xyd[:, 2] = trace_in_cam_frame[:, 2]
            # Flip the x coordinate by convention...
            return np.stack([
                (self.trace_image_w - 1 - trace_xyd[:, 0]) * inv_w,
                trace_xyd[:, 1] * inv_h,
                trace_xyd[:, 2]
            ], axis=1).astype(np.float32)
        # ---------------- Trace generator supervision target ----------------
        # The user's design states the trace generator MUST always be supervised on a
        # trace that starts at the *current* EE position (matching the inpainting
        # clamp at row 0 of the diffused trace). So the supervised target slice is
        # `[t_now_in_seg, seg_end_in_seg)`, regardless of the sampled anchor age `a`.
        # NOTE: These are robot frame positions
        future_trace_xyd_normalized = ee_trace_to_xyd(ee_full[t_now_in_seg:])

        # ---------------- Overlay trace for the action head ----------------
        # The action head is trained on overlays that may correspond to "old plans"
        # (so it learns to consume traces produced by the trace generator a few steps
        # ago, as in receding-horizon execution). We build this from `t_anchor`,
        # which equals `t_now` when `a == 0` and is older otherwise. This is the only
        # consumer of the anchor age now.
        seg_residual_overlay = ee_full[t_anchor_in_seg:]
        if seg_residual_overlay.shape[0] < 1:
            seg_residual_overlay = ee_full[t_anchor_in_seg:t_anchor_in_seg + 1]
        overlay_trace_xyd_normalized = ee_trace_to_xyd(seg_residual_overlay)
        cur_ee_xyd_normalized = overlay_trace_xyd_normalized[0]

        # Progress in [0, 1].
        seg_len = max(1, seg_end - seg_start)
        progress = float(np.clip((episode_step - seg_start) / seg_len, 0.0, 1.0))

        # ----- Image + overlay rendering -----
        base_image = _ensure_hwc_uint8(item["image"])

        if self.use_wrist_image:
            wrist_image = _ensure_hwc_uint8(item["wrist_image"])
        else:
            # Sus
            wrist_image = base_image

        if not self.is_computing_norm_stats:
            # Smooth low-frequency perturbation on the *overlay* trace only. The supervised
            # ``future_trace_xyd_normalized`` is left untouched: the trace generator must still
            # learn the true trace, but the action head must learn to tolerate imperfect
            # predicted traces (the inference-time failure mode). Per-sample sigma is
            # drawn from ``[0, trace_perturb_max_sigma]`` inside the helper.
            if self.trace_perturb_max_sigma > 0.0:
                overlay_trace_xyd_normalized = trace_utils.smooth_low_freq_perturb(
                    overlay_trace_xyd_normalized,
                    self.rdm,
                    max_sigma=self.trace_perturb_max_sigma,
                    num_freqs=self.trace_perturb_num_freqs,
                ).astype(np.float32)

            diceroll = self.rdm.rand()
            # Scene dropout (only when training) — implemented dataset-side for clarity.
            # - Planning input image: with probability `scene_dropout_rate`, zero out the base image.
            # - Execution input image: with probability `scene_dropout_rate`, replace with overlay-only image
            #   (zero canvas + overlay), keeping the trace visible.
            # We always also keep `overlay_image` for execution. Here we apply the dropout per-sample.
            if diceroll < self.scene_dropout_rate:
                # Use same diceroll for planning and execution side dropout. This may cause issues later with
                # pipelined trace generator -> execution generator, but for now it is fine.
                base_image = np.zeros_like(base_image)
            # Overlay dropout (dual of the planning-side scene dropout above): with
            # probability ``overlay_dropout_rate``, replace the overlay image with the
            # *clean* base image (no trace polyline). Forces the action head to act
            # without the trace cue occasionally. We use ``base_image_clean_np`` (the
            # pre-scene-dropout copy) so the action head gets a real RGB scene to
            # reason from rather than a zero canvas. Independent draw from the scene-
            # dropout draws above; this overrides the overlay if both fire.
            if diceroll >= self.scene_dropout_rate and diceroll < self.scene_dropout_rate + self.overlay_dropout_rate:
                # Mutually exclusive events. The executor will never get both dropouts at once
                overlay_image = base_image
            else:
                # Overlay base image with the GT (anchor-aged, possibly perturbed) trace. The
                # overlay uses ``overlay_trace_xyd_normalized`` (built from t_anchor), which can be
                # stale by up to ``h_train_max`` frames. The trace generator's supervision
                # target ``future_trace_xyd_normalized`` always starts at the current EE — see the
                # "Trace generator supervision target" block above.
                overlay_image = trace_utils.draw_polyline_overlay(
                    base_image,
                    overlay_trace_xyd_normalized[:, :2],
                    color=self.overlay_color,
                    line_thickness=self.overlay_thickness,
                    endpoint_radius=self.overlay_endpoint_radius,
                )

        # ----- Build state vector (8-dim, like LiberoSkillReasonDataset) -----
        state_vec = np.concatenate(
            [
                self.low_dim_features["eef_pos"][idx].flatten(),
                self.low_dim_features["eef_rot_axis_angle"][idx].flatten(),
                self.low_dim_features["gripper_control"][idx].flatten(),
            ],
            axis=-1,
        ).astype(np.float32)

        # ----- Action chunk with skill-end zero-padding -----
        # Clip slice to skill end (apply skill-horizon truncation).
        seg_end_idx_global = start_idx + seg_end
        slice_end = min(
            seg_end_idx_global, idx + (self.action_horizon - 1) * self.action_down_sample_steps + 1
        )
        slice_end = max(slice_end, idx + 1)
        actions_chunk = self.actions[idx:slice_end:self.action_down_sample_steps]
        if actions_chunk.shape[0] == 0:
            actions_chunk = self.actions[idx:idx + 1]
        action_is_pad_count = self.action_horizon - actions_chunk.shape[0]
        action_is_pad = torch.tensor(
            [False] * actions_chunk.shape[0] + [True] * action_is_pad_count, dtype=torch.bool
        )
        final_actions = pad_skill_horizon_actions(actions_chunk, self.action_horizon)

        # ----- Build the return dict -----
        return_dict = {
            "observation/image": base_image,
            "observation/wrist_image": wrist_image,
            "observation/overlay_image": overlay_image,
            "observation/state": torch.from_numpy(state_vec),
            "actions": torch.from_numpy(final_actions.astype(np.float32)),
            "action_is_pad": action_is_pad,
            "atomic_token": float(skill_id),
            "skill_name": skill_name,
            "skill_text": skill_text,
            # Per-episode plan string + 1-based current-skill index; combined into
            # the VLM prompt downstream by ``TraceTokenizePrompt``.
            "plan_text": plan_text,
            "skill_step_num": skill_step_num,
            "semantic_target_xy": sem_target_xyd_normalized,
            "current_ee_xy": cur_ee_xyd_normalized,
            "future_trace_xy": future_trace_xyd_normalized,
            "has_trace": True,
            "has_overlay": True,
            "progress": float(progress),
            "diffusion_loss_mask": True,  # action loss always applies for chunks we sampled
        }
        # Use the dataset task as the prompt (instruction).
        if "task_index" in item:
            try:
                task_text = self.meta.tasks[int(item["task_index"].item())]
            except Exception:
                task_text = ""
            return_dict["prompt"] = task_text
        for key in ["timestamp", "frame_index", "episode_index", "index", "task_index"]:
            return_dict[key] = item[key]
        return return_dict


class LiberoOracleDataset(LeRobotDataset):
    """
    Bare minimum "oracle" for pi05: Pass in the plan, skill, and 3d target as prompt.
    Jank implementation. I'm sorry....
    """
    def __init__(self, data_config, action_horizon: int):
        print("Using LiberoOracleDataset (pi05)")
        # Resolve root the same way LiberoReasonDataset does.

        import openpi
        OPENPI_ROOT = pathlib.Path(openpi.__file__).parent.resolve()
        REPO_ROOT = OPENPI_ROOT / '..' / '..'
        skill_annotations_path=str(REPO_ROOT / "data/libero-100/skill_target_traces_3d.json")
        root = _resolve_dataset_root(
            data_config.repo_id, skill_annotations_path
        )
        super().__init__(
            data_config.repo_id,
            root=root,
            revision="main",
        )
        self.data_config = data_config
        self.action_horizon = action_horizon
        self.action_down_sample_steps = int(getattr(data_config, "action_down_sample_steps", 1))
        self.use_wrist_image = bool(getattr(data_config, "use_wrist_image", True))
        self.is_computing_norm_stats = bool(getattr(data_config, "is_computing_norm_stats", False))

        # State and actions arrays.
        self.low_dim_keys = ["eef_pos", "eef_rot_axis_angle", "gripper_control"]
        self.low_dim_features: dict[str, np.ndarray] = {}
        states = torch.stack(self.hf_dataset["state"]).numpy().astype(np.float32)
        self.low_dim_features["eef_pos"] = states[:, :3]
        self.low_dim_features["eef_rot_axis_angle"] = states[:, 3:6]
        self.low_dim_features["gripper_control"] = states[:, 6:]
        self.actions = torch.stack(self.hf_dataset["actions"]).numpy().astype(np.float32)

        episode_indices = np.array(self.hf_dataset["episode_index"])
        unique_episodes = np.unique(episode_indices)
        episode_masks = episode_indices[:, None] == unique_episodes[None, :]
        episode_ends = np.where(episode_masks)[0][np.cumsum(episode_masks.sum(0)) - 1] + 1
        episode_starts = np.concatenate([[0], episode_ends[:-1]])
        self.episode_starts = episode_starts
        self.episode_ends = episode_ends

        # Load skill annotations and trace annotations.
        skill_path = os.path.expanduser(str(skill_annotations_path))
        if not os.path.isfile(skill_path):
            raise FileNotFoundError(f"skill_annotations_path not found: {skill_path}")

        self.skills_by_episode = _index_episodes(_safe_load_json(skill_path))

        # Read the (constant) image w/h that the trace coordinates live in.
        first_ep = next(iter(self.skills_by_episode.values()))
        self.trace_image_w = int(first_ep.get("image_width", 256))
        self.trace_image_h = int(first_ep.get("image_height", 256))

        self.indices = list(range(len(self.hf_dataset)))
        self.rdm = np.random.RandomState(int(getattr(data_config, "seed", 42)))

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, idx: int) -> dict:
        idx = self.indices[idx]
        item = self.hf_dataset[idx]
        ep_idx = int(item["episode_index"].item())
        start_idx = int(self.episode_starts[ep_idx])
        end_idx = int(self.episode_ends[ep_idx])
        episode_step = idx - start_idx

        # Lookup skill segments + traces for this episode.
        ep_data = self.skills_by_episode.get(ep_idx)
        segments = ep_data["segments"]

        # Identify the current skill segment.
        seg_idx = _segment_index_for_step(segments, episode_step)

        seg = segments[seg_idx]
        skill_raw = seg["skill"].strip()
        skill_text = skill_raw  # full raw expression including parameters, e.g. "PICKUP_FROM(white mug, table)"
        skill_name = _strip_skill_parameters(skill_raw)
        skill_id = trace_utils.skill_to_expert_id(skill_raw)

        # `plan_text` is the per-episode plan string from skill_annotations.json
        # (e.g. "1. PICKUP_FROM(...) 2. PLACE_ON(...) ..."). `skill_step_num` is the
        # 1-based position of the current skill within that plan, so the VLM prompt
        # can read "Plan: ...  Current step: 2. PICKUP_FROM(...)".
        plan_text = ep_data["plan"].strip()
        skill_step_num = seg_idx + 1

        trace_seg = None
        # Find the matching trace_seg by skill_index OR start/end step.
        for ts in ep_data.get("target_traces", []):
            # Can this just be an index lookup? Why not? - JCP
            if int(ts.get("skill_index", -1)) == seg_idx:
                trace_seg = ts
                break
        assert trace_seg is not None

        # Construct trace target and EE if available.
        seg_start = int(seg["start_step"]) if seg is not None else 0
        seg_end_raw = int(seg["end_step"]) if seg is not None else episode_step + 1
        seg_end = seg_end_raw if seg_end_raw != -1 else end_idx - start_idx
        seg_end = min(seg_end, end_idx - start_idx)

        # Semantic target: pixel coordinate + depth.
        sem = trace_seg["semantic_target"]
        sem_pt_pixel = sem["point"]  # [x, y]
        sem_target_xyd_normalized = np.array(
            [
                sem_pt_pixel[0] / self.trace_image_w - 1,
                sem_pt_pixel[1] / self.trace_image_h - 1,
                sem["depth"]
            ],
            dtype=np.float32,
        )

        formatted_prompt = f"Plan: {plan_text}; Skill: {skill_raw}; Target: {sem_pt_pixel[0]:.2f}, {sem_pt_pixel[1]:.2f}, {sem_pt_pixel[2]:.2f}"
        print(formatted_prompt)

        # ----- Build state vector (8-dim, like LiberoSkillReasonDataset) -----
        state_vec = np.concatenate(
            [
                self.low_dim_features["eef_pos"][idx].flatten(),
                self.low_dim_features["eef_rot_axis_angle"][idx].flatten(),
                self.low_dim_features["gripper_control"][idx].flatten(),
            ],
            axis=-1,
        ).astype(np.float32)

        # ----- Action chunk with skill-end zero-padding -----
        # Clip slice to skill end (apply skill-horizon truncation).
        seg_end_idx_global = start_idx + seg_end
        slice_end = min(
            seg_end_idx_global, idx + (self.action_horizon - 1) * self.action_down_sample_steps + 1
        )
        slice_end = max(slice_end, idx + 1)
        actions_chunk = self.actions[idx:slice_end:self.action_down_sample_steps]
        if actions_chunk.shape[0] == 0:
            actions_chunk = self.actions[idx:idx + 1]
        action_is_pad_count = self.action_horizon - actions_chunk.shape[0]
        action_is_pad = torch.tensor(
            [False] * actions_chunk.shape[0] + [True] * action_is_pad_count, dtype=torch.bool
        )
        final_actions = pad_skill_horizon_actions(actions_chunk, self.action_horizon)

        # ----- Build the return dict -----
        return_dict = {
            "observation/image": item['image'],
            "observation/wrist_image": item['wrist_image'],
            "observation/state": torch.from_numpy(state_vec),
            "actions": torch.from_numpy(final_actions.astype(np.float32)),
            "prompt": formatted_prompt
        }
        # Use the dataset task as the prompt (instruction).
        if "task_index" in item:
            try:
                task_text = self.meta.tasks[int(item["task_index"].item())]
            except Exception:
                task_text = ""
            return_dict["prompt"] = task_text
        for key in ["timestamp", "frame_index", "episode_index", "index", "task_index"]:
            return_dict[key] = item[key]
        return return_dict

