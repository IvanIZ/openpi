"""Dataset loader for LIBERO with reasoning (CoT) annotations.

Adapted from "Do What You Say" codebase (actalign). Loads reasoning annotations
from cot_simple.json and returns thought/action pairs for training.
"""

import copy
import json
import logging
import os

import numpy as np
import torch
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset



def pad_skill_horizon_actions(actions: np.ndarray, action_horizon: int) -> np.ndarray:
    """Pad a truncated skill horizon with stationary motion and held gripper state.

    LIBERO actions use 6 Cartesian delta dimensions followed by a gripper command.
    When a sampled step is near the end of a skill segment, we want the padded tail to
    keep the robot stationary while preserving the last gripper command.
    """
    if actions.ndim != 2:
        raise ValueError(f"Expected actions to be rank-2, got shape {actions.shape}.")
    if actions.shape[0] == 0:
        raise ValueError("Cannot pad an empty action sequence.")
    if action_horizon < actions.shape[0]:
        raise ValueError(
            f"Action horizon ({action_horizon}) is smaller than the action sequence length ({actions.shape[0]})."
        )

    pad_count = action_horizon - actions.shape[0]
    if pad_count == 0:
        return actions

    padding = np.zeros((pad_count, actions.shape[1]), dtype=actions.dtype)
    padding[:, -1] = actions[-1, -1]
    return np.concatenate([actions, padding], axis=0)


def _resolve_dataset_root(repo_id: str, reasoning_json_path: str | None) -> str | None:
    """Resolve the dataset root path for loading.

    yilin-wu/libero-100 (and similar) are often downloaded via huggingface-cli to the
    HF hub cache, not ~/.cache/lerobot. Use scan_cache_dir to find the actual path.
    Also try the dirname of reasoning_json_path if provided.
    """
    candidates = []
    if reasoning_json_path is not None:
        candidate = os.path.dirname(os.path.expanduser(reasoning_json_path))
        if os.path.isdir(candidate):
            meta_info = os.path.join(candidate, "meta", "info.json")
            if os.path.isfile(meta_info):
                return candidate
            candidates.append(candidate)

    try:
        from huggingface_hub import scan_cache_dir

        info = scan_cache_dir()
        for repo in getattr(info, "repos", []):
            if getattr(repo, "repo_id", None) == repo_id:
                for rev in getattr(repo, "revisions", []):
                    path = getattr(rev, "snapshot_path", None)
                    if path and os.path.isdir(path):
                        meta_info = os.path.join(path, "meta", "info.json")
                        if os.path.isfile(meta_info):
                            return path
                break
    except Exception as e:
        logging.debug("scan_cache_dir failed: %s", e)

    hub_cache = os.environ.get("HUGGINGFACE_HUB_CACHE")
    if hub_cache is None:
        hf_home = os.environ.get("HF_HOME")
        if hf_home is not None:
            hub_cache = os.path.join(hf_home, "hub")
        else:
            hub_cache = os.path.expanduser("~/.cache/huggingface/hub")

    repo_cache_dir = os.path.join(hub_cache, f"datasets--{repo_id.replace('/', '--')}")
    snapshots_dir = os.path.join(repo_cache_dir, "snapshots")
    if os.path.isdir(snapshots_dir):
        for snapshot in sorted(os.listdir(snapshots_dir), reverse=True):
            path = os.path.join(snapshots_dir, snapshot)
            meta_info = os.path.join(path, "meta", "info.json")
            if os.path.isfile(meta_info):
                return path

    return candidates[0] if candidates else None
