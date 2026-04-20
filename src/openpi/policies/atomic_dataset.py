import json
import logging
import os
import re

import numpy as np
import torch
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset

from openpi.training.config import AtomicDataConfig

INITIAL_SEGMENT_LENGTH = 10
TRANSITION_WINDOW_LENGTH = 5
_SKILL_NAME_RE = re.compile(r"^\s*([A-Za-z_]+)\s*(?:\(.*\))?\s*$")


def _iter_snapshot_candidates(path: str | os.PathLike[str]) -> list[str]:
    expanded = os.path.expanduser(os.fspath(path))
    if not os.path.isdir(expanded):
        return []

    candidates = [expanded]
    snapshots_dir = os.path.join(expanded, "snapshots")
    if os.path.isdir(snapshots_dir):
        for snapshot in sorted(os.listdir(snapshots_dir), reverse=True):
            candidates.append(os.path.join(snapshots_dir, snapshot))
    return candidates


def _is_dataset_root(path: str) -> bool:
    return os.path.isfile(os.path.join(path, "meta", "info.json"))


def resolve_dataset_root(repo_id: str, repo_path: str | None) -> str | None:
    """Resolve a LeRobot dataset root from an explicit path or HF cache."""
    if repo_path is not None:
        for candidate in _iter_snapshot_candidates(repo_path):
            if _is_dataset_root(candidate):
                logging.info("Using Atomic dataset at: %s", candidate)
                return candidate

    try:
        from huggingface_hub import scan_cache_dir

        info = scan_cache_dir()
        for repo in getattr(info, "repos", []):
            if getattr(repo, "repo_id", None) == repo_id:
                for rev in getattr(repo, "revisions", []):
                    snapshot_path = getattr(rev, "snapshot_path", None)
                    if snapshot_path and _is_dataset_root(snapshot_path):
                        logging.info("Using Atomic dataset from HF cache: %s", snapshot_path)
                        return snapshot_path
                break
    except Exception as exc:
        logging.debug("Atomic dataset cache scan failed: %s", exc)

    hub_cache = os.environ.get("HUGGINGFACE_HUB_CACHE")
    if hub_cache is None:
        hf_home = os.environ.get("HF_HOME")
        hub_cache = os.path.join(hf_home, "hub") if hf_home is not None else os.path.expanduser("~/.cache/huggingface/hub")

    repo_cache_dir = os.path.join(hub_cache, f"datasets--{repo_id.replace('/', '--')}")
    for candidate in _iter_snapshot_candidates(repo_cache_dir):
        if _is_dataset_root(candidate):
            logging.info("Using Atomic dataset from resolved cache path: %s", candidate)
            return candidate

    return None


def _normalize_segment_end(end_step: int | float) -> int:
    return int(1e9) if end_step == -1 else int(end_step)


def _segment_contains_step(segment: dict, step: int) -> bool:
    return int(segment["start_step"]) <= step < _normalize_segment_end(segment["end_step"])


def _get_thought(thoughts: list[dict], step: int, episode_id: int) -> dict[str, str | int | None]:
    for thought in thoughts:
        if _segment_contains_step(thought, step):
            return thought
    raise ValueError(f"No thought found for step {step} and episode {episode_id}")


def _get_segment_index(thoughts: list[dict], step: int, episode_id: int) -> int:
    for index, thought in enumerate(thoughts):
        if _segment_contains_step(thought, step):
            return index
    raise ValueError(f"No segment found for step {step} and episode {episode_id}")


def _strip_skill_parameters(skill: str) -> str:
    match = _SKILL_NAME_RE.match(skill)
    if match:
        return match.group(1).upper()
    return skill.strip().upper()


def _canonicalize_skill_name(skill: str) -> str:
    skill = skill.strip()
    if "(" in skill:
        skill = skill.split("(", 1)[0]
    match = _SKILL_NAME_RE.match(skill)
    if match is None:
        raise ValueError(f"Unable to parse skill name from {skill!r}.")
    return match.group(1)


def _to_python_str(value) -> str:
    if isinstance(value, str):
        return value
    if hasattr(value, "item"):
        try:
            return value.item()
        except Exception:
            pass
    return str(value)


def _segment_skill_name(segment: dict) -> str:
    if "skill" in segment:
        return _strip_skill_parameters(segment["skill"])
    return _strip_skill_parameters(segment["primary_action_verb"])


class Atomic_Dataset(LeRobotDataset):
    def __init__(self, data_config: AtomicDataConfig, action_horizon: int, delta_timestamps):
        root = resolve_dataset_root(data_config.repo_id, data_config.repo_path)
        super().__init__(
            data_config.repo_id,
            root=root,
            revision="main",
            delta_timestamps=delta_timestamps,
        )
        self.data_config = data_config

        self.use_reasoning = data_config.use_reasoning
        self.rdm = np.random.RandomState(data_config.seed)

        self.actions = torch.stack(self.hf_dataset["actions"]).numpy().astype(np.float32)
        episode_indices = np.array(self.hf_dataset["episode_index"])
        episode_starts = np.flatnonzero(np.r_[True, episode_indices[1:] != episode_indices[:-1]])
        episode_ends = np.r_[episode_starts[1:], len(episode_indices)]
        self.episode_starts = episode_starts
        self.episode_ends = episode_ends

        self.reasoning_json_path = data_config.reasoning_json_path
        self.reasoning = None
        if self.reasoning_json_path is not None:
            with open(self.reasoning_json_path, "r") as f:
                loaded = json.load(f)
            self.reasoning = {}
            for key, value in loaded.items():
                if key.isdigit():
                    self.reasoning[int(key)] = value
                else:
                    self.reasoning[key] = value

        self.indices = list(range(len(self.hf_dataset)))

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx: int) -> dict:
        """
        return_dict["thought"] is a length-2 list:
        - [instruction, "<skill>"] for action supervision
        - [instruction, "Planning ... <skill>"] for text reasoning supervision
        """
        dataset_idx = self.indices[idx]
        item = self.hf_dataset[dataset_idx]
        ep_idx = item["episode_index"].item()

        query_indices = None
        if self.delta_indices is not None:
            query_indices, padding = self._get_query_indices(dataset_idx, ep_idx)
            query_result = self._query_hf_dataset(query_indices)
            item = {**item, **padding}
            for key, val in query_result.items():
                item[key] = val

        if len(self.meta.video_keys) > 0:
            current_ts = item["timestamp"].item()
            query_timestamps = self._get_query_timestamps(current_ts, query_indices)
            video_frames = self._query_videos(query_timestamps, ep_idx)
            item = {**video_frames, **item}

        if self.image_transforms is not None:
            for cam in self.meta.camera_keys:
                item[cam] = self.image_transforms(item[cam])

        task_idx = item["task_index"].item()
        item["task"] = self.meta.tasks[task_idx]

        return_dict = {}
        current_idx_item = item

        if self.use_reasoning and self.reasoning is not None:
            episode_id = current_idx_item["episode_index"].item()
            episode_reasoning = self.reasoning[episode_id]
            reasonings = episode_reasoning["segments"]
            episode_step = int(dataset_idx - self.episode_starts[episode_id])

            reasoning_dict = _get_thought(reasonings, episode_step, episode_id)
            segment_index = _get_segment_index(reasonings, episode_step, episode_id)

            instruction = episode_reasoning.get(
                "instruction",
                reasoning_dict.get("action_name", _to_python_str(current_idx_item["task"])),
            )
            current_skill = _segment_skill_name(reasoning_dict)
            next_skill = None
            if segment_index + 1 < len(reasonings):
                next_skill = _segment_skill_name(reasonings[segment_index + 1])

            if episode_step <= INITIAL_SEGMENT_LENGTH:
                thought = f"Lets begin, the atomic skill is {current_skill}"

            elif (
                next_skill is not None
                and _normalize_segment_end(reasoning_dict["end_step"]) - episode_step <= TRANSITION_WINDOW_LENGTH
            ):
                thought = f"Planning new skill, the atomic skill is {next_skill}"

            elif (
                episode_step - int(reasoning_dict["start_step"]) <= TRANSITION_WINDOW_LENGTH
                and int(reasoning_dict["start_step"]) != 0
            ):
                thought = f"Planning new skill, the atomic skill is {current_skill}"

            else:
                thought = current_skill

            return_dict["prompt"] = instruction
            return_dict["thought"] = [instruction, thought]
        else:
            return_dict["prompt"] = _to_python_str(current_idx_item["task"])

        copy_key = [
            "timestamp",
            "frame_index",
            "episode_index",
            "index",
            "task_index",
            "image",
            "wrist_image",
            "state",
            "actions",
            "actions_is_pad",
            "task",
        ]
        for key in copy_key:
            return_dict[key] = current_idx_item[key]
        return return_dict
