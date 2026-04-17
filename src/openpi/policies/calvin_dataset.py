import copy
import json
import logging
import os

import numpy as np
import torch

from lerobot.common.datasets.lerobot_dataset import LeRobotDataset

def _resolve_dataset_root(repo_id: str, repo_path: str | None) -> str | None:
    """Resolve the dataset root path for loading.

    yilin-wu/libero-100 (and similar) are often downloaded via huggingface-cli to the
    HF hub cache, not ~/.cache/lerobot. Use scan_cache_dir to find the actual path.
    Also try the repo_path if provided.
    """
    candidates = []
    if repo_path is not None:
        candidate = os.path.expanduser(repo_path)
        if os.path.isdir(candidate):
            meta_info = os.path.join(candidate, "meta", "info.json")
            if os.path.isfile(meta_info):
                logging.info(f"Using local data at: {candidate}")
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
                            logging.info(f"Using HF cached data at: {path}")
                            return path
                break
    except Exception as e:
        logging.debug("scan_cache_dir failed: %s", e)

class CalvinDataset(LeRobotDataset):
    def __init__(self, data_config, action_horizon: int):
        # Resolve root from HF cache or reasoning_json_path. yilin-wu/libero-100 has no
        # version tags on HF, so we must load from local. Use revision="main" as fallback
        # to avoid get_safe_version when pulling meta/.
        root = _resolve_dataset_root(
            data_config.repo_id, data_config.repo_path
        )
        super().__init__(
            data_config.repo_id,
            root=root,
            revision="main",
        )
        self.data_config = data_config
        self.action_horizon = action_horizon
        self.action_down_sample_steps = data_config.action_down_sample_steps
        self.use_wrist_image = data_config.use_wrist_image

        self.low_dim_keys = ["eef_pos", "eef_rot_axis_angle", "gripper_control"]
        self.low_dim_features = {}
        states = torch.stack(self.hf_dataset['observation.state']).numpy().astype(np.float32)
        self.low_dim_features['eef_pos'] = states[:, :3]
        self.low_dim_features['eef_rot_axis_angle'] = states[:, 3:6]
        self.low_dim_features['gripper_control'] = states[:, 6:]
        self.actions = torch.stack(self.hf_dataset['action']).numpy().astype(np.float32)

        episode_indices = np.array(self.hf_dataset['episode_index'])
        task_indices = np.array(self.hf_dataset['task_index'])
        unique_episodes = np.unique(episode_indices)
        episode_masks = episode_indices[:, None] == unique_episodes[None, :]
        episode_ends = np.where(episode_masks)[0][np.cumsum(episode_masks.sum(0)) - 1] + 1
        episode_starts = np.concatenate([[0], episode_ends[:-1]])
        self.episode_starts = episode_starts
        self.episode_ends = episode_ends

        self.rdm = np.random.RandomState(data_config.seed)

        episode_info_fname = os.path.join(data_config.repo_path, "meta", "episodes.jsonl")
        self.episode_info = {}
        with open(episode_info_fname, 'r') as episode_file:
            for line in episode_file.readlines():
                data = json.loads(line)
                if len(data['tasks']) != 1:
                    print(f"Episode {data['episode_idx']} has a weird number of tasks: {len(data['tasks'])} != 1")
                task_nl = data['tasks'][0]
                task_split = task_nl.split(':', 1)
                if len(task_split) != 2:
                    print(f"Episode {data['episode_idx']} task split failed: `{task_nl}`")
                else:
                    task_nl = task_split[1].strip()
                data['task_nl'] = task_nl
                self.episode_info[data['episode_index']] = data

        # NOTE: in case you want to make train/test splits.
        self.indices = list(range(len(self.hf_dataset)))

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, raw_idx: int) -> dict:
        idx = self.indices[raw_idx]
        return_dict = {}
        current_item = self.hf_dataset[idx]
        ep_idx = current_item['episode_index'].item()
        start_idx = self.episode_starts[ep_idx]
        end_idx = self.episode_ends[ep_idx]
        task_nl = self.episode_info[ep_idx]['task_nl']
        
        return_dict['observation/image'] = current_item['observation.images.top']
        if self.use_wrist_image:
            return_dict['observation/wrist_image'] = current_item['observation.images.wrist']

        
        slice_end = min(end_idx, idx + (self.action_horizon - 1) * self.action_down_sample_steps + 1)
        actions = self.actions[idx: slice_end: self.action_down_sample_steps]
        action_is_pad = torch.tensor(
            [False] * actions.shape[0] + [True] * (self.action_horizon - actions.shape[0])
        )
        return_dict['action_is_pad'] = action_is_pad
        padding = np.repeat(actions[-1:], self.action_horizon - actions.shape[0], axis=0)
        final_action = np.concatenate([actions, padding], axis=0)
        return_dict['actions'] = torch.from_numpy(final_action.astype(np.float32))

        return_dict['thought'] = [task_nl]

        return_dict['observation/state'] = torch.from_numpy(
            np.concatenate([
                self.low_dim_features['eef_pos'][idx].flatten(),
                self.low_dim_features['eef_rot_axis_angle'][idx].flatten(),
                self.low_dim_features['gripper_control'][idx].flatten()
            ], axis=-1).astype(np.float32)
        )
        return return_dict
