from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
import numpy as np
import json
import torch
from openpi.training.config import AtomicDataConfig


def _get_thought(thoughts: list[dict], step: int, episode_id) -> dict[str, str | None]:
    for thought in thoughts:
        end_step = thought['end_frame']
        if end_step == -1:
            end_step = 1e9
        if thought['start_frame'] <= step <= end_step:
            return thought

    raise ValueError(f"No thought found for step {step} and {episode_id}")


class Atomic_Dataset(LeRobotDataset):
    def __init__(self, data_config: AtomicDataConfig, action_horizon: int, delta_timestamps):
        super().__init__(data_config.repo_id, delta_timestamps=delta_timestamps)
        self.data_config = data_config

        self.use_reasoning = data_config.use_reasoning
        self.rdm = np.random.RandomState(data_config.seed)

        self.actions = torch.stack(self.hf_dataset['actions']).numpy().astype(np.float32)

        self.reasoning_json_path = data_config.reasoning_json_path
        if self.reasoning_json_path is not None:
            with open(self.reasoning_json_path, 'r') as f:
                _loaded = json.load(f)
                self.reasoning = {}
                for k, v in _loaded.items():
                    if k.isdigit():
                        self.reasoning[int(k)] = v
                    else:
                        self.reasoning[k] = v

        self.indices = list(range(len(self.hf_dataset)))

    def __len__(self):
        return len(self.indices)

    def get_prob(
            self,
            start_step: int,
            end_step: int,
            now_step: int,
            start_prob: float = 0.2,
            end_prob: float = 0.95,
        ) -> float:
        """Linearly interpolate the probability from start_prob to end_prob."""
        assert start_step <= now_step <= end_step
        return start_prob + (end_prob - start_prob) * (now_step - start_step) / (end_step - start_step)

    def __getitem__(self, idx: int) -> dict:
        """
        return_dict['thought'] is a list of strings
            - if the length is 1, it only contains the latest reasoning content
            - if the length is 2, it contains the latest reasoning content and the updated reasoning content
        """
        item = self.hf_dataset[idx]
        ep_idx = item["episode_index"].item()

        query_indices = None
        if self.delta_indices is not None:
            query_indices, padding = self._get_query_indices(idx, ep_idx)
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
            image_keys = self.meta.camera_keys
            for cam in image_keys:
                item[cam] = self.image_transforms(item[cam])

        task_idx = item["task_index"].item()
        item["task"] = self.meta.tasks[task_idx]

        idx = self.indices[idx]
        return_dict = {}
        current_idx_item = item

        if self.use_reasoning:
            episode_id = current_idx_item['episode_index'].item()
            reasonings = self.reasoning[episode_id]['segments']

            reasoning_dict = _get_thought(reasonings, current_idx_item['frame_index'], episode_id)

            if current_idx_item['frame_index'] <= 10:
                thought_ = "Lets begin, the atomic skill is " + reasoning_dict['primary_action_verb']
                return_dict['thought'] = [reasoning_dict['action_name'], thought_]
            elif (reasoning_dict['end_frame'] - current_idx_item['frame_index']) <= 5 and 'update_chain_of_thought' in reasoning_dict:
                thought_ = "Planning new skill, the atomic skill is " + reasoning_dict['update_chain_of_thought'].split()[-1]
                return_dict['thought'] = [reasoning_dict['action_name'], thought_]
            elif (current_idx_item['frame_index'] - reasoning_dict['start_frame']) <= 5 and reasoning_dict['start_frame'] != 0:
                thought_ = "Planning new skill, the atomic skill is " + reasoning_dict['primary_action_verb']
                return_dict['thought'] = [reasoning_dict['action_name'], thought_]
            else:
                return_dict['thought'] = [reasoning_dict['action_name'], reasoning_dict['primary_action_verb']]

        copy_key = ['timestamp', 'frame_index', 'episode_index', 'index', 'task_index', 'image', 'wrist_image', 'state', 'actions', 'actions_is_pad', 'task']
        for key in copy_key:
            return_dict[key] = current_idx_item[key]
        return return_dict
