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

INITIAL_SEGMENT_LENGTH = 10
TRANSITION_WIN_LENGTH = 10
END_EPISODE_WIN = 5


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


def _get_thought(thoughts: list[dict], step: int) -> dict:
    for thought in thoughts:
        end_step = thought['end_step']
        if end_step == -1:
            end_step = int(1e9)
        if thought['start_step'] <= step < end_step:
            return thought
    raise ValueError(f"No thought found for step {step}")


def _get_segment_index(thoughts: list[dict], step: int) -> int:
    for i, thought in enumerate(thoughts):
        end_step = thought["end_step"]
        if end_step == -1:
            end_step = int(1e9)
        if thought["start_step"] <= step < end_step:
            return i
    raise ValueError(f"No segment found for step {step}")


def _format_done_progression(done_skills: list[str]) -> str:
    if not done_skills:
        return "Done: None"
    return f"Done: {', '.join(done_skills)}"


def _build_skill_progression(segments: list[dict], step: int, transition_win_length: int) -> str:
    current_segment_index = _get_segment_index(segments, step)
    if current_segment_index == 0:
        return _format_done_progression([])

    current_segment = segments[current_segment_index]
    if step < current_segment["start_step"] + transition_win_length:
        done_count = current_segment_index - 1
    else:
        done_count = current_segment_index

    done_skills = [segments[i]["skill"] for i in range(max(done_count, 0))]
    return _format_done_progression(done_skills)


def _build_skill_prediction_prefix(plan: str, segments: list[dict], step: int, transition_win_length: int) -> str:
    progression = _build_skill_progression(segments, step, transition_win_length)
    return f"Plan: {plan}; {progression}"


def _get_segment_end_step(segment: dict, default_end_step: int | None = None) -> int:
    end_step = segment["end_step"]
    if end_step == -1:
        if default_end_step is not None:
            return default_end_step
        return int(1e9)
    return end_step


# def _build_skill_completion_thought(
#     segments: list[dict],
#     step: int,
#     transition_win_length: int,
#     end_episode_win: int,
#     prob_predict_prev_skills: float,
#     rdm: np.random.RandomState,
#     episode_num_steps: int | None = None,
# ) -> tuple[str, str]:
#     current_segment_index = _get_segment_index(segments, step)
#     current_segment = segments[current_segment_index]
#     current_skill = current_segment["skill"]
#     current_end_step = _get_segment_end_step(current_segment, episode_num_steps)
#     is_first_segment = current_segment_index == 0
#     is_last_segment = current_segment_index == len(segments) - 1
#     plan_text = " ".join(f"{i + 1}. {segment['skill']}" for i, segment in enumerate(segments))

#     selected_segment_index = current_segment_index
#     selected_skill = current_skill
#     completion_label = "in-progress"

#     if is_first_segment and is_last_segment:
#         if step >= current_end_step - end_episode_win:
#             completion_label = "completed"
            
#     elif is_first_segment:
#         completion_label = "in-progress"

#     elif rdm.rand() < prob_predict_prev_skills:
#         selected_segment_index = rdm.randint(0, current_segment_index)
#         selected_skill = segments[selected_segment_index]["skill"]
#         completion_label = "completed"

#     elif step < current_segment["start_step"] + transition_win_length:
#         selected_segment_index = current_segment_index - 1
#         selected_skill = segments[selected_segment_index]["skill"]
#         completion_label = "completed"
        
#     elif is_last_segment and step >= current_end_step - end_episode_win:
#         completion_label = "completed"

#     prefix = f"Plan: {plan_text}; Current skill: {selected_segment_index + 1}. {selected_skill}"
#     return prefix, completion_label


def _build_skill_completion_thought(
    segments: list[dict],
    step: int,
    transition_win_length: int,
    end_episode_win: int,
    prob_predict_prev_skills: float,
    rdm: np.random.RandomState,
    episode_num_steps: int | None = None,
) -> tuple[str, str]:
    current_segment_index = _get_segment_index(segments, step)
    current_segment = segments[current_segment_index]
    current_skill = current_segment["skill"]
    current_end_step = _get_segment_end_step(current_segment, episode_num_steps)
    is_first_segment = current_segment_index == 0
    is_last_segment = current_segment_index == len(segments) - 1

    if is_first_segment and is_last_segment:
        if step >= current_end_step - end_episode_win:
            return current_skill, "completed"
        return current_skill, "in-progress"

    if is_first_segment:
        return current_skill, "in-progress"

    if rdm.rand() < prob_predict_prev_skills:
        previous_segment_index = rdm.randint(0, current_segment_index)
        return segments[previous_segment_index]["skill"], "completed"

    if step < current_segment["start_step"] + transition_win_length:
        return segments[current_segment_index - 1]["skill"], "completed"

    if is_last_segment and step >= current_end_step - end_episode_win:
        return current_skill, "completed"

    return current_skill, "in-progress"


class LiberoReasonDataset(LeRobotDataset):
    def __init__(self, data_config, action_horizon: int):
        # Resolve root from HF cache or reasoning_json_path. yilin-wu/libero-100 has no
        # version tags on HF, so we must load from local. Use revision="main" as fallback
        # to avoid get_safe_version when pulling meta/.
        root = _resolve_dataset_root(
            data_config.repo_id, data_config.reasoning_json_path
        )
        super().__init__(
            data_config.repo_id,
            root=root,
            revision="main",  # bypass version-tag check (repo has no v2.1 tag)
        )
        self.data_config = data_config
        self.action_horizon = action_horizon
        self.action_down_sample_steps = data_config.action_down_sample_steps
        self.use_reasoning = data_config.use_reasoning
        self.use_wrist_image = data_config.use_wrist_image
        self.use_outdated_reasoning = data_config.use_outdated_reasoning
        self.pred_reasoning_prob = 0.7
        self.is_computing_norm_stats = data_config.is_computing_norm_stats

        self.low_dim_keys = ["eef_pos", "eef_rot_axis_angle", "gripper_control"]
        self.low_dim_features = {}
        states = torch.stack(self.hf_dataset['state']).numpy().astype(np.float32)
        self.low_dim_features['eef_pos'] = states[:, :3]
        self.low_dim_features['eef_rot_axis_angle'] = states[:, 3:6]
        self.low_dim_features['gripper_control'] = states[:, 6:]
        self.actions = torch.stack(self.hf_dataset['actions']).numpy().astype(np.float32)

        episode_indices = np.array(self.hf_dataset['episode_index'])
        unique_episodes = np.unique(episode_indices)
        episode_masks = episode_indices[:, None] == unique_episodes[None, :]
        episode_ends = np.where(episode_masks)[0][np.cumsum(episode_masks.sum(0)) - 1] + 1
        episode_starts = np.concatenate([[0], episode_ends[:-1]])
        self.episode_starts = episode_starts
        self.episode_ends = episode_ends

        reasoning_path = data_config.reasoning_json_path
        if reasoning_path is None and root is not None:
            reasoning_path = os.path.join(root, "cot_simple.json")
        if reasoning_path is not None:
            expanded = os.path.expanduser(reasoning_path)
            if not os.path.isfile(expanded) and root is not None:
                alt = os.path.join(root, "cot_simple.json")
                if os.path.isfile(alt):
                    reasoning_path = alt
                else:
                    reasoning_path = expanded
            else:
                reasoning_path = expanded
        if reasoning_path is not None and os.path.isfile(reasoning_path):
            with open(reasoning_path, 'r') as f:
                _loaded = json.load(f)
                self.reasoning = {}
                for k, v in _loaded.items():
                    if k.isdigit():
                        self.reasoning[int(k)] = v
                    else:
                        self.reasoning[k] = v
        else:
            self.reasoning = None

        if data_config.use_val_dataset:
            if data_config.create_train_val_split:
                self.create_train_val_split()
            self.indices = self.get_indices('train')
        else:
            self.indices = list(range(len(self.hf_dataset)))

        self.rdm = np.random.RandomState(data_config.seed)

    def create_train_val_split(self):
        episode_num = len(self.episode_ends)
        val_num = int(episode_num * self.data_config.val_ratio)
        np.random.seed(self.data_config.seed)
        train_episode_idx = np.random.choice(episode_num, episode_num - val_num, replace=False)
        train_episode_idx = np.sort(train_episode_idx)
        val_episode_idx = np.setdiff1d(np.arange(episode_num), train_episode_idx)
        os.makedirs(self.data_config.norm_stats_dir, exist_ok=True)
        with open(os.path.join(self.data_config.norm_stats_dir, 'train_val_split.json'), 'w') as f:
            json.dump({
                'train_episode_idx': train_episode_idx.tolist(),
                'val_episode_idx': val_episode_idx.tolist()
            }, f)

    def get_indices(self, split):
        split_file = os.path.join(self.data_config.norm_stats_dir, 'train_val_split.json')
        with open(split_file, 'r') as f:
            split_idx = json.load(f)[f'{split}_episode_idx']
        indices = []
        for idx in split_idx:
            start_idx = 0 if idx == 0 else self.episode_ends[idx - 1]
            end_idx = self.episode_ends[idx]
            if self.is_computing_norm_stats:
                if (
                    self.reasoning is not None and
                    'vision_language_episode_idx' in self.reasoning and
                    idx in self.reasoning['vision_language_episode_idx']
                ):
                    continue
            indices += list(range(start_idx, end_idx))
        return indices

    def get_val_dataset(self):
        val_set = copy.copy(self)
        val_set.indices = self.get_indices('val')
        val_set.pred_reasoning_prob = 1.0
        return val_set

    def __len__(self):
        return len(self.indices)

    def get_prob(self, start_step, end_step, now_step, start_prob=0.8, end_prob=0.4):
        assert start_step <= now_step < end_step
        return start_prob - (start_prob - end_prob) * (now_step - start_step) / (end_step - start_step)

    def __getitem__(self, idx: int) -> dict:
        idx = self.indices[idx]
        return_dict = {}
        current_idx_item = self.hf_dataset[idx]
        ep_idx = current_idx_item['episode_index'].item()
        start_idx = self.episode_starts[ep_idx]
        end_idx = self.episode_ends[ep_idx]

        freeze_action = False
        return_dict['act_with_outdated_thought'] = False
        return_dict['think_with_outdated_thought'] = False

        if self.use_reasoning and self.reasoning is not None:
            reasonings = self.reasoning[ep_idx]['segments']

            if ep_idx in self.reasoning.get('vision_language_episode_idx', []):
                if not self.is_computing_norm_stats:
                    reasoning_idx = self.rdm.randint(0, len(reasonings))
                    return_dict['thought'] = [
                        reasonings[reasoning_idx]['content'],
                        reasonings[reasoning_idx]['updated_content']
                    ]
                    return_dict['observation/image'] = self.hf_dataset[idx]['image']
                    if self.use_wrist_image:
                        return_dict['observation/wrist_image'] = self.hf_dataset[idx]['wrist_image']
                    freezing_action = [0., 0., 0., 0., 0., 0., 0.]
                    return_dict['actions'] = torch.tensor(freezing_action, dtype=torch.float32).repeat(self.action_horizon, 1)
                    return_dict['action_is_pad'] = torch.tensor([True] * self.action_horizon)
                    return_dict['state'] = torch.zeros(8, dtype=torch.float32)
                    for key in ['timestamp', 'frame_index', 'episode_index', 'index', 'task_index']:
                        return_dict[key] = current_idx_item[key]
                    return return_dict

            episode_start_interval = self.reasoning[ep_idx].get('episode_start_interval', [0, 1])
            reasoning_dict = _get_thought(reasonings, idx - start_idx)

            # ========== reasoning segments ==========
            if reasoning_dict.get('updated_content') is not None:

                # prob to use previous thought
                reasoning_end_step = reasoning_dict['end_step'] if reasoning_dict['end_step'] != -1 else end_idx - start_idx
                prev_reasoning_prob = self.get_prob(
                    reasoning_dict['start_step'], reasoning_end_step, idx - start_idx
                )

                if self.rdm.rand() < prev_reasoning_prob:
                    # case 1: ====== input previous reasoning, output updated reasoning ======
                    # case 1.1: the first segment
                    if (idx - start_idx) < episode_start_interval[1]:
                        return_dict['thought'] = [reasoning_dict['content'], reasoning_dict['updated_content']]

                    # case 1.2: the non-first segment
                    elif self.rdm.rand() < self.pred_reasoning_prob:
                        return_dict['thought'] = [reasoning_dict['content'], reasoning_dict['updated_content']]

                    # case 2: ====== input previous reasoning, output action ======
                    # the robot should be able to act even the reasoning is outdated
                    else:
                        return_dict['thought'] = [reasoning_dict['content']]
                        return_dict['act_with_outdated_thought'] = True
                else:
                    # case 3: ====== input updated reasoning, output action ======
                    return_dict['thought'] = [reasoning_dict['updated_content_w_instruction']]
                    if reasoning_dict['end_step'] == -1:
                        freeze_action = True
            
            # ========== acting segments with outdated reasoning ==========
            # NOTE: this case does not seem to be used as cot_simple.json for libero-100 (or 10) does not containt outdated_content
            elif 'outdated_content' in reasoning_dict and self.use_outdated_reasoning:
                outdate_prob = self.get_prob(
                    reasoning_dict['start_step'], reasoning_dict['end_step'],
                    idx - start_idx, 0.4, 0.0
                )

                # case 1: ====== input outdated reasoning, output <BEGIN_OF_REASONING> ======
                if self.rdm.rand() < outdate_prob:
                    return_dict['thought'] = [reasoning_dict['outdated_content'], reasoning_dict['content']]
                    # we only supervise the <BEGIN_OF_REASONING> token
                    return_dict['think_with_outdated_thought'] = True

                # case 2: ====== input latest reasoning, output action ======
                else:
                    return_dict['thought'] = [reasoning_dict['content']]
            
            # ========== normal acting segments ==========
            else:
                return_dict['thought'] = [reasoning_dict['content']]
        elif self.reasoning is not None:
            return_dict['prompt'] = self.reasoning[ep_idx]['segments'][0]['content'].strip()

        return_dict['observation/image'] = self.hf_dataset[idx]['image']
        if self.use_wrist_image:
            return_dict['observation/wrist_image'] = self.hf_dataset[idx].get('wrist_image', self.hf_dataset[idx]['image'])

        state_idx = np.array([idx])
        low_dim_dict = {}
        for key in self.low_dim_keys:
            low_dim_dict[key] = self.low_dim_features[key][state_idx]

        slice_end = min(end_idx, idx + (self.action_horizon - 1) * self.action_down_sample_steps + 1)
        actions = self.actions[idx: slice_end: self.action_down_sample_steps]
        action_is_pad = torch.tensor(
            [False] * actions.shape[0] + [True] * (self.action_horizon - actions.shape[0])
        )
        return_dict['action_is_pad'] = action_is_pad
        padding = np.repeat(actions[-1:], self.action_horizon - actions.shape[0], axis=0)
        final_action = np.concatenate([actions, padding], axis=0)

        if freeze_action:
            return_dict['actions'] = torch.from_numpy(
                np.repeat(final_action[:1], self.action_horizon, axis=0).astype(np.float32)
            )
        else:
            return_dict['actions'] = torch.from_numpy(final_action.astype(np.float32))

        return_dict['observation/state'] = torch.from_numpy(
            np.concatenate([
                low_dim_dict['eef_pos'].flatten(),
                low_dim_dict['eef_rot_axis_angle'].flatten(),
                low_dim_dict['gripper_control'].flatten()
            ], axis=-1).astype(np.float32)
        )

        for key in ['timestamp', 'frame_index', 'episode_index', 'index', 'task_index']:
            return_dict[key] = current_idx_item[key]

        return return_dict


# A new class for LiberSkillReasonDataset
class LiberoSkillReasonDataset(LeRobotDataset):
    def __init__(self, data_config, action_horizon: int):
        # Resolve root from HF cache or reasoning_json_path. yilin-wu/libero-100 has no
        # version tags on HF, so we must load from local. Use revision="main" as fallback
        # to avoid get_safe_version when pulling meta/.
        root = _resolve_dataset_root(
            data_config.repo_id, data_config.reasoning_json_path
        )
        super().__init__(
            data_config.repo_id,
            root=root,
            revision="main",  # bypass version-tag check (repo has no v2.1 tag)
        )

        print("Using new skill reasoning dataset (Apr. 14)")
        self.data_config = data_config
        self.action_horizon = action_horizon
        self.action_down_sample_steps = data_config.action_down_sample_steps
        self.use_reasoning = data_config.use_reasoning
        self.use_wrist_image = data_config.use_wrist_image
        self.use_outdated_reasoning = data_config.use_outdated_reasoning
        self.pred_reasoning_prob = 0.7
        self.updated_skill_prob = 0.5
        self.learn_plan_generation_prob = 0.95
        self.learn_reasoning_prob = 0.5
        self.no_image_prob = 0.5
        self.no_image_logged = False
        self.is_computing_norm_stats = data_config.is_computing_norm_stats

        # new params for training experiments
        self.prob_predict_prev_skills = 0.0     # disable this in v3
        print("self.prob_predict_prev_skills---> ", self.prob_predict_prev_skills, self.learn_reasoning_prob)

        self.low_dim_keys = ["eef_pos", "eef_rot_axis_angle", "gripper_control"]
        self.low_dim_features = {}
        states = torch.stack(self.hf_dataset['state']).numpy().astype(np.float32)
        self.low_dim_features['eef_pos'] = states[:, :3]
        self.low_dim_features['eef_rot_axis_angle'] = states[:, 3:6]
        self.low_dim_features['gripper_control'] = states[:, 6:]
        self.actions = torch.stack(self.hf_dataset['actions']).numpy().astype(np.float32)

        episode_indices = np.array(self.hf_dataset['episode_index'])
        unique_episodes = np.unique(episode_indices)
        episode_masks = episode_indices[:, None] == unique_episodes[None, :]
        episode_ends = np.where(episode_masks)[0][np.cumsum(episode_masks.sum(0)) - 1] + 1
        episode_starts = np.concatenate([[0], episode_ends[:-1]])
        self.episode_starts = episode_starts
        self.episode_ends = episode_ends

        reasoning_path = data_config.reasoning_json_path
        if reasoning_path is None and root is not None:
            reasoning_path = os.path.join(root, "cot_simple.json")
        if reasoning_path is not None:
            expanded = os.path.expanduser(reasoning_path)
            if not os.path.isfile(expanded) and root is not None:
                alt = os.path.join(root, "cot_simple.json")
                if os.path.isfile(alt):
                    reasoning_path = alt
                else:
                    reasoning_path = expanded
            else:
                reasoning_path = expanded
        if reasoning_path is not None and os.path.isfile(reasoning_path):
            with open(reasoning_path, 'r') as f:
                _loaded = json.load(f)
                self.reasoning = {}
                for k, v in _loaded.items():
                    if k.isdigit():
                        self.reasoning[int(k)] = v
                    else:
                        self.reasoning[k] = v
        else:
            self.reasoning = None

        if data_config.use_val_dataset:
            if data_config.create_train_val_split:
                self.create_train_val_split()
            self.indices = self.get_indices('train')
        else:
            self.indices = list(range(len(self.hf_dataset)))

        self.rdm = np.random.RandomState(data_config.seed)

    def create_train_val_split(self):
        episode_num = len(self.episode_ends)
        val_num = int(episode_num * self.data_config.val_ratio)
        np.random.seed(self.data_config.seed)
        train_episode_idx = np.random.choice(episode_num, episode_num - val_num, replace=False)
        train_episode_idx = np.sort(train_episode_idx)
        val_episode_idx = np.setdiff1d(np.arange(episode_num), train_episode_idx)
        os.makedirs(self.data_config.norm_stats_dir, exist_ok=True)
        with open(os.path.join(self.data_config.norm_stats_dir, 'train_val_split.json'), 'w') as f:
            json.dump({
                'train_episode_idx': train_episode_idx.tolist(),
                'val_episode_idx': val_episode_idx.tolist()
            }, f)

    def get_indices(self, split):
        split_file = os.path.join(self.data_config.norm_stats_dir, 'train_val_split.json')
        with open(split_file, 'r') as f:
            split_idx = json.load(f)[f'{split}_episode_idx']
        indices = []
        for idx in split_idx:
            start_idx = 0 if idx == 0 else self.episode_ends[idx - 1]
            end_idx = self.episode_ends[idx]
            if self.is_computing_norm_stats:
                if (
                    self.reasoning is not None and
                    'vision_language_episode_idx' in self.reasoning and
                    idx in self.reasoning['vision_language_episode_idx']
                ):
                    continue
            indices += list(range(start_idx, end_idx))
        return indices

    def get_val_dataset(self):
        val_set = copy.copy(self)
        val_set.indices = self.get_indices('val')
        val_set.pred_reasoning_prob = 1.0
        return val_set

    def __len__(self):
        return len(self.indices)

    def get_prob(self, start_step, end_step, now_step, start_prob=0.8, end_prob=0.4):
        assert start_step <= now_step < end_step
        return start_prob - (start_prob - end_prob) * (now_step - start_step) / (end_step - start_step)

    def __getitem__(self, idx: int) -> dict:
        # print("_getitem__ version 2 !!!!!!!!!!!")
        idx = self.indices[idx]
        return_dict = {}
        current_idx_item = self.hf_dataset[idx]
        ep_idx = current_idx_item['episode_index'].item()
        start_idx = self.episode_starts[ep_idx]
        end_idx = self.episode_ends[ep_idx]

        freeze_action = False
        return_dict['act_with_outdated_thought'] = False
        return_dict['think_with_outdated_thought'] = False

        if self.use_reasoning and self.reasoning is not None:
            episode_reasoning = self.reasoning[ep_idx]
            episode_step = idx - start_idx
            segments = episode_reasoning["segments"]
            reasoning_dict = _get_thought(segments, episode_step)

            # ========== reasoning segments ==========
            reasoning_end_step = reasoning_dict['end_step'] if reasoning_dict['end_step'] != -1 else end_idx - start_idx
            end_idx = start_idx + reasoning_end_step

            # Can be None, without target annotations file
            return_dict['target'] = reasoning_dict.get('target')

            if episode_step < INITIAL_SEGMENT_LENGTH:
                # Learn to plan with some probability, otherwise output actions
                if self.rdm.rand() < self.learn_reasoning_prob:
                    # some probability to learn plan generation
                    if self.rdm.rand() < self.learn_plan_generation_prob:
                        return_dict['thought'] = ["Instruction: " + episode_reasoning['instruction'], episode_reasoning['plan']]
                        # No targets during plan generation
                        return_dict['target'] = None
                    # some probability to output skill selection
                    else:
                        return_dict['thought'] = list(
                            _build_skill_completion_thought(
                                segments,
                                episode_step,
                                TRANSITION_WIN_LENGTH,
                                END_EPISODE_WIN,
                                self.prob_predict_prev_skills,
                                self.rdm,
                                episode_reasoning.get("num_steps"),
                            )
                        )

                else:
                    # case 2: ====== input updated reasoning, output action ======
                    return_dict['thought'] = [reasoning_dict['skill']]
            
            
            # ========== normal acting segments ==========
            else:
                # Cut off actions at transitions
                if self.rdm.rand() < self.learn_reasoning_prob:
                    return_dict['thought'] = list(
                        _build_skill_completion_thought(
                            segments,
                            episode_step,
                            TRANSITION_WIN_LENGTH,
                            END_EPISODE_WIN,
                            self.prob_predict_prev_skills,
                            self.rdm,
                            episode_reasoning.get("num_steps"),
                        )
                    )
                else:
                    return_dict['thought'] = [reasoning_dict['skill']]

        elif self.reasoning is not None:
            return_dict['prompt'] = self.reasoning[ep_idx]['segments'][0]['content'].strip()

        return_dict['observation/image'] = self.hf_dataset[idx]['image']
        if self.use_wrist_image:
            return_dict['observation/wrist_image'] = self.hf_dataset[idx].get('wrist_image', self.hf_dataset[idx]['image'])

        if return_dict['target'] is not None:
            if self.rdm.rand() < self.no_image_prob:
                if not self.no_image_logged:
                    logging.debug("Zeroing image observations randomly since targets were found!")
                    self.no_image_logged = True
                return_dict['observation/image'] = torch.zeros_like(return_dict['observation/image'])
                return_dict['observation/wrist_image'] = torch.zeros_like(return_dict['observation/wrist_image'])

        state_idx = np.array([idx])
        low_dim_dict = {}
        for key in self.low_dim_keys:
            low_dim_dict[key] = self.low_dim_features[key][state_idx]

        slice_end = min(end_idx, idx + (self.action_horizon - 1) * self.action_down_sample_steps + 1)
        actions = self.actions[idx: slice_end: self.action_down_sample_steps]
        action_is_pad = torch.tensor(
            [False] * actions.shape[0] + [True] * (self.action_horizon - actions.shape[0])
        )
        return_dict['action_is_pad'] = action_is_pad
        final_action = pad_skill_horizon_actions(actions, self.action_horizon)

        if freeze_action:
            return_dict['actions'] = torch.from_numpy(
                np.repeat(final_action[:1], self.action_horizon, axis=0).astype(np.float32)
            )
        else:
            return_dict['actions'] = torch.from_numpy(final_action.astype(np.float32))

        return_dict['observation/state'] = torch.from_numpy(
            np.concatenate([
                low_dim_dict['eef_pos'].flatten(),
                low_dim_dict['eef_rot_axis_angle'].flatten(),
                low_dim_dict['gripper_control'].flatten()
            ], axis=-1).astype(np.float32)
        )

        for key in ['timestamp', 'frame_index', 'episode_index', 'index', 'task_index']:
            return_dict[key] = current_idx_item[key]

        return return_dict


    # # First full-finetune version backup ----------- predict skill progression [this is a backup such that edits can be made directly to __getitem__]
    # def __getitem__(self, idx: int) -> dict:
    #     print("_getitem__ version 1!!!!!!!!!!!")
    #     idx = self.indices[idx]
    #     return_dict = {}
    #     current_idx_item = self.hf_dataset[idx]
    #     ep_idx = current_idx_item['episode_index'].item()
    #     start_idx = self.episode_starts[ep_idx]
    #     end_idx = self.episode_ends[ep_idx]

    #     freeze_action = False
    #     return_dict['act_with_outdated_thought'] = False
    #     return_dict['think_with_outdated_thought'] = False

    #     if self.use_reasoning and self.reasoning is not None:
    #         episode_reasoning = self.reasoning[ep_idx]
    #         episode_step = idx - start_idx
    #         segments = episode_reasoning["segments"]
    #         reasoning_dict = _get_thought(segments, episode_step)
    #         skill_prediction_prefix = _build_skill_prediction_prefix(
    #             episode_reasoning["plan"],
    #             segments,
    #             episode_step,
    #             TRANSITION_WIN_LENGTH,
    #         )

    #         # ========== reasoning segments ==========
    #         reasoning_end_step = reasoning_dict['end_step'] if reasoning_dict['end_step'] != -1 else end_idx - start_idx
    #         end_idx = start_idx + reasoning_end_step

    #         if episode_step < INITIAL_SEGMENT_LENGTH:
    #             # Learn to plan with some probability, otherwise output actions
    #             if self.rdm.rand() < self.learn_reasoning_prob:
    #                 # some probability to learn plan generation
    #                 if self.rdm.rand() < self.learn_plan_generation_prob:
    #                     return_dict['thought'] = ["Instruction: " + episode_reasoning['instruction'], episode_reasoning['plan']]
    #                 # some probability to output skill selection
    #                 else:
    #                     return_dict['thought'] = [skill_prediction_prefix, reasoning_dict['skill']]

    #             else:
    #                 # case 2: ====== input updated reasoning, output action ======
    #                 return_dict['thought'] = [reasoning_dict['skill']]
            
            
    #         # ========== normal acting segments ==========
    #         else:
    #             # Cut off actions at transitions
    #             if self.rdm.rand() < self.learn_reasoning_prob:
    #                 return_dict['thought'] = [skill_prediction_prefix, reasoning_dict['skill']]
    #             else:
    #                 return_dict['thought'] = [reasoning_dict['skill']]

    #         # Can be None, without target annotations file
    #         return_dict['target'] = reasoning_dict.get('target')

    #     elif self.reasoning is not None:
    #         return_dict['prompt'] = self.reasoning[ep_idx]['segments'][0]['content'].strip()

    #     return_dict['observation/image'] = self.hf_dataset[idx]['image']
    #     if self.use_wrist_image:
    #         return_dict['observation/wrist_image'] = self.hf_dataset[idx].get('wrist_image', self.hf_dataset[idx]['image'])

    #     state_idx = np.array([idx])
    #     low_dim_dict = {}
    #     for key in self.low_dim_keys:
    #         low_dim_dict[key] = self.low_dim_features[key][state_idx]

    #     slice_end = min(end_idx, idx + (self.action_horizon - 1) * self.action_down_sample_steps + 1)
    #     actions = self.actions[idx: slice_end: self.action_down_sample_steps]
    #     action_is_pad = torch.tensor(
    #         [False] * actions.shape[0] + [True] * (self.action_horizon - actions.shape[0])
    #     )
    #     return_dict['action_is_pad'] = action_is_pad
    #     padding = np.repeat(np.zeros_like(actions[-1:]), self.action_horizon - actions.shape[0], axis=0)
    #     final_action = np.concatenate([actions, padding], axis=0)

    #     if freeze_action:
    #         return_dict['actions'] = torch.from_numpy(
    #             np.repeat(final_action[:1], self.action_horizon, axis=0).astype(np.float32)
    #         )
    #     else:
    #         return_dict['actions'] = torch.from_numpy(final_action.astype(np.float32))

    #     return_dict['observation/state'] = torch.from_numpy(
    #         np.concatenate([
    #             low_dim_dict['eef_pos'].flatten(),
    #             low_dim_dict['eef_rot_axis_angle'].flatten(),
    #             low_dim_dict['gripper_control'].flatten()
    #         ], axis=-1).astype(np.float32)
    #     )

    #     for key in ['timestamp', 'frame_index', 'episode_index', 'index', 'task_index']:
    #         return_dict[key] = current_idx_item[key]

    #     return return_dict
