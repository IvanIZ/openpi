# GT-VLA: Target-Conditioned Trace Guidance for Generalizable Robotic Manipulation

<p align="center">
Anonymous Authors
</p>

## Abstract

![Overview Image](/images/front_fig_v2.3.png)
Vision-Language-Action (VLA) models have shown strong performance on robotic manipulation, but they often struggle to generalize to unseen tasks, configurations, and long-horizon settings.
A key challenge is that VLAs overfit to training scenes and fail to follow novel language instructions.
Off-the-shelf vision-language models (VLMs) often provide stronger generalization, but cannot directly control robot actions.
To combine the common sense of VLMs with VLA control, we propose **Guided Trace VLA (GT-VLA)**, **a steerable framework that accepts guidance from an external generalist VLM through trace-conditioned action generation**.
GT-VLA uses a generalist model to identify semantic guidance for the current skill, converts this guidance into a 2D visual trace, and conditions its action policy on the resulting trace-rendered observation.
This design separates semantic target acquisition, trace generation, and low-level action execution, allowing high-level guidance to propagate to robot actions.
GT-VLA uses a Mixture-of-Experts architecture with skill-specific trace and action modules for robust execution.
We evaluate GT-VLA on LIBERO and a physical robot platform, showing improved generalization over recent VLA baselines in both settings.

## Installation (Training)

This section is copied from OpenPi (`README_OPENPI.md`).

When cloning this repo, make sure to update submodules:

```bash
git clone --recurse-submodules git@github.com:Physical-Intelligence/openpi.git

# Or if you already cloned the repo:
git submodule update --init --recursive
```

We use [uv](https://docs.astral.sh/uv/) to manage Python dependencies. See the [uv installation instructions](https://docs.astral.sh/uv/getting-started/installation/) to set it up. Once uv is installed, run the following to set up the environment:

```bash
GIT_LFS_SKIP_SMUDGE=1 uv sync
GIT_LFS_SKIP_SMUDGE=1 uv pip install -e .
```

NOTE: `GIT_LFS_SKIP_SMUDGE=1` is needed to pull LeRobot as a dependency.

## Training

Training is done with the script `scripts/train.py`.
At the minimum, one argument should be provided, for the configuration to train.

### Training data setup (LIBERO-100)

Download this huggingface repo: [https://huggingface.co/datasets/nvidia/libero-r-datasets](https://huggingface.co/datasets/nvidia/libero-r-datasets)

Our train scripts expect it to be placed in the `data` folder as a subfolder of this repository: `openpi/data/libero-100/`

```bash
cd data
# Download just the libero-100 dataset
hf download --repo-type dataset --local-dir libero-r-datasets --include libero-100-r/* -- nvidia/libero-r-datasets
# Fix paths to match what is expected by train scripts
mv libero-r-datasets/libero-100-r libero-100

# Move trace annotation file into the folder
unzip skill_target_traces.zip
cp skill_target_traces.json libero-100/
```

### Preset train configs

|Config|Description|
|---|---|
|`pi05_libero_100`|pi05 on libero-100|
|`trace_vla_moe`|GT-VLA (full)|
|`trace_vla_single`|GT-VLA-single ablation|
|`target_vla_actionmoe`|G-VLA ablation|

### Training commands

```bash
export WANDB_MODE=disabled  # Or enabled, if you want WANDB
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.9

# Run once, for the whole dataset. Other times the norm stats can be copied
python scripts/compute_norm_stats.py --config-name trace_vla_moe

# For additional flags, see scripts/smoke_run.sh
python scripts/train.py trace_vla_moe --exp-name=test --overwrite
```

## Data Annotation Format

Metadata fields are stripped out for clarity. Our annotations have some extra information about the annotation procedure itself.
```json
{
  # Each episode is an entry in the top-level dictionary
  "0": {
    "episode_index": 0,
    "task_index": 0,
    "instruction": "put the white mug on the left plate and put the yellow and white mug on the right plate",
    "num_steps": 292,
    "fps": 10,
    "plan": "1. PICKUP_FROM(white mug, table) 2. PLACE_ON(white mug, left plate) 3. PICKUP_FROM(yellow and white mug, table) 4. PLACE_ON(yellow and white mug, right plate)",
    "segments": [   # One entry for each skill
      {
        "start_step": 0,
        "end_step": 115,
        "skill": "PICKUP_FROM(white mug, table)"
      },
      {
        "start_step": 115,
        "end_step": 165,
        "skill": "PLACE_ON(white mug, left plate)"
      },
      {
        "start_step": 165,
        "end_step": 245,
        "skill": "PICKUP_FROM(yellow and white mug, table)"
      },
      {
        "start_step": 245,
        "end_step": 292,
        "skill": "PLACE_ON(yellow and white mug, right plate)"
      }
    ],
    "target_traces": [  # One trace for each skill.
      {
        "skill_index": 0,
        "skill": "PICKUP_FROM(white mug, table)",
        "start_step": 0,
        "end_step": 115,
        "semantic_target": {
          "label": "semantic_target",
          "point": [
            175,
            135
          ]
        },
        "end_effector_trace": {
          "trace": [ <list of 2D points in image space, rounded to pixels> ]
          "raw_trace": [ <list of 2D points in image space> ]
        },
      },
      <more traces>
    ]
  },
  <more episodes>
}
```


## Inference

GT-VLA inference requires an external VLM to generate plans and target points.
We use Gemini 3.1 pro to label the data and run the model; trying other VLMs during
inference may result in degraded performance if the VLM is unable to accurately identify
semantic targets.

```python
import numpy as np

from openpi.models import trace_utils as _trace_utils

TRACE_OVERLAY_COLOR = (0, 255, 255)        # cyan, matches data_config.overlay_color
TRACE_OVERLAY_THICKNESS = 2                  # matches data_config.overlay_thickness
TRACE_OVERLAY_ENDPOINT_RADIUS = 2.5          # matches data_config.overlay_endpoint_radius

def render_overlay_image(base_image: np.ndarray, trace_xy_norm: np.ndarray) -> np.ndarray:
    """Draw the cyan trace polyline onto a copy of the base image — same code path
    as the dataset's overlay rendering, so the action expert sees the same style."""
    return _trace_utils.draw_polyline_overlay(
        base_image,
        np.asarray(trace_xy_norm, dtype=np.float32),
        color=TRACE_OVERLAY_COLOR,
        line_thickness=TRACE_OVERLAY_THICKNESS,
        endpoint_radius=TRACE_OVERLAY_ENDPOINT_RADIUS,
    )

def _make_obs_dict(task_prompt: str, plan_text: str, skill_idx: int, skill_text: str,
        robot_state,                        # Robot state (7-dim (x y z rx ry rz g) for LIBERO)
        overhead_image, wrist_image,        # (H, W, 3) RGB images. Need to flip 180 if using LIBERO environment
        semantic_target_xy, ee_position_xy, # (2,) arrays; 0-1 image space position of target and EE
        with_overlay: bool) -> dict:
    """Build the obs dict the TraceVLA policy expects (mirrors inference_example's
    make_planning_obs / make_execution_obs key set), with fully synthetic, fixed values.

    ``with_overlay=False`` is the planning-mode obs fed to ``sample_trace``;
    ``with_overlay=True`` is the execution-mode obs fed to ``predict_completion`` / ``infer``.
    """
    obs = {
        "observation/image": overhead_image,
        "observation/wrist_image": wrist_image,
        "observation/state": robot_state,
        "atomic_token": float(_trace_utils.skill_to_expert_id(skill_text)),
        "semantic_target_xy": semantic_target_xy,
        "current_ee_xy": ee_position_xy,
        "skill_text": skill_text,
        "skill_name": skill_text.split("(", 1)[0].strip().upper(),
        "plan_text": plan_text,
        "skill_step_num": int(skill_idx + 1),
        "prompt": task_prompt,
        "has_trace": True,
        "has_overlay": False,
        "progress": 0.0,
    }
    if with_overlay:
        obs["observation/overlay_image"] = overhead_image,
        obs["has_overlay"] = True
    return obs


from openpi.policies import policy_config as _policy_config
from openpi.training import config as _config
from openpi.training import checkpoints as _checkpoints

#####################################
# EDIT CODE TO MATCH YOUR RUN CONFIG
#####################################
vla_config = _config.get_config("Define which config you are running here!")
checkpoint_dir = "Define your checkpoint directory here!"
assets_path = "Define your path to the assets folder here!"

data_config = vla_config.data.create(vla_config.assets_dirs, vla_config.model)
norm_stats = _checkpoints.load_norm_stats(assets_path, data_config.asset_id)
policy = _policy_config.create_trained_trace_vla_policy(
    vla_config,
    checkpoint_dir,
    norm_stats=norm_stats,
)

task = "do something with the robot"


# This part is pseudocoded.

# First, get a plan from the VLM, insert your api of choice here
obs = get_env_observation()
# For the plan and skill format, see the data annotation file
plan: list[str] = make_plan_with_vlm(task, obs)
plan_text = " ".join(f'{i+1}. {skill}' for i, skill in enumerate(plan))
for skill_idx, skill in plan:

    # Get semantic target using VLM
    obs = get_env_observation()
    target_xy: np.ndarray = get_semantic_target(plan, skill, obs)

    COMPLETION_THRESHOLD = 0.9
    COMPLETION_CONSECUTIVE = 2
    above_counts = 0
    while above_counts < COMPLETION_CONSECUTIVE:
        # Get trace
        obs = get_env_observation()
        planning_obs = _make_obs_dict(task, plan_text, skill_idx, skill,
            obs.robot_state, obs.overhead_image, obs.wrist_image,
            target_xy, obs.ee_position_xy,
            False
        )
        trace = policy.sample_trace(planning_obs)        # (N, 2) in [0, 1]

        EXEC_PER_PLAN = 2
        for _ in range(EXEC_PER_PLAN):
            # Execute action policy, with overlay image
            obs = get_env_observation()
            overlay_image = render_overlay_image(obs.overhead_image, trace)
            exec_obs = _make_obs_dict(task, plan_text, skill_idx, skill,
                obs.robot_state, overlay_image, obs.wrist_image,
                target_xy, obs.ee_position_xy,
                True
            )
            infer_result = policy.infer(exec_obs)
            actions = np.asarray(infer_result["actions"])      # (action_horizon, K)
            progress = float(np.asarray(infer_result["progress"]))
            if progress > COMPLETION_THRESHOLD:
                above_counts += 1
            else:
                above_counts = 0
            if above_counts >= COMPLETION_CONSECUTIVE:
                break
```
