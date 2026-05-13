from __future__ import annotations

import dataclasses
import functools
import logging
import platform
from typing import Any

import etils.epath as epath
import jax
import jax.experimental
import jax.numpy as jnp
import numpy as np
import tqdm_loggable.auto as tqdm

import openpi.models.model as _model
import openpi.training.config as _config

def get_dataset(config: _config.TrainConfig):
    """Build a TraceObservation-aware data loader using the existing TorchDataLoader."""

    if not isinstance(config.data, _config.LeRobotTraceVLADataConfig):
        raise TypeError(
            f"train_trace_vla requires LeRobotTraceVLADataConfig, got {type(config.data).__name__}"
        )
    data_config = config.data.create(config.assets_dirs, config.model)
    logging.info(f"data_config: {data_config}")

    if config.model.trace_dim == 3:
        from openpi.policies.libero_trace_dataset import LiberoTrace3DDataset  # noqa: PLC0415
        dataset = LiberoTrace3DDataset(data_config, action_horizon=config.model.action_horizon)
    else:
        from openpi.policies.libero_trace_dataset import LiberoTraceDataset  # noqa: PLC0415
        dataset = LiberoTraceDataset(data_config, action_horizon=config.model.action_horizon)

    return dataset

def main(config: _config.TrainConfig):
    logging.info(f"Running on: {platform.node()}")

    if config.batch_size % jax.device_count() != 0:
        raise ValueError(
            f"Batch size {config.batch_size} must be divisible by the number of devices {jax.device_count()}."
        )
    jax.config.update("jax_compilation_cache_dir", str(epath.Path("~/.cache/jax").expanduser()))

    rng = jax.random.key(config.seed)

    ds = get_dataset(config)
    print('future_trace_xy:', ds[0]['future_trace_xy'])
    print('semantic_target_xy:', ds[0]['semantic_target_xy'])

if __name__ == "__main__":
    main(_config.cli())
