"""Training script for the Pi0TraceVLA model.

Adapted from ``scripts/train_atomic.py`` but specialized for TraceVLA:

  - Builds a ``LiberoTraceDataset`` and converts batches to ``TraceObservation``.
  - The model returns three losses (action, trace, completion) which we mean and
    log separately, with the total loss going through gradient descent.
  - For weight loading, we replicate pi05_base's action expert (``*_1`` weights)
    to the trace expert (``*_2`` weights), and for the trace MoE FFN we copy the
    pi05_base ``mlp_1`` dense FFN into all K experts (``moe_2/expert_{k}``).
"""
from __future__ import annotations

import dataclasses
import functools
import logging
import platform
from typing import Any

import etils.epath as epath
import flax.nnx as nnx
from flax.training import common_utils
import flax.traverse_util as traverse_util
import jax
import jax.experimental
import jax.numpy as jnp
import numpy as np
import optax
import tqdm_loggable.auto as tqdm
import wandb

import openpi.models.model as _model
import openpi.models.trace_observation as _trace_obs
import openpi.shared.array_typing as at
import openpi.shared.nnx_utils as nnx_utils
import openpi.training.checkpoints as _checkpoints
import openpi.training.config as _config
import openpi.training.data_loader as _data_loader
import openpi.training.optimizer as _optimizer
import openpi.training.sharding as sharding
import openpi.training.utils as training_utils


def init_logging():
    level_mapping = {"DEBUG": "D", "INFO": "I", "WARNING": "W", "ERROR": "E", "CRITICAL": "C"}

    class CustomFormatter(logging.Formatter):
        def format(self, record):
            record.levelname = level_mapping.get(record.levelname, record.levelname)
            return super().format(record)

    formatter = CustomFormatter(
        fmt="%(asctime)s.%(msecs)03d [%(levelname)s] %(message)-80s (%(process)d:%(filename)s:%(lineno)s)",
        datefmt="%H:%M:%S",
    )
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    logger.handlers[0].setFormatter(formatter)


def init_wandb(config: _config.TrainConfig, *, resuming: bool, log_code: bool = False, enabled: bool = True):
    if not enabled:
        wandb.init(mode="disabled")
        return
    ckpt_dir = config.checkpoint_dir
    if not ckpt_dir.exists():
        raise FileNotFoundError(f"Checkpoint directory {ckpt_dir} does not exist.")
    if resuming:
        run_id = (ckpt_dir / "wandb_id.txt").read_text().strip()
        wandb.init(id=run_id, resume="must", project=config.project_name)
    else:
        wandb.init(name=config.exp_name, config=dataclasses.asdict(config), project=config.project_name)
        (ckpt_dir / "wandb_id.txt").write_text(wandb.run.id)
    if log_code:
        wandb.run.log_code(epath.Path(__file__).parent.parent)


# ---------------------------------------------------------------------------
# Weight loading: pi05_base -> Pi0TraceVLA
# ---------------------------------------------------------------------------

def _load_and_filter_weights(loader, params_shape, num_trace_experts: int = 5):
    """Replicate pi05_base's action-expert weights into the trace-expert stream
    and fan out the dense FFN ``mlp_1`` into the K hard MoE experts at ``moe_2/expert_*``.
    """
    loaded_params = loader.load(params_shape)
    flat_loaded = traverse_util.flatten_dict(loaded_params)

    # Mapping: stream-2 attention/norm weights inherit from stream-1 (action expert) at init time.
    # The gemma_300m action expert (and its LoRA variant) has num_heads=8, num_kv_heads=1, so it
    # uses split q_einsum + kv_einsum (not the fused qkv_einsum). We therefore copy q_einsum_1
    # and kv_einsum_1 separately. We also keep the qkv_einsum_2 <- qkv_einsum_1 entry so this
    # remap also works if a future variant uses num_heads == num_kv_heads.
    suffix_pairs = [
        ("q_einsum_2", "q_einsum_1"),
        ("kv_einsum_2", "kv_einsum_1"),
        ("qkv_einsum_2", "qkv_einsum_1"),
        ("attn_vec_einsum_2", "attn_vec_einsum_1"),
        ("pre_attention_norm_2", "pre_attention_norm_1"),
        ("pre_ffw_norm_2", "pre_ffw_norm_1"),
        ("final_norm_2", "final_norm_1"),
    ]
    for trg_suffix, src_suffix in suffix_pairs:
        for k in list(flat_loaded.keys()):
            if src_suffix not in k:
                continue
            # The trace expert is always full-FT (`trace_moe_gemma_300m`, never the LoRA
            # variant), so its nnx state has no `lora_a`/`lora_b` slots. When the action
            # expert is LoRA-fied (`gemma_300m_lora`), `flat_loaded` does contain
            # `..._1/lora_a`/`..._1/lora_b` (back-filled by `_merge_params(missing_regex=".*lora.*")`
            # from the model's reference state). Copying those into `..._2/lora_a` would add
            # keys the trace stream cannot accept, and `replace_by_pure_dict` would refuse.
            if any("lora" in seg for seg in k):
                continue
            trg_key = tuple(seg if seg != src_suffix else trg_suffix for seg in k)
            flat_loaded[trg_key] = flat_loaded[k]

    # Map dense FFN (mlp_1) -> trace MoE experts (moe_2/expert_{k}/w{1,2,3}).
    # mlp_1 stores (gating_einsum: (L, 2, in, hidden)) and (linear: (L, hidden, in)) when scanned.
    # The dense FF uses GELU; the MoE expert uses SiLU. We accept this activation mismatch (same
    # treatment as AtomicVLA) — finetuning bridges it quickly.
    gating_keys = [k for k in flat_loaded if k[-2:] == ("mlp_1", "gating_einsum")]
    linear_keys = [k for k in flat_loaded if k[-2:] == ("mlp_1", "linear")]

    for k in gating_keys:
        gating = flat_loaded[k]  # shape (L, 2, in, hidden) for scanned module.
        # gating[..., 0, ...] -> w1; gating[..., 1, ...] -> w3.
        w1 = gating[..., 0, :, :]  # (L, in, hidden)
        w3 = gating[..., 1, :, :]  # (L, in, hidden)
        prefix = k[:-1]  # path up to "mlp_1"
        for e in range(num_trace_experts):
            flat_loaded[(*prefix[:-1], "moe_2", f"expert_{e}", "w1", "kernel")] = w1
            flat_loaded[(*prefix[:-1], "moe_2", f"expert_{e}", "w3", "kernel")] = w3

    for k in linear_keys:
        linear = flat_loaded[k]  # shape (L, hidden, in)
        prefix = k[:-1]
        for e in range(num_trace_experts):
            flat_loaded[(*prefix[:-1], "moe_2", f"expert_{e}", "w2", "kernel")] = linear

    return traverse_util.unflatten_dict(flat_loaded)


def update_params(orig_params, partial_params):
    for k, v in partial_params.items():
        if isinstance(v, dict):
            if k not in orig_params:
                orig_params[k] = {}
            orig_params[k] = update_params(orig_params.get(k, {}), v)
        else:
            orig_params[k] = v
    return orig_params


# ---------------------------------------------------------------------------
# Train init / step
# ---------------------------------------------------------------------------

@at.typecheck
def init_train_state(config, init_rng, mesh, *, resume: bool):
    tx = _optimizer.create_optimizer(config.optimizer, config.lr_schedule, weight_decay_mask=None)

    def init(rng, partial_params=None):
        rng, model_rng = jax.random.split(rng)
        model = config.model.create(model_rng)

        if partial_params is not None:
            graphdef, state = nnx.split(model)
            state_dict = state.to_pure_dict() if hasattr(state, "to_pure_dict") else dict(state)
            updated_state_dict = update_params(state_dict, partial_params)
            state.replace_by_pure_dict(updated_state_dict)
            model = nnx.merge(graphdef, state)

        params = nnx.state(model)
        params = nnx_utils.state_map(
            params, config.freeze_filter, lambda p: p.replace(p.value.astype(jnp.bfloat16))
        )
        opt_state = tx.init(params.filter(config.trainable_filter))
        return training_utils.TrainState(
            step=0,
            params=params,
            model_def=nnx.graphdef(model),
            tx=tx,
            opt_state=opt_state,
            ema_decay=config.ema_decay,
            ema_params=None if config.ema_decay is None else params,
        )

    train_state_shape = jax.eval_shape(init, init_rng, None)
    state_sharding = sharding.fsdp_sharding(train_state_shape, mesh, log=True)

    if resume:
        return train_state_shape, state_sharding

    # Build fresh model to get its params shape, then apply weight remap.
    model = config.model.create(init_rng)
    params = nnx.state(model)
    params_dict = params.to_pure_dict() if hasattr(params, "to_pure_dict") else dict(params)
    # Resolve K (num trace experts) from the model config.
    num_trace_experts = int(getattr(config.model, "num_trace_experts", 5))
    partial_params = _load_and_filter_weights(config.weight_loader, params_dict, num_trace_experts=num_trace_experts)

    replicated_sharding = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec())
    train_state = jax.jit(
        init,
        donate_argnums=(1,),
        in_shardings=replicated_sharding,
        out_shardings=state_sharding,
    )(init_rng, partial_params)
    return train_state, state_sharding


@at.typecheck
def train_step(
    config: _config.TrainConfig,
    rng: at.KeyArrayLike,
    state: training_utils.TrainState,
    batch: tuple[_trace_obs.TraceObservation, _model.Actions],
) -> tuple[training_utils.TrainState, dict[str, at.Array]]:
    model = nnx.merge(state.model_def, state.params)
    model.train()

    @at.typecheck
    def loss_fn(model, rng, observation, actions):
        per_sample, info = model.compute_loss(rng, observation, actions, train=True)
        return jnp.mean(per_sample), info

    train_rng = jax.random.fold_in(rng, state.step)
    observation, actions = batch
    diff_state = nnx.DiffState(0, config.trainable_filter)
    (loss, train_info), grads = nnx.value_and_grad(loss_fn, argnums=diff_state, has_aux=True)(
        model, train_rng, observation, actions
    )

    params = state.params.filter(config.trainable_filter)
    updates, new_opt_state = state.tx.update(grads, state.opt_state, params)
    new_params = optax.apply_updates(params, updates)
    nnx.update(model, new_params)
    new_params = nnx.state(model)

    new_state = dataclasses.replace(state, step=state.step + 1, params=new_params, opt_state=new_opt_state)
    if state.ema_decay is not None:
        new_state = dataclasses.replace(
            new_state,
            ema_params=jax.tree.map(
                lambda old, new: state.ema_decay * old + (1 - state.ema_decay) * new,
                state.ema_params,
                new_params,
            ),
        )

    kernel_params = nnx.state(
        model,
        nnx.All(
            nnx.Param,
            nnx.Not(nnx_utils.PathRegex(".*/(bias|scale|pos_embedding|input_embedding)")),
            lambda _, x: x.value.ndim > 1,
        ),
    )
    info = {
        "loss": loss,
        "grad_norm": optax.global_norm(grads),
        "param_norm": optax.global_norm(kernel_params),
    }
    info.update(jax.tree.map(jnp.mean, train_info))
    return new_state, info


# ---------------------------------------------------------------------------
# Trace data loader
# ---------------------------------------------------------------------------

def _create_trace_data_loader(config: _config.TrainConfig, *, sharding: jax.sharding.Sharding | None, shuffle: bool, num_batches=None, seed: int = 0):
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
    # Use the standard pipeline's norm-stats handling: norm stats must already be computed
    # via `python pace/openpi/scripts/compute_norm_stats.py --config-name trace_vla` (or
    # `trace_vla_lora`), which writes them under `assets/<repo_id>/`. If they are missing,
    # `transform_dataset` will raise with an actionable message.
    transformed = _data_loader.transform_dataset(dataset, data_config, skip_norm_stats=False)

    local_batch_size = config.batch_size // jax.process_count()
    torch_loader = _data_loader.TorchDataLoader(
        transformed,
        local_batch_size=local_batch_size,
        sharding=sharding,
        shuffle=shuffle,
        num_batches=num_batches,
        num_workers=config.num_workers,
        seed=seed,
    )

    class _Wrapper:
        def __init__(self, loader, dc):
            self._loader = loader
            self._dc = dc

        def data_config(self):
            return self._dc

        def __iter__(self):
            for batch in self._loader:
                yield _trace_obs.TraceObservation.from_dict(batch), batch["actions"]

    return _Wrapper(torch_loader, data_config)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(config: _config.TrainConfig):
    init_logging()
    logging.info(f"Running on: {platform.node()}")

    if config.batch_size % jax.device_count() != 0:
        raise ValueError(
            f"Batch size {config.batch_size} must be divisible by the number of devices {jax.device_count()}."
        )
    jax.config.update("jax_compilation_cache_dir", str(epath.Path("~/.cache/jax").expanduser()))

    rng = jax.random.key(config.seed)
    train_rng, init_rng = jax.random.split(rng)
    mesh = sharding.make_mesh(config.fsdp_devices)
    data_sharding = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec(sharding.DATA_AXIS))
    replicated_sharding = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec())

    checkpoint_manager, resuming = _checkpoints.initialize_checkpoint_dir(
        config.checkpoint_dir,
        keep_period=config.keep_period,
        overwrite=config.overwrite,
        resume=config.resume,
    )
    init_wandb(config, resuming=resuming, enabled=config.wandb_enabled)

    data_loader = _create_trace_data_loader(
        config,
        sharding=data_sharding,
        shuffle=True,
        seed=config.seed,
    )
    data_iter = iter(data_loader)
    batch = next(data_iter)
    logging.info(f"Initialized data loader:\n{training_utils.array_tree_to_info(batch)}")

    train_state, train_state_sharding = init_train_state(config, init_rng, mesh, resume=resuming)
    jax.block_until_ready(train_state)
    logging.info(f"Initialized train state:\n{training_utils.array_tree_to_info(train_state.params)}")

    if resuming:
        train_state = _checkpoints.restore_state(checkpoint_manager, train_state, data_loader)

    ptrain_step = jax.jit(
        functools.partial(train_step, config),
        in_shardings=(replicated_sharding, train_state_sharding, data_sharding),
        out_shardings=(train_state_sharding, replicated_sharding),
        donate_argnums=(1,),
    )

    start_step = int(train_state.step)
    pbar = tqdm.tqdm(
        range(start_step, config.num_train_steps),
        initial=start_step,
        total=config.num_train_steps,
        dynamic_ncols=True,
    )

    infos = []
    for step in pbar:
        with sharding.set_mesh(mesh):
            train_state, info = ptrain_step(train_rng, train_state, batch)
        infos.append(info)
        if step % config.log_interval == 0:
            stacked_infos = common_utils.stack_forest(infos)
            reduced_info = jax.device_get(jax.tree.map(jnp.mean, stacked_infos))
            info_str = ", ".join(f"{k}={v:.4f}" for k, v in reduced_info.items())
            pbar.write(f"Train at step {step}: {info_str}")
            wandb.log(reduced_info, step=step)
            infos = []
        batch = next(data_iter)
        if (step % config.save_interval == 0 and step > start_step) or step == config.num_train_steps - 1:
            _checkpoints.save_state(checkpoint_manager, train_state, data_loader, step)

    logging.info("Waiting for checkpoint manager to finish")
    checkpoint_manager.wait_until_finished()


if __name__ == "__main__":
    main(_config.cli())
