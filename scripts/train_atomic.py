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
import openpi.shared.array_typing as at
import openpi.shared.nnx_utils as nnx_utils
import openpi.training.checkpoints as _checkpoints
import openpi.training.config as _config
import openpi.training.data_loader as _data_loader
import openpi.training.optimizer as _optimizer
import openpi.training.sharding as sharding
import openpi.training.utils as training_utils
import openpi.training.weight_loaders as _weight_loaders
import openpi.transforms as _transforms


def init_logging():
    """Custom logging format for better readability."""
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
        wandb.init(
            name=config.exp_name,
            config=dataclasses.asdict(config),
            project=config.project_name,
        )
        (ckpt_dir / "wandb_id.txt").write_text(wandb.run.id)

    if log_code:
        wandb.run.log_code(epath.Path(__file__).parent.parent)


def _load_and_filter_weights(loader, params_shape):
    loaded_params = loader.load(params_shape)
    flat_loaded = traverse_util.flatten_dict(loaded_params)

    mlp_to_moe_mapping = {
        ("PaliGemma","llm","layers","moe_1","expert_0","gating_einsum"): ("PaliGemma","llm","layers","mlp_1","gating_einsum"),
        ("PaliGemma","llm","layers","moe_1","expert_0","linear"): ("PaliGemma","llm","layers","mlp_1","linear"),
    }

    for moe_key, mlp_key in mlp_to_moe_mapping.items():
        if mlp_key in flat_loaded:
            flat_loaded[moe_key] = flat_loaded[mlp_key]  # 复制到 expert_0

    keys_to_remove = [k for k in flat_loaded if "mlp_1" in k]
    for k in keys_to_remove:
        flat_loaded.pop(k)

    return traverse_util.unflatten_dict(flat_loaded)


def update_params(orig_params, partial_params):
    """
    递归更新 orig_params 中对应 partial_params 的部分，保留其他不变
    """
    for k, v in partial_params.items():
        if isinstance(v, dict):
            if k not in orig_params:
                orig_params[k] = {}
            orig_params[k] = update_params(orig_params.get(k, {}), v)
        else:
            orig_params[k] = v
    return orig_params


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

        params = nnx_utils.state_map(params, config.freeze_filter, lambda p: p.replace(p.value.astype(jnp.bfloat16)))

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

    model = config.model.create(init_rng)
    params = nnx.state(model)
    params_dict = params.to_pure_dict() if hasattr(params, "to_pure_dict") else dict(params)
    def copy_moe0_to_mlp1(params_dict):
        flat = traverse_util.flatten_dict(params_dict)
        new_flat = dict(flat)  # shallow copy

        mapping = {
            ("PaliGemma","llm","layers","mlp_1","gating_einsum"): ("PaliGemma","llm","layers","moe_1","expert_0","gating_einsum"),
            ("PaliGemma","llm","layers","mlp_1","linear"): ("PaliGemma","llm","layers","moe_1","expert_0","linear"),
        }

        for new_key, src_key in mapping.items():
            if src_key in flat:
                new_flat[new_key] = flat[src_key]

        return traverse_util.unflatten_dict(new_flat)
    params_dict = copy_moe0_to_mlp1(params_dict)

    partial_params = _load_and_filter_weights(config.weight_loader, params_dict)
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
    batch: tuple[_model.Observation, _model.Actions],
) -> tuple[training_utils.TrainState, dict[str, at.Array]]:
    model = nnx.merge(state.model_def, state.params)
    model.train()

    @at.typecheck
    def loss_fn(
        model: _model.BaseModel, rng: at.KeyArrayLike, observation: _model.Observation, actions: _model.Actions
    ):
        chunked_loss, info = model.compute_loss(rng, observation, actions, train=True)
        return jnp.mean(chunked_loss), info

    train_rng = jax.random.fold_in(rng, state.step)
    observation, actions = batch

    diff_state = nnx.DiffState(0, config.trainable_filter)
    (loss, train_info), grads = nnx.value_and_grad(loss_fn, argnums=diff_state,has_aux=True)(model, train_rng, observation, actions)

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
                lambda old, new: state.ema_decay * old + (1 - state.ema_decay) * new, state.ema_params, new_params
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
    info.update(train_info)
    return new_state, info

def _create_output_transform(config: _config.TrainConfig) -> tuple[_transforms.DataTransformFn, _transforms.DataTransformFn]:
    """Creates the output transforms to be applied to model outputs and targets for validation."""
    data_config = config.data.create(config.assets_dirs, config.model)
    norm_stats = data_config.norm_stats
    output_transform = _transforms.CompositeTransform([
            *data_config.model_transforms.outputs,
            _transforms.Unnormalize(norm_stats, use_quantiles=data_config.use_quantile_norm),
            *data_config.data_transforms.outputs,
        ])
    
    # we need to filter out the ExtractFASTActions transform
    # since for data loader, the 'action' key is real action, not tokens
    _output_transform_list = output_transform.transforms
    _target_transform_list = [x for x in _output_transform_list if not isinstance(x, _transforms.ExtractFASTActions)]
    target_transform = _transforms.CompositeTransform(_target_transform_list)
    return output_transform, target_transform

@at.typecheck
def infer_step(
    config: _config.TrainConfig,
    rng: at.KeyArrayLike,
    state: training_utils.TrainState,
    batch: tuple[_model.Observation | _model.AtomicObservation, _model.Actions],
) -> dict[str, at.Array]:
    """
    Neural network inference for validation.
    The output transforms are not jittable, so they should be applied outside this function.

    Returns:
        outputs: A dictionary to compute MSE: 
        ```
        {
            "state": Array,
            "actions": Array,
            "targets": Array,
            "action_mask": Array,
            "other informatioin from the `model.sample_actions` function"
        }
        ```
    """
    if state.ema_decay is None:
        model = nnx.merge(state.model_def, state.params)
    else:
        model = nnx.merge(state.model_def, state.ema_params)
    model.eval()

    sample_actions = nnx_utils.module_jit(model.sample_actions)

    infer_rng = jax.random.fold_in(rng, state.step)

    observation, targets = batch
    actions, val_info = sample_actions(rng=infer_rng, observation=observation)

    # deep copy to avoid inplace modification
    _inputs = jax.tree.map(lambda x: x, observation)
    _targets = jax.tree.map(lambda x: x, targets)
    _action_mask = jnp.ones(actions.shape[0], dtype=jnp.bool_)
    if isinstance(observation, _model.AtomicObservation):
        _action_mask = _inputs.diffusion_loss_mask

    outputs = {
        "state": _inputs.state,
        "actions": actions,
        "targets": _targets,
        "action_mask": _action_mask,
        **val_info,
    }
    return outputs

@at.typecheck
def compute_mse(
        state: at.Float[at.Array, 'b s'],
        actions: at.Float[at.Array, 'b ah ad'],
        targets: at.Float[at.Array, 'b ah ad'],
        action_mask: at.Bool[at.Array, 'b'],
        output_transform: _transforms.DataTransformFn,
        target_transform: _transforms.DataTransformFn,
        ) -> dict[str, at.ArrayLike]:
    batch_size = state.shape[0]
    errors = []
    for i in range(batch_size):
        state_i = np.asarray(state[i])
        action_i = np.asarray(actions[i])
        target_i = np.asarray(targets[i])
        
        transformed_action_i = output_transform({
            "state": state_i,
            "actions": action_i
        })['actions']
        transformed_target_i = target_transform({
            "state": state_i,
            "actions": target_i
        })['actions']
        errors.append(transformed_target_i - transformed_action_i)
    
    errors = np.asanyarray(errors)
    broadcasted_action_mask = np.broadcast_to(
        action_mask[:, None, None], errors.shape)
    if np.sum(broadcasted_action_mask) == 0:
        mse = np.nan
    else:
        mse = np.mean(errors[broadcasted_action_mask] ** 2)
    return {'action_mse': mse,
            'num_action_loss_fraction': np.sum(action_mask) / batch_size}

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

    data_loader = _data_loader.create_data_loader(
        config,
        sharding=data_sharding,
        shuffle=True,
    )
    data_iter = iter(data_loader)
    batch = next(data_iter)
    logging.info(f"Initialized data loader:\n{training_utils.array_tree_to_info(batch)}")
    # training_utils.inspect_prompts(batch)          # this inspect_prompts does not return anything, so comment it out for now

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
