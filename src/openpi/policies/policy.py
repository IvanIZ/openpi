from collections.abc import Sequence
import logging
import pathlib
import re
import threading
import time
from typing import Any, TypeAlias

import flax
import flax.traverse_util
import jax
import jax.numpy as jnp
import numpy as np
from openpi_client import base_policy as _base_policy
import torch
from typing_extensions import override

from openpi import transforms as _transforms
from openpi.models import gemmoe as _gemmoe
from openpi.models import model as _model
from openpi.models import tokenizer as _tokenizer
from openpi.shared import array_typing as at
from openpi.shared import nnx_utils

BasePolicy: TypeAlias = _base_policy.BasePolicy


class Policy(BasePolicy):
    def __init__(
        self,
        model: _model.BaseModel,
        *,
        rng: at.KeyArrayLike | None = None,
        transforms: Sequence[_transforms.DataTransformFn] = (),
        output_transforms: Sequence[_transforms.DataTransformFn] = (),
        sample_kwargs: dict[str, Any] | None = None,
        metadata: dict[str, Any] | None = None,
        pytorch_device: str = "cpu",
        is_pytorch: bool = False,
    ):
        """Initialize the Policy.

        Args:
            model: The model to use for action sampling.
            rng: Random number generator key for JAX models. Ignored for PyTorch models.
            transforms: Input data transformations to apply before inference.
            output_transforms: Output data transformations to apply after inference.
            sample_kwargs: Additional keyword arguments to pass to model.sample_actions.
            metadata: Additional metadata to store with the policy.
            pytorch_device: Device to use for PyTorch models (e.g., "cpu", "cuda:0").
                          Only relevant when is_pytorch=True.
            is_pytorch: Whether the model is a PyTorch model. If False, assumes JAX model.
        """
        self._model = model
        self._input_transform = _transforms.compose(transforms)
        self._output_transform = _transforms.compose(output_transforms)
        self._sample_kwargs = sample_kwargs or {}
        self._metadata = metadata or {}
        self._is_pytorch_model = is_pytorch
        self._pytorch_device = pytorch_device
        self._debug_prefill = bool(self._sample_kwargs.get("debug_prefill", False))

        self._tokenizer = _tokenizer.PaligemmaTokenizer(max_len=50)

        if self._is_pytorch_model:
            self._model = self._model.to(pytorch_device)
            self._model.eval()
            self._sample_actions = model.sample_actions
        else:
            # JAX model setup
            self._sample_actions = nnx_utils.module_jit(model.sample_actions)
            self._rng = rng or jax.random.key(0)

    @override
    def infer(self, obs: dict, *, noise: np.ndarray | None = None) -> dict:  # type: ignore[misc]
        # Make a copy since transformations may modify the inputs in place.
        inputs = jax.tree.map(lambda x: x, obs)
        inputs = self._input_transform(inputs)
        if not self._is_pytorch_model:
            # Make a batch and convert to jax.Array.
            inputs = jax.tree.map(lambda x: jnp.asarray(x)[np.newaxis, ...], inputs)
            self._rng, sample_rng_or_pytorch_device = jax.random.split(self._rng)
        else:
            # Convert inputs to PyTorch tensors and move to correct device
            inputs = jax.tree.map(lambda x: torch.from_numpy(np.array(x)).to(self._pytorch_device)[None, ...], inputs)
            sample_rng_or_pytorch_device = self._pytorch_device

        # Prepare kwargs for sample_actions
        sample_kwargs = dict(self._sample_kwargs)
        if noise is not None:
            noise = torch.from_numpy(noise).to(self._pytorch_device) if self._is_pytorch_model else jnp.asarray(noise)

            if noise.ndim == 2:  # If noise is (action_horizon, action_dim), add batch dimension
                noise = noise[None, ...]  # Make it (1, action_horizon, action_dim)
            sample_kwargs["noise"] = noise

        observation = _model.Observation.from_dict(inputs)

        subtask = None

        all_actions = self._sample_actions(sample_rng_or_pytorch_device, observation, **sample_kwargs)

        if isinstance(all_actions, tuple):
            actions, second = all_actions
            outputs = {"state": inputs["state"], "actions": actions}
            if subtask is None and not isinstance(second, dict):
                subtask = self._tokenizer._tokenizer.decode(second[second != 0].tolist())
        else:
            outputs = {"state": inputs["state"], "actions": all_actions}

        start_time = time.monotonic()
        model_time = time.monotonic() - start_time
        if self._is_pytorch_model:
            outputs = jax.tree.map(lambda x: np.asarray(x[0, ...].detach().cpu()), outputs)
        else:
            outputs = jax.tree.map(lambda x: np.asarray(x[0, ...]), outputs)

        outputs['subtask'] = subtask
        outputs = self._output_transform(outputs)
        outputs["policy_timing"] = {
            "infer_ms": model_time * 1000,
        }
        return outputs

    @property
    def metadata(self) -> dict[str, Any]:
        return self._metadata


class TraceVLAPolicy(BasePolicy):
    """Inference policy for ``Pi0TraceVLA``.

    Exposes two endpoints, mirroring the model's two forward passes:

      - :meth:`infer` (default endpoint, override of :class:`BasePolicy`):
        runs the **execution forward** — both action denoising and skill-completion
        prediction in one shared prefix prefill. Returns
        ``{"state", "actions", "progress", ...}`` where ``progress`` is the per-skill
        completion-progress scalar in ``[0, 1]``.

      - :meth:`sample_trace`: runs the **planning forward** — generates a
        ``(N, 2)`` normalized-image-space trace via flow matching, conditioned on
        the semantic target keypoint and the current EE keypoint. The caller is
        expected to render this trace as the overlay image and feed it back into
        :meth:`infer` via ``obs["observation/overlay_image"]``.

    Closed-loop trace caching (i.e. how often to call ``sample_trace`` vs
    ``infer``) is left to the caller — typical deployment runs ``infer`` at the
    control loop rate and ``sample_trace`` at a lower rate, switching skills
    when ``progress`` exceeds a threshold.
    """

    def __init__(
        self,
        model: _model.BaseModel,
        *,
        rng: at.KeyArrayLike | None = None,
        transforms: Sequence[_transforms.DataTransformFn] = (),
        output_transforms: Sequence[_transforms.DataTransformFn] = (),
        sample_kwargs: dict[str, Any] | None = None,
        metadata: dict[str, Any] | None = None,
    ):
        # Validate the model exposes the public completion / trace endpoints we need.
        for method_name in ("sample_actions_and_completion", "sample_trace"):
            if not hasattr(model, method_name):
                raise ValueError(
                    f"TraceVLAPolicy requires model.{method_name}(); the model passed "
                    f"in does not expose it. Use a Pi0TraceVLA-derived model."
                )

        self._model = model
        self._sample_actions_and_completion = nnx_utils.module_jit(model.sample_actions_and_completion)
        self._sample_trace = nnx_utils.module_jit(model.sample_trace)
        self._predict_completion = nnx_utils.module_jit(model.predict_completion) \
            if hasattr(model, "predict_completion") else None

        self._input_transform = _transforms.compose(transforms)
        self._output_transform = _transforms.compose(output_transforms)
        self._sample_kwargs = sample_kwargs or {}
        self._metadata = metadata or {}
        # Use explicit `is None` rather than `rng or ...` because a JAX PRNGKey is
        # a 0-d array and would raise on `bool()`.
        self._rng = jax.random.key(0) if rng is None else rng

    def _prepare_inputs(self, obs: dict) -> tuple[dict, "_model.Observation"]:
        """Apply input transforms, batchify, and build a TraceObservation."""
        from openpi.models import trace_observation as _trace_obs  # local import to avoid cycles
        inputs = jax.tree.map(lambda x: x, obs)
        inputs = self._input_transform(inputs)
        inputs = jax.tree.map(lambda x: jnp.asarray(x)[np.newaxis, ...], inputs)
        observation = _trace_obs.TraceObservation.from_dict(inputs)
        return inputs, observation

    @override
    def infer(self, obs: dict, *, noise: np.ndarray | None = None) -> dict:  # type: ignore[misc]
        inputs, observation = self._prepare_inputs(obs)
        self._rng, sample_rng = jax.random.split(self._rng)

        sample_kwargs = dict(self._sample_kwargs)
        if noise is not None:
            n = jnp.asarray(noise)
            if n.ndim == 2:  # (action_horizon, action_dim) -> add batch dim
                n = n[None, ...]
            sample_kwargs["noise"] = n

        start_time = time.monotonic()
        actions, progress = self._sample_actions_and_completion(sample_rng, observation, **sample_kwargs)
        model_time = time.monotonic() - start_time

        outputs = {
            "state": inputs["state"],
            "actions": actions,
            "progress": progress,
        }
        # Unbatch and convert to numpy.
        outputs = jax.tree.map(lambda x: np.asarray(x[0, ...]), outputs)
        outputs = self._output_transform(outputs)
        outputs["policy_timing"] = {"infer_ms": model_time * 1000.0}
        return outputs

    def sample_trace(self, obs: dict, *, num_steps: int = 10) -> np.ndarray:
        """Sample a normalized image-space trace from the planning forward.

        Returns a ``(N, 2)`` numpy array with each waypoint in ``[0, 1]^2``. The caller
        is responsible for de-normalizing to the target image resolution and rendering
        the overlay (e.g. via ``trace_utils.draw_polyline_overlay``).
        """
        _inputs, observation = self._prepare_inputs(obs)
        self._rng, plan_rng = jax.random.split(self._rng)
        trace = self._sample_trace(plan_rng, observation, num_steps=num_steps)
        return np.asarray(trace[0])

    def predict_completion(self, obs: dict) -> np.ndarray:
        """Standalone completion-progress query (no action sampling).

        Cheaper than :meth:`infer` when actions are not needed this step (e.g. running
        completion checks at a lower cadence than the action loop). Returns a
        scalar in ``[0, 1]``.
        """
        if self._predict_completion is None:
            raise RuntimeError("Underlying model does not expose `predict_completion`.")
        _inputs, observation = self._prepare_inputs(obs)
        self._rng, query_rng = jax.random.split(self._rng)
        progress = self._predict_completion(query_rng, observation)
        return np.asarray(progress[0])

    @property
    def metadata(self) -> dict[str, Any]:
        return self._metadata


class PolicyRecorder(_base_policy.BasePolicy):
    """Records the policy's behavior to disk."""

    def __init__(self, policy: _base_policy.BasePolicy, record_dir: str):
        self._policy = policy

        logging.info(f"Dumping policy records to: {record_dir}")
        self._record_dir = pathlib.Path(record_dir)
        self._record_dir.mkdir(parents=True, exist_ok=True)
        self._record_step = 0

    @override
    def infer(self, obs: dict) -> dict:  # type: ignore[misc]
        results = self._policy.infer(obs)

        data = {"inputs": obs, "outputs": results}
        data = flax.traverse_util.flatten_dict(data, sep="/")

        output_path = self._record_dir / f"step_{self._record_step}"
        self._record_step += 1

        np.save(output_path, np.asarray(data))
        return results
