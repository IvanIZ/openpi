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

        self._tokenizer = _tokenizer.PaligemmaTokenizer(max_len=50)
        self._is_fuse_model = False

        if self._is_pytorch_model:
            self._model = self._model.to(pytorch_device)
            self._model.eval()
            self._sample_actions = model.sample_actions
        else:
            # JAX model setup
            self._sample_actions = nnx_utils.module_jit(model.sample_actions)
            self._rng = rng or jax.random.key(0)
            if hasattr(model, 'prefill'):
                self._is_fuse_model = True
                self._prefill = nnx_utils.module_jit(model.prefill)
                self._reason = nnx_utils.module_jit(model.reason)
                self._act = nnx_utils.module_jit(model.act)

    def _decode_reasoning_tokens(self, raw_tokens: list[int]) -> str | None:
        """Decode generated reasoning tokens while removing protocol/control tokens."""
        kept_tokens: list[int] = []
        for token in raw_tokens[1:]:
            if token == _tokenizer.PALIGEMMA_EOS_TOKEN:
                break
            if token in {
                0,
                _tokenizer.BEGIN_OF_ACTION,
                _tokenizer.BEGIN_OF_REASONING,
                _tokenizer.END_OF_PREFIX_TOKEN,
            }:
                continue
            kept_tokens.append(token)

        if not kept_tokens:
            return None

        text = self._tokenizer._tokenizer.decode(kept_tokens)
        text = re.sub(r"<(?:loc|seg)\d+>", " ", text)
        text = " ".join(text.split())
        return text or None


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

        if "diffusion_loss_mask" in inputs:
            observation = _model.FuseObservation.from_dict(inputs)
        else:
            observation = _model.Observation.from_dict(inputs)

        subtask = None

        if self._is_fuse_model:
            prefill_rng, reason_rng, action_rng = jax.random.split(sample_rng_or_pytorch_device, 3)
            processed_obs, kv_cache, _, eop_logit, prefix_mask, prefix_positions, has_boa = \
                self._prefill(prefill_rng, observation)

            # Diagnostic: show prefill decision and logits for the two special tokens.
            eop_logit_np = np.asarray(eop_logit[0, 0])
            boa_logit = float(eop_logit_np[257021])  # BEGIN_OF_ACTION
            bor_logit = float(eop_logit_np[257020])  # BEGIN_OF_REASONING
            logging.info(
                f"[Fuse] prefill decision: has_boa={bool(np.asarray(has_boa).item())}, "
                f"logit(BOA)={boa_logit:.2f}, logit(BOR)={bor_logit:.2f}"
            )

            # Always generate reasoning tokens. reason() forces BEGIN_OF_REASONING
            # at step 0 regardless of the prefill decision, so this works even when
            # has_boa=True. This matches actalign which forces thinking when the
            # scene plan is empty, and in general the reasoning output is always
            # useful context for subsequent calls.
            reasoning_tokens = self._reason(
                reason_rng, eop_logit, kv_cache, prefix_mask, prefix_positions
            )
            raw = np.asarray(reasoning_tokens[0]).tolist()
            subtask = self._decode_reasoning_tokens(raw)

            # Generate actions using the prefix KV cache from prefill
            actions = self._act(
                action_rng, processed_obs, kv_cache, prefix_mask, prefix_positions,
            )
            all_actions = (actions, {})
        else:
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


class ReasoningPolicy(BasePolicy):
    """Stateful inference policy for Pi0Fuse-style think-then-act serving."""

    def __init__(
        self,
        model: _model.BaseModel,
        *,
        rng: at.KeyArrayLike | None = None,
        transforms: Sequence[_transforms.DataTransformFn] = (),
        output_transforms: Sequence[_transforms.DataTransformFn] = (),
        sample_kwargs: dict[str, Any] | None = None,
        metadata: dict[str, Any] | None = None,
        initial_scene_plan: str = "",
        force_initial_reasoning: bool = True,
    ):
        for method_name in ("prefill", "reason", "act"):
            if not hasattr(model, method_name):
                raise ValueError(f"ReasoningPolicy requires model.{method_name}(), but it is missing.")

        self._prefill = nnx_utils.module_jit(model.prefill, static_argnames=("temperature",))
        self._reason = nnx_utils.module_jit(
            model.reason,
            static_argnames=("temperature", "max_decoding_steps"),
        )
        self._act = nnx_utils.module_jit(model.act)

        self._input_transform = _transforms.compose(transforms)
        self._output_transform = _transforms.compose(output_transforms)
        self._rng = rng or jax.random.key(0)
        self._sample_kwargs = sample_kwargs or {}
        self._metadata = metadata or {}

        self._tokenizer = _tokenizer.PaligemmaTokenizer(max_len=50)

        self._temperature = float(self._sample_kwargs.get("temperature", 0.0))
        
        self._max_reasoning_steps = int(self._sample_kwargs.get("max_reasoning_steps", 256))
        self._force_initial_reasoning = bool(
            self._sample_kwargs.get("force_initial_reasoning", force_initial_reasoning)
        )

        self._initial_scene_plan = initial_scene_plan
        self._scene_plan = initial_scene_plan
        self._instruction: str | None = None
        self._thought: str | None = None

        self._lock = threading.Lock()
        self._is_thinking = False

    def start(self) -> None:
        """Reset rollout-local reasoning state."""
        self._scene_plan = self._initial_scene_plan
        self._instruction = None
        self._thought = None
        self.is_thinking = False

    def _prepare_obs(self, obs: dict) -> dict:
        if "prompt" not in obs:
            raise ValueError("ReasoningPolicy expects a 'prompt' key in observations.")

        prompt = obs["prompt"]
        if not isinstance(prompt, str):
            prompt = str(prompt.item() if hasattr(prompt, "item") else prompt)

        if self._instruction is None:
            self._instruction = f"Instruction: {prompt}. \n"
        if self._thought is None:
            self._thought = f"{self._instruction}{self._scene_plan}"

        obs["thought"] = [self._thought]
        obs["act_with_outdated_thought"] = False
        obs["think_with_outdated_thought"] = False
        return obs

    def _decode_reasoning_tokens(self, raw_tokens: list[int]) -> str | None:
        kept_tokens: list[int] = []
        for token in raw_tokens[1:]:
            if token == _tokenizer.PALIGEMMA_EOS_TOKEN:
                break
            if token in {
                0,
                _tokenizer.BEGIN_OF_ACTION,
                _tokenizer.BEGIN_OF_REASONING,
                _tokenizer.END_OF_PREFIX_TOKEN,
            }:
                continue
            kept_tokens.append(token)

        if not kept_tokens:
            return None

        text = self._tokenizer._tokenizer.decode(kept_tokens)
        text = re.sub(r"<(?:loc|seg)\d+>", " ", text)
        text = " ".join(text.split())
        return text or None

    def _update_thought(self, scene_plan: str | None) -> None:
        if scene_plan:
            self._scene_plan = scene_plan
        self._thought = f"{self._instruction}{self._scene_plan}"

    @override
    def infer(self, obs: dict, *, noise: np.ndarray | None = None) -> dict:  # type: ignore[override]
        del noise  # ReasoningPolicy does not consume external diffusion noise.

        # Make a copy since transformations may modify the inputs in place.
        inputs = jax.tree.map(lambda x: x, obs)
        inputs = self._prepare_obs(inputs)
        inputs = self._input_transform(inputs)
        inputs = jax.tree.map(lambda x: jnp.asarray(x)[np.newaxis, ...], inputs)
        observation = _model.FuseObservation.from_dict(inputs)

        prefill_rng, reason_rng, action_rng, self._rng = jax.random.split(self._rng, 4)
        processed_obs, kv_cache, _, eop_logit, prefix_mask, prefix_positions, has_boa = self._prefill(
            prefill_rng, observation, temperature=self._temperature
        )

        to_act = bool(np.asarray(has_boa).item())
        to_think = not to_act
        if self._force_initial_reasoning and self._scene_plan == self._initial_scene_plan:
            to_think = True
            to_act = False

        self.is_thinking = to_think

        if to_think:
            print("mode thinking now...")
        elif to_act:
            print("mode acting now...")
        else:
            print("mode doing nothing...")

        if to_think:
            reasoning_tokens = self._reason(
                reason_rng,
                eop_logit,
                kv_cache,
                prefix_mask,
                prefix_positions,
                temperature=self._temperature,
                max_decoding_steps=self._max_reasoning_steps,
            )
            raw_reasoning_tokens = np.asarray(reasoning_tokens[0]).tolist()
            scene_plan = self._decode_reasoning_tokens(raw_reasoning_tokens)
            self._update_thought(scene_plan)
            self.is_thinking = False
            return {
                "isthinking": np.True_,
                "thought": scene_plan or "",
                "subtask": self._scene_plan,
            }

        actions = self._act(
            action_rng,
            processed_obs,
            kv_cache,
            prefix_mask,
            prefix_positions,
        )
        outputs = {
            "state": np.asarray(inputs["state"][0, ...]),
            "actions": np.asarray(actions[0, ...]),
            "subtask": self._scene_plan,
            "isthinking": np.False_,
        }
        transformed = self._output_transform(outputs)
        transformed["isthinking"] = np.False_
        return transformed

    @property
    def is_thinking(self) -> bool:
        with self._lock:
            return self._is_thinking

    @is_thinking.setter
    def is_thinking(self, value: bool) -> None:
        with self._lock:
            self._is_thinking = value

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
