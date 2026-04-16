import logging
import dataclasses

import einops
import flax.nnx as nnx
import flax.nnx.bridge as nnx_bridge
import jax
import jax.numpy as jnp
import numpy as np
from typing_extensions import override

from openpi.models import model as _model
from openpi.models import pi0_config
import openpi.models.gemmoe as _gemma
import openpi.models.siglip as _siglip
import openpi.models.tokenizer as _tokenizer
from openpi.shared import array_typing as at
import openpi.shared.nnx_utils as nnx_utils

logger = logging.getLogger("openpi")

PALIGEMMA_EOS_TOKEN = 1


def put_along_last_axis(arr, indices, values):
    """Like np.put_along_axis(..., axis=-1), since jax is missing it."""
    assert arr.ndim == indices.ndim == values.ndim, (arr.ndim, indices.ndim, values.ndim)
    onehot = jax.nn.one_hot(indices, arr.shape[-1], dtype=values.dtype)
    put_mask = jnp.einsum("...i,...in->...n", jnp.ones(values.shape, jnp.int32), onehot)
    put_values = jnp.einsum("...i,...in->...n", values, onehot)
    return jnp.where(put_mask, put_values, arr)


def make_attn_mask(input_mask, mask_ar):
    """Adapted from big_vision.

    Tokens can attend to valid inputs tokens which have a cumulative mask_ar
    smaller or equal to theirs. This way `mask_ar` bool[?B, N] can be used to
    setup several types of attention, for example:

      [[1 1 1 1 1 1]]: pure causal attention.

      [[0 0 0 1 1 1]]: prefix-lm attention. The first 3 tokens can attend between
          themselves and the last 3 tokens have a causal attention. The first
          entry could also be a 1 without changing behaviour.

      [[1 0 1 0 1 0 0 1 0 0]]: causal attention between 4 blocks. Tokens of a
          block can attend all previous blocks and all tokens on the same block.

    Args:
      input_mask: bool[B, N] true if its part of the input, false if padding.
      mask_ar: bool[?B, N] mask that's true where previous tokens cannot depend on
        it and false where it shares the same attention mask as the previous token.
    """
    mask_ar = jnp.broadcast_to(mask_ar, input_mask.shape)
    cumsum = jnp.cumsum(mask_ar, axis=1)
    attn_mask = cumsum[:, None, :] <= cumsum[:, :, None]
    valid_mask = input_mask[:, None, :] * input_mask[:, :, None]
    return jnp.logical_and(attn_mask, valid_mask)


@at.typecheck
def posemb_sincos(
    pos: at.Real[at.Array, " b"], embedding_dim: int, min_period: float, max_period: float
) -> at.Float[at.Array, "b {embedding_dim}"]:
    """Computes sine-cosine positional embedding vectors for scalar positions."""
    if embedding_dim % 2 != 0:
        raise ValueError(f"embedding_dim ({embedding_dim}) must be divisible by 2")

    fraction = jnp.linspace(0.0, 1.0, embedding_dim // 2)
    period = min_period * (max_period / min_period) ** fraction
    sinusoid_input = jnp.einsum(
        "i,j->ij",
        pos,
        1.0 / period * 2 * jnp.pi,
        precision=jax.lax.Precision.HIGHEST,
    )
    return jnp.concatenate([jnp.sin(sinusoid_input), jnp.cos(sinusoid_input)], axis=-1)


class Pi0Atomic(_model.BaseModel):
    def __init__(self, config: pi0_config.Pi0AtomicConfig, rngs: nnx.Rngs):
        super().__init__(config.action_dim, config.action_horizon, config.max_token_len)
        self.pi05 = config.pi05
        paligemma_config = _gemma.get_config(config.paligemma_variant)
        action_expert_config = _gemma.get_config(config.action_expert_variant)
        n = action_expert_config.num_local_experts
        target_weight = 0.5
        desired_logit = np.log((n - 1) * target_weight / (1 - target_weight))
        mean_sigma_scale = (10.0 + 100.0) / 2.0
        router_kernel_scale = desired_logit / mean_sigma_scale

        # TODO: rewrite gemma in NNX. For now, use bridge.
        llm = nnx_bridge.ToNNX(
            _gemma.Module(
                configs=[paligemma_config, action_expert_config],
                embed_dtype=config.dtype,
                adarms=config.pi05,
                router_kernel_scale=router_kernel_scale,
            )
        )
        llm.lazy_init(rngs=rngs, method="init", use_adarms=[False, True] if config.pi05 else [False, False])
        img = nnx_bridge.ToNNX(
            _siglip.Module(
                num_classes=paligemma_config.width,
                variant="So400m/14",
                pool_type="none",
                scan=True,
                dtype_mm=config.dtype,
            )
        )
        img.lazy_init(next(iter(config.fake_obs().images.values())), train=False, rngs=rngs)
        self.PaliGemma = nnx.Dict(llm=llm, img=img)
        self.action_in_proj = nnx.Linear(config.action_dim, action_expert_config.width, rngs=rngs)
        if config.pi05:
            self.time_mlp_in = nnx.Linear(action_expert_config.width, action_expert_config.width, rngs=rngs)
            self.time_mlp_out = nnx.Linear(action_expert_config.width, action_expert_config.width, rngs=rngs)
        else:
            self.state_proj = nnx.Linear(config.action_dim, action_expert_config.width, rngs=rngs)
            self.action_time_mlp_in = nnx.Linear(2 * action_expert_config.width, action_expert_config.width, rngs=rngs)
            self.action_time_mlp_out = nnx.Linear(action_expert_config.width, action_expert_config.width, rngs=rngs)
        self.action_out_proj = nnx.Linear(action_expert_config.width, config.action_dim, rngs=rngs)

        self.num_local_experts = action_expert_config.num_local_experts
        scales = jnp.linspace(10.0, 100.0, self.num_local_experts)
        base = jnp.eye(self.num_local_experts) * scales[:, None]
        if action_expert_config.width > self.num_local_experts:
            base = jnp.pad(base, ((0, 0), (0, action_expert_config.width - self.num_local_experts)))
        self.sigma_emb = nnx.Variable(base, trainable=False)

        self.deterministic = True

    @at.typecheck
    def embed_prefix(
        self, obs: _model.AtomicObservation
    ) -> tuple[at.Float[at.Array, "b s emb"], at.Bool[at.Array, "b s"], at.Int[at.Array, "b s"]]:
        input_mask = []
        ar_mask = []
        tokens = []

        for name in obs.images:
            image_tokens, _ = self.PaliGemma.img(obs.images[name], train=False)
            tokens.append(image_tokens)
            input_mask.append(
                einops.repeat(
                    obs.image_masks[name],
                    "b -> b s",
                    s=image_tokens.shape[1],
                )
            )
            ar_mask.append(0 * input_mask[-1])

        assert obs.tokenized_prompt is not None, "Tokenized prompt is required"
        assert obs.tokenized_prompt_mask is not None, "Tokenized prompt mask is required"
        assert obs.token_ar_mask is not None, "Token auto-regressive mask is required"
        assert obs.token_loss_mask is not None, "Token loss mask is required"

        txt_emb = self.PaliGemma.llm(obs.tokenized_prompt, method="embed")
        tokens.append(txt_emb)
        input_mask.append(obs.tokenized_prompt_mask)
        ar_mask.append(obs.token_ar_mask)

        tokens = jnp.concatenate(tokens, axis=1)
        input_mask = jnp.concatenate(input_mask, axis=1)
        ar_mask = jnp.concatenate(ar_mask, axis=1)
        return tokens, input_mask, ar_mask

    @at.typecheck
    def embed_atomic_skill(self, obs: _model.AtomicObservation) -> at.Float[at.Array, "b s emb"]:
        atomic_embed_mid = self.sigma_emb[obs.atomic_token.astype(jnp.int32)]
        atomic_embed = einops.repeat(atomic_embed_mid, f"b emb -> b {self.action_horizon} emb")
        return atomic_embed

    @at.typecheck
    def embed_suffix(
        self, obs: _model.AtomicObservation, noisy_actions: _model.Actions, timestep: at.Float[at.Array, " b"]
    ) -> tuple[
        at.Float[at.Array, "b s emb"],
        at.Bool[at.Array, "b s"],
        at.Bool[at.Array, "b s"],
        at.Float[at.Array, "b emb"] | None,
    ]:
        input_mask = []
        ar_mask = []
        tokens = []
        if not self.pi05:
            state_token = self.state_proj(obs.state)[:, None, :]
            tokens.append(state_token)
            input_mask.append(jnp.ones((obs.state.shape[0], 1), dtype=jnp.bool_))
            ar_mask += [True]

        action_tokens = self.action_in_proj(noisy_actions)
        time_emb = posemb_sincos(timestep, self.action_in_proj.out_features, min_period=4e-3, max_period=4.0)
        if self.pi05:
            time_emb = self.time_mlp_in(time_emb)
            time_emb = nnx.swish(time_emb)
            time_emb = self.time_mlp_out(time_emb)
            time_emb = nnx.swish(time_emb)
            action_expert_tokens = action_tokens
            adarms_cond = time_emb
        else:
            time_tokens = einops.repeat(time_emb, "b emb -> b s emb", s=self.action_horizon)
            action_time_tokens = jnp.concatenate([action_tokens, time_tokens], axis=-1)
            action_time_tokens = self.action_time_mlp_in(action_time_tokens)
            action_time_tokens = nnx.swish(action_time_tokens)
            action_time_tokens = self.action_time_mlp_out(action_time_tokens)
            action_expert_tokens = action_time_tokens
            adarms_cond = None
        tokens.append(action_expert_tokens)
        input_mask.append(jnp.ones(action_expert_tokens.shape[:2], dtype=jnp.bool_))
        ar_mask += [True] + ([False] * (self.action_horizon - 1))
        tokens = jnp.concatenate(tokens, axis=1)
        input_mask = jnp.concatenate(input_mask, axis=1)
        ar_mask = jnp.array(ar_mask)
        ar_mask = jnp.broadcast_to(ar_mask, (tokens.shape[0], tokens.shape[1]))
        return tokens, input_mask, ar_mask, adarms_cond

    @override
    def compute_loss(
        self, rng: at.KeyArrayLike, observation: _model.AtomicObservation, actions: _model.Actions, *, train: bool = False
    ) -> at.Float[at.Array, "*b ah"]:
        preprocess_rng, noise_rng, time_rng = jax.random.split(rng, 3)
        observation = _model.preprocess_observation(
            preprocess_rng, observation, train=train, image_keys=list(observation.images.keys())
        )

        batch_shape = actions.shape[:-2]
        noise = jax.random.normal(noise_rng, actions.shape)
        time = jax.random.beta(time_rng, 1.5, 1, batch_shape) * 0.999 + 0.001
        time_expanded = time[..., None, None]
        x_t = time_expanded * noise + (1 - time_expanded) * actions
        u_t = noise - actions

        atomic_cond_embed = self.embed_atomic_skill(observation)
        prefix_tokens, prefix_mask, prefix_ar_mask = self.embed_prefix(observation)
        suffix_tokens, suffix_mask, suffix_ar_mask, adarms_cond = self.embed_suffix(observation, x_t, time)
        input_mask = jnp.concatenate([prefix_mask, suffix_mask], axis=1)
        ar_mask = jnp.concatenate([prefix_ar_mask, suffix_ar_mask], axis=1)
        attn_mask = make_attn_mask(input_mask, ar_mask)
        positions = jnp.cumsum(input_mask, axis=1) - 1
        (prefix_out, suffix_out), _ = self.PaliGemma.llm(
            [prefix_tokens, suffix_tokens],
            mask=attn_mask,
            positions=positions,
            adarms_cond=[None, adarms_cond],
            cond=atomic_cond_embed,
        )

        # Text next-token prediction loss
        shifted_targets = observation.tokenized_prompt[:, 1:]
        txt_logits = self.PaliGemma.llm(
            prefix_out[:, -shifted_targets.shape[1] - 1: -1],
            method="embedder_decode",
        )
        txt_logp = jax.nn.log_softmax(txt_logits, axis=-1)
        txt_loss_mask = observation.token_loss_mask[:, 1:]
        txt_token_pplx = jnp.take_along_axis(
            txt_logp, shifted_targets[..., None], axis=-1
        )[..., 0]

        # Split: decision token (first suffix token) vs rest (reasoning text)
        first_suffix_pos = jnp.argmax(txt_loss_mask.astype(jnp.int32), axis=-1)
        decision_mask = jnp.zeros_like(txt_loss_mask)
        decision_mask = decision_mask.at[
            jnp.arange(decision_mask.shape[0]), first_suffix_pos
        ].set(True)
        rest_mask = txt_loss_mask & ~decision_mask

        decision_loss = -jnp.sum(txt_token_pplx * decision_mask, axis=-1)
        rest_loss = (
            -jnp.sum(txt_token_pplx * rest_mask, axis=-1) /
            jnp.clip(jnp.sum(rest_mask, axis=-1), 1)
        )

        # Action flow matching loss
        v_t = self.action_out_proj(suffix_out[:, -self.action_horizon:])
        action_loss = jnp.mean(jnp.mean(jnp.square(v_t - u_t), axis=-1), axis=1)
        loss = decision_loss / 20.0 + rest_loss / 50.0 + action_loss
        info = {
            'text_loss': rest_loss,
            'decision_loss': decision_loss,
            'action_loss': action_loss,
        }

        return loss, info

    @at.typecheck
    def prefill(
        self,
        rng: at.KeyArrayLike,
        observation: _model.AtomicObservation,
        *,
        temprature: float = 0.,
    ) -> tuple[
        _model.AtomicObservation,
        _gemma.KVCache,
        at.Int[at.Array, "b 1"],
        at.Float[at.Array, "b 1 v"],
        at.Bool[at.Array, "b s"],
        at.Int[at.Array, "b s"],
        at.Bool[at.Array, "b"],
    ]:
        """Prefill the prefix. Used for policy serving.

        Args:
            rng: PRNG key.
            observation: AtomicObservation.
            temprature: decoding temprature.

        Returns:
            tuple containing:
                - observation: preprocessed observation.
                - kv_cache: KV cache for the prefix.
                - token: the next token after <END_OF_PREFIX>.
                - eop_logit: logit for the <END_OF_PREFIX> token.
                - prefix_mask: input mask for the prefix.
                - prefix_positions: position id for the prefix.
                - to_act: whether to act or to reason.
        """
        observation = _model.preprocess_observation(
            None, observation, train=False, image_keys=list(observation.images.keys())
        )
        first_one_indices = jnp.argmax(observation.token_ar_mask, axis=-1)
        padding_mask = jnp.arange(observation.token_ar_mask.shape[-1]) >= first_one_indices[..., jnp.newaxis]
        observation = dataclasses.replace(
            observation,
            tokenized_prompt=jnp.where(padding_mask, 0, observation.tokenized_prompt),
            tokenized_prompt_mask=jnp.logical_not(padding_mask),
        )

        prefix_tokens, prefix_mask, prefix_ar_mask = self.embed_prefix(observation)
        prefix_attn_mask = make_attn_mask(prefix_mask, prefix_ar_mask)
        prefix_positions = jnp.cumsum(prefix_mask, axis=1) - 1
        (pre_logit, _), kv_cache = self.PaliGemma.llm(
            [prefix_tokens, None],
            mask=prefix_attn_mask,
            positions=prefix_positions,
            adarms_cond=[None, None],
            cond=None,
        )

        eop_indices = prefix_positions[:, -1]
        eop_pre_logit = jnp.take_along_axis(pre_logit, eop_indices[:, None, None], axis=1)
        eop_logit = self.PaliGemma.llm(eop_pre_logit, method="embedder_decode")

        valid_tokens = jnp.array([_tokenizer.BEGIN_OF_ACTION, _tokenizer.BEGIN_OF_REASONING])
        valid_mask = jnp.full((1, 1, eop_logit.shape[-1]), -jnp.inf)
        valid_mask = valid_mask.at[:, :, valid_tokens].set(0)
        eop_logit = eop_logit + valid_mask
        if temprature > 0.0:
            token = jax.random.categorical(rng, eop_logit / temprature, axis=-1)
        else:
            token = jnp.argmax(eop_logit, axis=-1)
        has_boa = jnp.any(token == _tokenizer.BEGIN_OF_ACTION, axis=1)

        return observation, kv_cache, token, eop_logit, prefix_mask, prefix_positions, has_boa

    @at.typecheck
    def reason(
        self,
        rng: at.KeyArrayLike,
        last_logit: at.Float[at.Array, "b 1 v"],
        prefix_kv_cache: _gemma.KVCache,
        prefix_mask: at.Bool[at.Array, "b p"],
        prefix_positions: at.Int[at.Array, "b p"],
        *,
        temprature: float = 0.,
        max_decoding_steps: int = 256,
    ) -> at.Int[at.Array, "b _s"]:
        """Output reasoning tokens after `prefill`.

        Args:
            rng: PRNG key.
            last_logit: logit for the last token of the prefix.
            prefix_kv_cache: KV cache for the prefix.
            prefix_mask: input mask for the prefix.
            prefix_positions: position id for the prefix.
            temprature: decoding temprature.
            max_decoding_steps: maximum decoding steps.
        """
        step_rng = jax.random.fold_in(rng, 0)
        if temprature > 0.0:
            token = jax.random.categorical(step_rng, last_logit / temprature, axis=-1)
        else:
            token = jnp.argmax(last_logit, axis=-1)
        has_eos = jnp.any(token == PALIGEMMA_EOS_TOKEN, axis=1)
        all_eos = jnp.all(has_eos)
        output_tokens = jnp.zeros((last_logit.shape[0], max_decoding_steps), dtype=token.dtype)

        # Pad KV cache so its shape remains fixed throughout the while loop (shape: l, b, t, k, h)
        kv_cache = jax.tree.map(
            lambda x: jnp.pad(x, ((0, 0), (0, 0), (0, max_decoding_steps), (0, 0), (0, 0))),
            prefix_kv_cache,
        )
        # attn_mask shape: (b, 1, prefix_len + 1 + max_decoding_steps)
        prefix_attn_mask = einops.repeat(prefix_mask, "b p -> b s p", s=1)
        attn_mask = jnp.pad(prefix_attn_mask, ((0, 0), (0, 0), (0, max_decoding_steps + 1)))
        attn_mask = attn_mask.at[:, :, -1].set(True)

        @at.typecheck
        def _wrap_cache(
            cache_appended: at.Float[at.Array, "l b t k h"],
            step: at.Int[at.Array, ""],
        ) -> at.Float[at.Array, "l b t-1 k h"]:
            new_value = cache_appended[:, :, -1]
            cache = cache_appended[:, :, :-1]
            cache = jax.lax.dynamic_update_index_in_dim(
                cache, new_value, prefix_mask.shape[1] + 1 + step, axis=2
            )
            return cache

        def decode_step(carry):
            last_logit, output_tokens, kv_cache, attn_mask, _, step = carry
            step_rng = jax.random.fold_in(rng, step)

            if temprature > 0.0:
                token = jax.random.categorical(step_rng, last_logit / temprature, axis=-1)
            else:
                token = jnp.argmax(last_logit, axis=-1)
            token = jnp.where(
                step == 0,
                jnp.full_like(token, _tokenizer.BEGIN_OF_REASONING),
                token,
            )
            output_tokens = put_along_last_axis(
                output_tokens, jnp.broadcast_to(step, (token.shape[0], 1)), token
            )

            has_eos = jnp.any(token == PALIGEMMA_EOS_TOKEN, axis=1)
            all_eos = jnp.all(has_eos)

            token_embedding = self.PaliGemma.llm(token, method="embed")
            positions = prefix_positions[:, [-1]] + step + 1
            (last_pre_logit, _), kv_cache_appended = self.PaliGemma.llm(
                [token_embedding, None],
                mask=attn_mask,
                positions=positions,
                kv_cache=kv_cache,
                adarms_cond=[None, None],
                cond=None,
            )

            last_logit = self.PaliGemma.llm(last_pre_logit, method="embedder_decode")
            kv_cache = jax.tree.map(lambda x: _wrap_cache(x, step), kv_cache_appended)
            attn_mask = attn_mask.at[:, :, prefix_mask.shape[1] + 1 + step].set(True)
            return last_logit, output_tokens, kv_cache, attn_mask, all_eos, step + 1

        def decode_cond(carry):
            _, _, _, _, all_eos, step = carry
            return (~all_eos) & (step < max_decoding_steps)

        _, suffix_txt_tokens, kv_cache, _, _, _ = jax.lax.while_loop(
            decode_cond, decode_step,
            (last_logit, output_tokens, kv_cache, attn_mask, all_eos, 0),
        )

        return suffix_txt_tokens

    @at.typecheck
    def act(
        self,
        rng: at.KeyArrayLike,
        observation: _model.AtomicObservation,
        prefix_cache: _gemma.KVCache,
        prefix_mask: at.Bool[at.Array, "b p"],
        prefix_positions: at.Int[at.Array, "b p"],
        *,
        num_steps: int | at.Int[at.Array, ""] = 10,
    ) -> at.Float[at.Array, "b ah ad"]:
        """Sample action after `prefill`.

        Args:
            rng: PRNG key.
            observation: AtomicObservation.
            prefix_cache: KV cache for the img + txt prefix.
            prefix_mask: input mask for the img + txt prefix.
            prefix_positions: position id for the img + txt prefix.
            num_steps: number of action denoising steps.

        Returns:
            Denoised actions of shape (b, ah, ad).
        """
        boa_token = jnp.broadcast_to(_tokenizer.BEGIN_OF_ACTION, (prefix_mask.shape[0], 1))
        prefix_attn_mask = einops.repeat(prefix_mask, "b p -> b s p", s=1)
        boa_attn_mask = jnp.concatenate(
            [prefix_attn_mask, jnp.ones((prefix_attn_mask.shape[0], 1, 1), dtype=jnp.bool_)],
            axis=-1,
        )
        boa_positions = prefix_positions[:, [-1]] + 1
        boa_token_embedding = self.PaliGemma.llm(boa_token, method="embed")
        (boa_pre_logit, _), img_txt_kv_cache = self.PaliGemma.llm(
            [boa_token_embedding, None],
            mask=boa_attn_mask,
            positions=boa_positions,
            kv_cache=prefix_cache,
            adarms_cond=[None, None],
            cond=None,
        )
        img_txt_mask = jnp.pad(prefix_mask, ((0, 0), (0, 1)), constant_values=1)

        dt = -1.0 / num_steps
        batch_size = observation.state.shape[0]
        noise = jax.random.normal(rng, (batch_size, self.action_horizon, self.action_dim))
        atomic_cond_embed = self.embed_atomic_skill(observation)

        def denoise_step(carry):
            x_t, time = carry
            suffix_tokens, suffix_mask, suffix_ar_mask, adarms_cond = self.embed_suffix(
                observation, x_t, jnp.broadcast_to(time, batch_size)
            )
            suffix_attn_mask = make_attn_mask(suffix_mask, suffix_ar_mask)
            prefix_attn_mask = einops.repeat(img_txt_mask, "b p -> b s p", s=suffix_tokens.shape[1])
            full_attn_mask = jnp.concatenate([prefix_attn_mask, suffix_attn_mask], axis=-1)
            positions = jnp.sum(img_txt_mask, axis=-1)[:, None] + jnp.cumsum(suffix_mask, axis=-1) - 1

            (prefix_out, suffix_out), _ = self.PaliGemma.llm(
                [None, suffix_tokens],
                mask=full_attn_mask,
                positions=positions,
                kv_cache=img_txt_kv_cache,
                adarms_cond=[None, adarms_cond],
                cond=atomic_cond_embed,
            )
            assert prefix_out is None
            v_t = self.action_out_proj(suffix_out[:, -self.action_horizon:])

            return x_t + dt * v_t, time + dt

        def denoise_cond(carry):
            x_t, time = carry
            return time >= -dt / 2

        x_0, _ = jax.lax.while_loop(denoise_cond, denoise_step, (noise, 1.0))
        return x_0

    @override
    def sample_actions(
        self,
        rng: at.KeyArrayLike,
        observation: _model.Observation,
        *,
        num_steps: int | at.Int[at.Array, ""] = 10,
        noise: at.Float[at.Array, "b ah ad"] | None = None,
    ) -> _model.Actions:
        observation = _model.preprocess_observation(None, observation, train=False)
        # note that we use the convention more common in diffusion literature, where t=1 is noise and t=0 is the target
        # distribution. yes, this is the opposite of the pi0 paper, and I'm sorry.
        dt = -1.0 / num_steps
        batch_size = observation.state.shape[0]
        if noise is None:
            noise = jax.random.normal(rng, (batch_size, self.action_horizon, self.action_dim))

        # first fill KV cache with a forward pass of the prefix
        prefix_tokens, prefix_mask, prefix_ar_mask = self.embed_prefix(observation)
        prefix_attn_mask = make_attn_mask(prefix_mask, prefix_ar_mask)
        positions = jnp.cumsum(prefix_mask, axis=1) - 1
        _, kv_cache = self.PaliGemma.llm([prefix_tokens, None], mask=prefix_attn_mask, positions=positions)

        def step(carry):
            x_t, time = carry
            suffix_tokens, suffix_mask, suffix_ar_mask, adarms_cond = self.embed_suffix(
                observation, x_t, jnp.broadcast_to(time, batch_size)
            )
            # `suffix_attn_mask` is shape (b, suffix_len, suffix_len) indicating how the suffix tokens can attend to each
            # other
            suffix_attn_mask = make_attn_mask(suffix_mask, suffix_ar_mask)
            # `prefix_attn_mask` is shape (b, suffix_len, prefix_len) indicating how the suffix tokens can attend to the
            # prefix tokens
            prefix_attn_mask = einops.repeat(prefix_mask, "b p -> b s p", s=suffix_tokens.shape[1])
            # `combined_mask` is shape (b, suffix_len, prefix_len + suffix_len) indicating how the suffix tokens (which
            # generate the queries) can attend to the full prefix + suffix sequence (which generates the keys and values)
            full_attn_mask = jnp.concatenate([prefix_attn_mask, suffix_attn_mask], axis=-1)
            assert full_attn_mask.shape == (
                batch_size,
                suffix_tokens.shape[1],
                prefix_tokens.shape[1] + suffix_tokens.shape[1],
            )
            # `positions` is shape (b, suffix_len) indicating the positions of the suffix tokens
            positions = jnp.sum(prefix_mask, axis=-1)[:, None] + jnp.cumsum(suffix_mask, axis=-1) - 1

            (prefix_out, suffix_out), _ = self.PaliGemma.llm(
                [None, suffix_tokens],
                mask=full_attn_mask,
                positions=positions,
                kv_cache=kv_cache,
                adarms_cond=[None, adarms_cond],
            )
            assert prefix_out is None
            v_t = self.action_out_proj(suffix_out[:, -self.action_horizon :])

            return x_t + dt * v_t, time + dt

        def cond(carry):
            x_t, time = carry
            # robust to floating-point error
            return time >= -dt / 2

        x_0, _ = jax.lax.while_loop(cond, step, (noise, 1.0))
        return x_0