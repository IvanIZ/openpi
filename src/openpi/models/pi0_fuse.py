"""Pi0Fuse model: pi05-compatible architecture with joint textual reasoning and action loss.

Combines pi05's architecture (AdaRMSNorm, no state_proj) for weight compatibility
with "Do What You Say" paper's reasoning loss approach (text CE + action diffusion).
"""

import dataclasses
import logging

import einops
import flax.nnx as nnx
import flax.nnx.bridge as nnx_bridge
import jax
import jax.numpy as jnp
from typing_extensions import override

from openpi.models import model as _model
import openpi.models.gemma as _gemma
import openpi.models.siglip as _siglip
import openpi.models.tokenizer as _tokenizer
from openpi.shared import array_typing as at
import openpi.shared.nnx_utils as nnx_utils

logger = logging.getLogger("openpi")

PALIGEMMA_EOS_TOKEN = 1


def make_attn_mask(input_mask, mask_ar):
    """Adapted from big_vision. See pi0.py for full docstring."""
    mask_ar = jnp.broadcast_to(mask_ar, input_mask.shape)
    cumsum = jnp.cumsum(mask_ar, axis=1)
    attn_mask = cumsum[:, None, :] <= cumsum[:, :, None]
    valid_mask = input_mask[:, None, :] * input_mask[:, :, None]
    return jnp.logical_and(attn_mask, valid_mask)


@at.typecheck
def posemb_sincos(
    pos: at.Real[at.Array, " b"], embedding_dim: int, min_period: float, max_period: float
) -> at.Float[at.Array, "b {embedding_dim}"]:
    if embedding_dim % 2 != 0:
        raise ValueError(f"embedding_dim ({embedding_dim}) must be divisible by 2")
    fraction = jnp.linspace(0.0, 1.0, embedding_dim // 2)
    period = min_period * (max_period / min_period) ** fraction
    sinusoid_input = jnp.einsum(
        "i,j->ij", pos, 1.0 / period * 2 * jnp.pi,
        precision=jax.lax.Precision.HIGHEST,
    )
    return jnp.concatenate([jnp.sin(sinusoid_input), jnp.cos(sinusoid_input)], axis=-1)


def put_along_last_axis(arr, indices, values):
    assert arr.ndim == indices.ndim == values.ndim, (arr.ndim, indices.ndim, values.ndim)
    onehot = jax.nn.one_hot(indices, arr.shape[-1], dtype=values.dtype)
    put_mask = jnp.einsum("...i,...in->...n", jnp.ones(values.shape, jnp.int32), onehot)
    put_values = jnp.einsum("...i,...in->...n", values, onehot)
    return jnp.where(put_mask, put_values, arr)


@dataclasses.dataclass(frozen=True)
class Pi0FuseConfig(_model.BaseModelConfig):
    """Config for Pi0Fuse model (pi05-compatible with reasoning loss)."""
    dtype: str = "bfloat16"
    paligemma_variant: _gemma.Variant = "gemma_2b"
    action_expert_variant: _gemma.Variant = "gemma_300m"

    action_dim: int = 32
    action_horizon: int = 16
    max_token_len: int = 415

    # L = L_text + diffusion_loss_coeff * L_diffusion
    diffusion_loss_coeff: float = 1.0

    # Pi05 uses AdaRMSNorm in action expert and discretized state in text
    pi05: bool = True

    @property
    @override
    def model_type(self) -> _model.ModelType:
        return _model.ModelType.PI0_FUSE

    @override
    def create(self, rng: at.KeyArrayLike) -> "Pi0Fuse":
        return Pi0Fuse(self, rngs=nnx.Rngs(rng))

    @override
    def inputs_spec(self, *, batch_size: int = 1) -> tuple[_model.FuseObservation, _model.Actions]:
        image_spec = jax.ShapeDtypeStruct([batch_size, *_model.IMAGE_RESOLUTION, 3], jnp.float32)
        image_mask_spec = jax.ShapeDtypeStruct([batch_size], jnp.bool_)

        with at.disable_typechecking():
            observation_spec = _model.FuseObservation(
                images={
                    "base_0_rgb": image_spec,
                    "left_wrist_0_rgb": image_spec,
                    "right_wrist_0_rgb": image_spec,
                },
                image_masks={
                    "base_0_rgb": image_mask_spec,
                    "left_wrist_0_rgb": image_mask_spec,
                    "right_wrist_0_rgb": image_mask_spec,
                },
                state=jax.ShapeDtypeStruct([batch_size, self.action_dim], jnp.float32),
                tokenized_prompt=jax.ShapeDtypeStruct([batch_size, self.max_token_len], jnp.int32),
                tokenized_prompt_mask=jax.ShapeDtypeStruct([batch_size, self.max_token_len], jnp.bool_),
                token_ar_mask=jax.ShapeDtypeStruct([batch_size, self.max_token_len], jnp.int32),
                token_loss_mask=jax.ShapeDtypeStruct([batch_size, self.max_token_len], jnp.bool_),
                diffusion_loss_mask=jax.ShapeDtypeStruct([batch_size], jnp.bool_),
            )
        action_spec = jax.ShapeDtypeStruct([batch_size, self.action_horizon, self.action_dim], jnp.float32)
        return observation_spec, action_spec

    def get_freeze_filter(self) -> nnx.filterlib.Filter:
        filters = []
        has_lora = False
        gemma_params_filter = nnx_utils.PathRegex(".*llm.*")
        action_expert_params_filter = nnx_utils.PathRegex(".*llm.*_1.*")
        if "lora" in self.paligemma_variant:
            filters.append(gemma_params_filter)
            if "lora" not in self.action_expert_variant:
                filters.append(nnx.Not(action_expert_params_filter))
            has_lora = True
        elif "lora" in self.action_expert_variant:
            filters.append(action_expert_params_filter)
            has_lora = True

        if has_lora:
            filters.append(nnx.Not(nnx_utils.PathRegex(".*lora.*")))
        if not filters:
            return nnx.Nothing
        return nnx.All(*filters)


class Pi0Fuse(_model.BaseModel):
    """Pi05-compatible model with joint text reasoning loss and action diffusion loss.

    Architecture matches pi05 (AdaRMSNorm, no state_proj) for loading pi05 base weights.
    Adds text cross-entropy loss on reasoning tokens alongside flow-matching action loss.
    """

    def __init__(self, config: Pi0FuseConfig, rngs: nnx.Rngs):
        super().__init__(config.action_dim, config.action_horizon, config.max_token_len)
        self.pi05 = config.pi05
        paligemma_config = _gemma.get_config(config.paligemma_variant)
        action_expert_config = _gemma.get_config(config.action_expert_variant)

        llm = nnx_bridge.ToNNX(
            _gemma.Module(
                configs=[paligemma_config, action_expert_config],
                embed_dtype=config.dtype,
                adarms=config.pi05,
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
        self.diffusion_loss_coeff = config.diffusion_loss_coeff
        self.deterministic = True

    @at.typecheck
    def embed_img_txt(
        self, obs: _model.FuseObservation
    ) -> tuple[at.Float[at.Array, "b s emb"], at.Bool[at.Array, "b s"], at.Int[at.Array, "b s"]]:
        """Embed images and tokenized prompt (text prefix + text suffix)."""
        input_mask = []
        ar_mask = []
        embeddings = []

        for name in obs.images:
            image_emb, _ = self.PaliGemma.img(obs.images[name], train=False)
            embeddings.append(image_emb)
            input_mask.append(
                einops.repeat(obs.image_masks[name], "b -> b s", s=image_emb.shape[1])
            )
            ar_mask.append(0 * input_mask[-1])

        assert obs.tokenized_prompt is not None
        assert obs.tokenized_prompt_mask is not None
        assert obs.token_ar_mask is not None

        txt_emb = self.PaliGemma.llm(obs.tokenized_prompt, method="embed")
        embeddings.append(txt_emb)
        input_mask.append(obs.tokenized_prompt_mask)
        ar_mask.append(obs.token_ar_mask)

        embeddings = jnp.concatenate(embeddings, axis=1)
        input_mask = jnp.concatenate(input_mask, axis=1)
        ar_mask = jnp.concatenate(ar_mask, axis=1)
        return embeddings, input_mask, ar_mask

    @at.typecheck
    def embed_action_suffix(
        self, obs: _model.FuseObservation, noisy_actions: _model.Actions, timestep: at.Float[at.Array, " b"]
    ) -> tuple[
        at.Float[at.Array, "b s emb"],
        at.Bool[at.Array, "b s"],
        at.Bool[at.Array, " s"],
        at.Float[at.Array, "b emb"] | None,
    ]:
        """Embed action tokens for the action expert. Pi05-style: AdaRMSNorm, no state token."""
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
        return tokens, input_mask, ar_mask, adarms_cond

    @override
    def compute_loss(
        self, rng: at.KeyArrayLike, observation: _model.FuseObservation, actions: _model.Actions, *, train: bool = False
    ) -> at.Float[at.Array, "*b"]:
        preprocess_rng, noise_rng, time_rng = jax.random.split(rng, 3)
        observation = _model.preprocess_observation(
            preprocess_rng, observation, train=train,
            image_keys=list(observation.images.keys())
        )

        batch_shape = actions.shape[:-2]
        noise = jax.random.normal(noise_rng, actions.shape)
        time = jax.random.beta(time_rng, 1.5, 1, batch_shape) * 0.999 + 0.001
        time_expanded = time[..., None, None]
        x_t = time_expanded * noise + (1 - time_expanded) * actions
        u_t = noise - actions

        img_txt_tokens, img_txt_mask, img_txt_ar_mask = self.embed_img_txt(observation)
        action_tokens, action_mask, action_ar_mask, adarms_cond = self.embed_action_suffix(observation, x_t, time)

        input_mask = jnp.concatenate([img_txt_mask, action_mask], axis=1)
        ar_mask = jnp.concatenate([img_txt_ar_mask, jnp.broadcast_to(action_ar_mask, (input_mask.shape[0], action_mask.shape[1]))], axis=1)
        attn_mask = make_attn_mask(input_mask, ar_mask)
        positions = jnp.cumsum(input_mask, axis=1) - 1

        (img_txt_pre_logits, action_out), _ = self.PaliGemma.llm(
            [img_txt_tokens, action_tokens], mask=attn_mask, positions=positions,
            adarms_cond=[None, adarms_cond],
        )

        # --- Text CE loss (reasoning loss) ---
        txt_targets = jax.nn.one_hot(
            observation.tokenized_prompt[:, 1:],
            _gemma.PALIGEMMA_VOCAB_SIZE,
        )
        txt_logits = self.PaliGemma.llm(
            img_txt_pre_logits[:, -1 - txt_targets.shape[1]: -1],
            method="deembed",
        )
        txt_logp = jax.nn.log_softmax(txt_logits, axis=-1)
        txt_loss_mask = observation.token_loss_mask[:, 1:]
        txt_token_pplx = jnp.sum(txt_targets * txt_logp, axis=-1)
        txt_loss = (
            -jnp.sum(txt_token_pplx * txt_loss_mask, axis=-1) /
            jnp.clip(jnp.sum(txt_loss_mask, axis=-1), 1)
        )

        # --- Action diffusion loss (flow matching) ---
        # we only sample when diffusion_loss_mask is False to contribute text loss
        v_t = self.action_out_proj(action_out[:, -self.action_horizon:])
        action_loss = jnp.mean(
            jnp.square(v_t - u_t) * observation.diffusion_loss_mask[:, None, None],
            axis=(-2, -1),
        )

        loss = txt_loss + self.diffusion_loss_coeff * action_loss
        return loss

    @override
    def sample_actions(
        self,
        rng: at.KeyArrayLike,
        observation: _model.FuseObservation,
        *,
        num_steps: int | at.Int[at.Array, ""] = 10,
    ) -> tuple[_model.Actions, dict[str, at.Array]]:
        observation = _model.preprocess_observation(
            None, observation, train=False,
            image_keys=list(observation.images.keys())
        )

        info = {}
        info['num_action_loss_fraction'] = (
            jnp.sum(observation.diffusion_loss_mask) /
            observation.diffusion_loss_mask.shape[0]
        )

        img_txt_tokens, img_txt_mask, img_txt_ar_mask = self.embed_img_txt(observation)
        img_txt_attn_mask = make_attn_mask(img_txt_mask, img_txt_ar_mask)
        positions = jnp.cumsum(img_txt_mask, axis=1) - 1

        (img_txt_pre_logits, _), kv_cache = self.PaliGemma.llm(
            [img_txt_tokens, None], mask=img_txt_attn_mask, positions=positions,
            adarms_cond=[None, None],
        )

        dt = -1.0 / num_steps
        batch_size = observation.state.shape[0]
        noise = jax.random.normal(rng, (batch_size, self.action_horizon, self.action_dim))

        def step(carry):
            x_t, t = carry
            suffix_tokens, suffix_mask, suffix_ar_mask, adarms_cond = self.embed_action_suffix(
                observation, x_t, jnp.broadcast_to(t, batch_size)
            )
            suffix_attn_mask = make_attn_mask(suffix_mask, suffix_ar_mask)
            prefix_attn_mask = einops.repeat(img_txt_mask, "b p -> b s p", s=suffix_tokens.shape[1])
            full_attn_mask = jnp.concatenate([prefix_attn_mask, suffix_attn_mask], axis=-1)
            pos = jnp.sum(img_txt_mask, axis=-1)[:, None] + jnp.cumsum(suffix_mask, axis=-1) - 1

            (_, suffix_out), _ = self.PaliGemma.llm(
                [None, suffix_tokens], mask=full_attn_mask, positions=pos,
                kv_cache=kv_cache, adarms_cond=[None, adarms_cond],
            )
            v_t = self.action_out_proj(suffix_out[:, -self.action_horizon:])
            return x_t + dt * v_t, t + dt

        def cond(carry):
            _, t = carry
            return t >= -dt / 2

        x_0, _ = jax.lax.while_loop(cond, step, (noise, 1.0))
        return x_0, info

    @at.typecheck
    def prefill(
        self,
        rng: at.KeyArrayLike,
        observation: _model.FuseObservation,
        *,
        temperature: float = 0.,
    ):
        """Prefill the prefix for inference. Returns KV cache and decision to think or act."""
        observation = _model.preprocess_observation(
            None, observation, train=False,
            image_keys=list(observation.images.keys())
        )

        first_one_indices = jnp.argmax(observation.token_ar_mask, axis=-1)
        padding_mask = jnp.arange(observation.token_ar_mask.shape[-1]) >= first_one_indices[..., jnp.newaxis]
        observation = dataclasses.replace(
            observation,
            tokenized_prompt=jnp.where(padding_mask, 0, observation.tokenized_prompt),
            tokenized_prompt_mask=jnp.logical_not(padding_mask),
        )

        prefix_token_embeddings, prefix_mask, prefix_ar_mask = self.embed_img_txt(observation)
        prefix_attn_mask = make_attn_mask(prefix_mask, prefix_ar_mask)
        prefix_positions = jnp.cumsum(prefix_mask, axis=1) - 1

        (pre_logit, _), kv_cache = self.PaliGemma.llm(
            [prefix_token_embeddings, None], mask=prefix_attn_mask, positions=prefix_positions,
            adarms_cond=[None, None],
        )

        eop_indices = prefix_positions[:, -1]
        eop_pre_logit = jnp.take_along_axis(pre_logit, eop_indices[:, None, None], axis=1)
        eop_logit = self.PaliGemma.llm(eop_pre_logit, method="deembed")

        valid_tokens = jnp.array([_tokenizer.BEGIN_OF_ACTION, _tokenizer.BEGIN_OF_REASONING])
        valid_mask = jnp.full((1, 1, eop_logit.shape[-1]), -jnp.inf)
        valid_mask = valid_mask.at[:, :, valid_tokens].set(0)
        eop_logit = eop_logit + valid_mask

        if temperature > 0.0:
            token = jax.random.categorical(rng, eop_logit / temperature, axis=-1)
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
        temperature: float = 0.,
        max_decoding_steps: int = 256,
    ) -> at.Int[at.Array, "b _s"]:
        """Autoregressive reasoning token generation.

        Uses the LLM's built-in _update_cache for KV cache management (matching
        the approach in pi0.py's sample_low_level_task).
        """
        batch_size = prefix_mask.shape[0]
        output_tokens = jnp.zeros((batch_size, max_decoding_steps), dtype=jnp.int32)
        # Do not sample control tokens after step 0; they are protocol markers, not reasoning text.
        blocked_tokens = jnp.array(
            [_tokenizer.BEGIN_OF_ACTION, _tokenizer.BEGIN_OF_REASONING, _tokenizer.END_OF_PREFIX_TOKEN]
        )
        blocked_logits = jnp.zeros((1, 1, _gemma.PALIGEMMA_VOCAB_SIZE), dtype=last_logit.dtype)
        blocked_logits = blocked_logits.at[:, :, blocked_tokens].set(-jnp.inf)

        idx, k_cache, v_cache = prefix_kv_cache
        k_cache = jnp.pad(k_cache, ((0, 0), (0, 0), (0, max_decoding_steps), (0, 0), (0, 0)))
        v_cache = jnp.pad(v_cache, ((0, 0), (0, 0), (0, max_decoding_steps), (0, 0), (0, 0)))
        kv_cache = (idx, k_cache, v_cache)

        def decode_step(carry):
            rng, last_logit, output_tokens, kv_cache, all_eos, step = carry
            step_rng = jax.random.fold_in(rng, step)
            sample_logit = jnp.where(step == 0, last_logit, last_logit + blocked_logits)

            if temperature > 0.0:
                token = jax.random.categorical(step_rng, sample_logit / temperature, axis=-1)
            else:
                token = jnp.argmax(sample_logit, axis=-1)

            token = jnp.where(
                step == 0,
                jnp.full_like(token, _tokenizer.BEGIN_OF_REASONING),
                token,
            )
            output_tokens = put_along_last_axis(
                output_tokens, jnp.broadcast_to(step, (batch_size, 1)), token
            )

            has_eos = jnp.any(token == PALIGEMMA_EOS_TOKEN, axis=1)
            all_eos = jnp.all(has_eos)

            token_embedding = self.PaliGemma.llm(token, method="embed")
            positions = prefix_positions[:, [-1]] + step + 1

            decode_visible = jnp.arange(max_decoding_steps) <= step
            full_mask = jnp.concatenate([prefix_mask, jnp.broadcast_to(decode_visible, (batch_size, max_decoding_steps))], axis=-1)
            mask = full_mask[:, None, :]

            (last_pre_logit, _), kv_cache = self.PaliGemma.llm(
                [token_embedding, None], mask=mask, positions=positions,
                kv_cache=kv_cache, adarms_cond=[None, None],
            )
            last_logit = self.PaliGemma.llm(last_pre_logit, method="deembed")

            return rng, last_logit, output_tokens, kv_cache, all_eos, step + 1

        def decode_cond(carry):
            _, _, _, _, all_eos, step = carry
            return (~all_eos) & (step < max_decoding_steps)

        _, _, output_tokens, _, _, _ = jax.lax.while_loop(
            decode_cond, decode_step,
            (rng, last_logit, output_tokens, kv_cache, False, 0),
        )
        return output_tokens

    @at.typecheck
    def act(
        self,
        rng: at.KeyArrayLike,
        observation: _model.FuseObservation,
        prefix_cache: _gemma.KVCache,
        prefix_mask: at.Bool[at.Array, "b p"],
        prefix_positions: at.Int[at.Array, "b p"],
        *,
        num_steps: int | at.Int[at.Array, ""] = 10,
    ) -> at.Float[at.Array, "b ah ad"]:
        """Sample actions after prefill, using diffusion denoising."""
        # Pad prefix KV cache by 1 to make room for the BOA token.
        # _init_cache sized the cache exactly to prefix_len, so there is no
        # spare slot for the additional token that act() needs to insert.
        idx, k_cache, v_cache = prefix_cache
        k_cache = jnp.pad(k_cache, ((0, 0), (0, 0), (0, 1), (0, 0), (0, 0)))
        v_cache = jnp.pad(v_cache, ((0, 0), (0, 0), (0, 1), (0, 0), (0, 0)))
        prefix_cache = (idx, k_cache, v_cache)

        boa_token = jnp.broadcast_to(
            _tokenizer.BEGIN_OF_ACTION, (prefix_mask.shape[0], 1)
        )
        prefix_attn_mask = einops.repeat(prefix_mask, "b p -> b s p", s=1)
        boa_attn_mask = jnp.concatenate(
            [prefix_attn_mask, jnp.ones((prefix_attn_mask.shape[0], 1, 1), dtype=jnp.bool_)],
            axis=-1,
        )
        boa_positions = prefix_positions[:, [-1]] + 1
        boa_token_embedding = self.PaliGemma.llm(boa_token, method="embed")
        (_, _), img_txt_kv_cache = self.PaliGemma.llm(
            [boa_token_embedding, None], mask=boa_attn_mask, positions=boa_positions,
            kv_cache=prefix_cache, adarms_cond=[None, None],
        )
        img_txt_mask = jnp.pad(prefix_mask, ((0, 0), (0, 1)), constant_values=1)

        dt = -1.0 / num_steps
        batch_size = observation.state.shape[0]
        noise = jax.random.normal(rng, (batch_size, self.action_horizon, self.action_dim))

        def denoise_step(carry):
            x_t, t = carry
            suffix_tokens, suffix_mask, suffix_ar_mask, adarms_cond = self.embed_action_suffix(
                observation, x_t, jnp.broadcast_to(t, batch_size)
            )
            suffix_attn_mask = make_attn_mask(suffix_mask, suffix_ar_mask)
            prefix_attn_mask = einops.repeat(img_txt_mask, "b p -> b s p", s=suffix_tokens.shape[1])
            full_attn_mask = jnp.concatenate([prefix_attn_mask, suffix_attn_mask], axis=-1)
            pos = jnp.sum(img_txt_mask, axis=-1)[:, None] + jnp.cumsum(suffix_mask, axis=-1) - 1

            (_, suffix_out), _ = self.PaliGemma.llm(
                [None, suffix_tokens], mask=full_attn_mask, positions=pos,
                kv_cache=img_txt_kv_cache, adarms_cond=[None, adarms_cond],
            )
            v_t = self.action_out_proj(suffix_out[:, -self.action_horizon:])
            return x_t + dt * v_t, t + dt

        def denoise_cond(carry):
            _, t = carry
            return t >= -dt / 2

        x_0, _ = jax.lax.while_loop(denoise_cond, denoise_step, (noise, 1.0))
        return x_0
