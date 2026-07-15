from collections.abc import Sequence
import dataclasses
from typing import Literal

import flax.nnx as nnx
import flax.struct as struct
import jax.numpy as jnp

from openpi.shared import array_typing as at


@dataclasses.dataclass(frozen=True)
class OverviewActionConditioningConfig:
    enabled: bool = False
    rank: int = 16
    lora_alpha: float = 16.0
    target: Literal["q", "o", "q_o"] = "q_o"
    use_state_context: bool = True

    def __post_init__(self):
        if self.rank <= 0:
            raise ValueError(f"rank must be positive, got {self.rank}")
        if self.lora_alpha <= 0:
            raise ValueError(f"lora_alpha must be positive, got {self.lora_alpha}")
        if self.target not in ("q", "o", "q_o"):
            raise ValueError(f"target must be one of q/o/q_o, got {self.target!r}")


@struct.dataclass
class PrefixEmbeddings:
    tokens: at.Float[at.Array, "b s emb"]
    input_mask: at.Bool[at.Array, "b s"]
    ar_mask: at.Bool[at.Array, " s"]
    image_lengths: tuple[int, ...] = struct.field(pytree_node=False)
    language_length: int = struct.field(pytree_node=False)


def _masked_mean(values: at.Array, mask: at.Array) -> at.Array:
    mask = mask.astype(values.dtype)
    denominator = jnp.maximum(jnp.sum(mask, axis=1, keepdims=True), 1)
    return jnp.sum(values * mask[..., None], axis=1) / denominator


class OverviewContextEncoder(nnx.Module):
    def __init__(
        self,
        *,
        vlm_width: int,
        action_width: int,
        num_views: int,
        dtype: str,
        rngs: nnx.Rngs,
    ):
        self.vlm_width = vlm_width
        self.num_views = num_views
        self.source_norm = nnx.LayerNorm(vlm_width, use_bias=False, dtype=dtype, rngs=rngs)
        input_width = (num_views + 1) * vlm_width + num_views
        self.mlp_in = nnx.Linear(input_width, action_width, dtype=dtype, rngs=rngs)
        self.mlp_out = nnx.Linear(action_width, action_width, dtype=dtype, rngs=rngs)

    def __call__(
        self,
        prefix_out: at.Array,
        prefix: PrefixEmbeddings,
        image_masks: Sequence[at.Array],
        language_mask: at.Array | None,
    ) -> at.Array:
        if len(prefix.image_lengths) != self.num_views or len(image_masks) != self.num_views:
            raise ValueError(
                f"Expected {self.num_views} image segments and masks, got "
                f"{len(prefix.image_lengths)} segments and {len(image_masks)} masks"
            )

        pooled_sources = []
        offset = 0
        for length, image_mask in zip(prefix.image_lengths, image_masks, strict=True):
            segment = prefix_out[:, offset : offset + length]
            token_mask = jnp.broadcast_to(image_mask[:, None], segment.shape[:2])
            pooled_sources.append(self.source_norm(_masked_mean(segment, token_mask)))
            offset += length

        if prefix.language_length:
            if language_mask is None:
                raise ValueError("language_mask is required when language_length is non-zero")
            language_end = offset + prefix.language_length
            language_context = _masked_mean(prefix_out[:, offset:language_end], language_mask)
            offset = language_end
        else:
            language_context = jnp.zeros((prefix_out.shape[0], self.vlm_width), dtype=prefix_out.dtype)
        pooled_sources.append(self.source_norm(language_context))

        if offset != prefix_out.shape[1]:
            raise ValueError(f"Prefix layout covers {offset} tokens, but prefix_out has {prefix_out.shape[1]}")

        view_validity = jnp.stack(image_masks, axis=-1).astype(prefix_out.dtype)
        overview_input = jnp.concatenate([*pooled_sources, view_validity], axis=-1)
        context = self.mlp_in(overview_input)
        context = nnx.swish(context)
        return self.mlp_out(context)


class StateContextEncoder(nnx.Module):
    def __init__(self, *, state_dim: int, action_width: int, dtype: str, rngs: nnx.Rngs):
        self.mlp_in = nnx.Linear(state_dim, action_width, dtype=dtype, rngs=rngs)
        self.mlp_out = nnx.Linear(action_width, action_width, dtype=dtype, rngs=rngs)

    def __call__(self, state: at.Array) -> at.Array:
        context = self.mlp_in(state)
        context = nnx.swish(context)
        return self.mlp_out(context)


class ActionConditioningController(nnx.Module):
    def __init__(
        self,
        *,
        action_width: int,
        num_layers: int,
        num_heads: int,
        dtype: str,
        rngs: nnx.Rngs,
    ):
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.scene_norm = nnx.LayerNorm(action_width, dtype=dtype, rngs=rngs)
        self.state_norm = nnx.LayerNorm(action_width, dtype=dtype, rngs=rngs)
        self.hidden_proj = nnx.Linear(2 * action_width, action_width, dtype=dtype, rngs=rngs)
        self.gate_proj = nnx.Linear(
            action_width,
            num_layers * 2 * num_heads,
            dtype=dtype,
            kernel_init=nnx.initializers.zeros_init(),
            bias_init=nnx.initializers.zeros_init(),
            rngs=rngs,
        )

    def __call__(self, scene_context: at.Array, state_context: at.Array) -> tuple[at.Array, at.Array]:
        controller_input = jnp.concatenate([self.scene_norm(scene_context), self.state_norm(state_context)], axis=-1)
        hidden = nnx.swish(self.hidden_proj(controller_input))
        raw_gates = self.gate_proj(hidden).reshape(scene_context.shape[0], self.num_layers, 2, self.num_heads)
        return jnp.tanh(raw_gates[:, :, 0]), jnp.tanh(raw_gates[:, :, 1])


class OverviewActionConditioning(nnx.Module):
    def __init__(
        self,
        config: OverviewActionConditioningConfig,
        *,
        vlm_width: int,
        action_width: int,
        state_dim: int,
        num_views: int,
        num_layers: int,
        num_heads: int,
        dtype: str,
        rngs: nnx.Rngs,
    ):
        self.config = config
        self.action_width = action_width
        self.dtype = dtype
        self.overview_encoder = OverviewContextEncoder(
            vlm_width=vlm_width,
            action_width=action_width,
            num_views=num_views,
            dtype=dtype,
            rngs=rngs,
        )
        self.state_encoder = StateContextEncoder(
            state_dim=state_dim,
            action_width=action_width,
            dtype=dtype,
            rngs=rngs,
        )
        self.controller = ActionConditioningController(
            action_width=action_width,
            num_layers=num_layers,
            num_heads=num_heads,
            dtype=dtype,
            rngs=rngs,
        )

    def encode_overview(
        self,
        prefix_out: at.Array,
        prefix: PrefixEmbeddings,
        image_masks: Sequence[at.Array],
        language_mask: at.Array | None,
    ) -> at.Array:
        return self.overview_encoder(prefix_out, prefix, image_masks, language_mask)

    def encode_state(self, state: at.Array) -> at.Array:
        if not self.config.use_state_context:
            return jnp.zeros((state.shape[0], self.action_width), dtype=self.dtype)
        return self.state_encoder(state)

    def make_gates(self, scene_context: at.Array, state_context: at.Array) -> tuple[at.Array, at.Array]:
        return self.controller(scene_context, state_context)

    def __call__(
        self,
        prefix_out: at.Array,
        prefix: PrefixEmbeddings,
        image_masks: Sequence[at.Array],
        language_mask: at.Array | None,
        state: at.Array,
    ) -> tuple[at.Array, at.Array]:
        scene_context = self.encode_overview(prefix_out, prefix, image_masks, language_mask)
        state_context = self.encode_state(state)
        return self.make_gates(scene_context, state_context)
