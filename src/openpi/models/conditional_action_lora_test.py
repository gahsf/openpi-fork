import flax.traverse_util
import jax
import jax.numpy as jnp
import numpy as np

from openpi.models import gemma
from openpi.models import overview_action_conditioning as oac
from openpi.models.pi0 import make_attn_mask


def _init_model(*, enabled: bool):
    config = gemma.get_config("dummy")
    conditioning_config = oac.OverviewActionConditioningConfig(
        enabled=enabled,
        rank=2,
        lora_alpha=2.0,
        target="q",
    )
    model = gemma.Module(
        configs=[config, config],
        embed_dtype="float32",
        adarms=False,
        action_conditioning_config=conditioning_config,
    )
    _, params = model.init_with_output(jax.random.key(0), use_adarms=[False, False], method=model.init)
    return config, model, params


def _conditional_param_leaves(params):
    flat_params = flax.traverse_util.flatten_dict(params["params"], sep="/")
    return {path: value for path, value in flat_params.items() if "conditional_q_lora_1" in path}


def test_disabled_model_has_no_conditional_query_parameters():
    _, _, params = _init_model(enabled=False)

    assert not _conditional_param_leaves(params)


def test_conditional_query_lora_identity_cache_effect_and_gradients():
    config, model, params = _init_model(enabled=True)
    prefix = jax.random.normal(jax.random.key(1), (1, 4, config.width))
    suffix = jax.random.normal(jax.random.key(2), (1, 3, config.width))
    prefix_mask = jnp.ones((1, 4), dtype=jnp.bool_)
    suffix_mask = jnp.ones((1, 3), dtype=jnp.bool_)
    prefix_ar_mask = jnp.zeros((4,), dtype=jnp.bool_)
    suffix_ar_mask = jnp.array([True, True, False])

    prefix_attention_mask = make_attn_mask(prefix_mask, prefix_ar_mask)
    prefix_positions = jnp.cumsum(prefix_mask, axis=1) - 1
    zero_gates = jnp.zeros((1, config.depth, config.num_heads), dtype=jnp.float32)
    nonzero_q_gates = jnp.full_like(zero_gates, 0.5)

    (prefix_out, _), prefix_cache = model.apply(
        params,
        [prefix, None],
        mask=prefix_attention_mask,
        positions=prefix_positions,
    )
    (conditioned_prefix_out, _), conditioned_prefix_cache = model.apply(
        params,
        [prefix, None],
        mask=prefix_attention_mask,
        positions=prefix_positions,
        action_conditioning=(nonzero_q_gates, zero_gates),
    )

    np.testing.assert_array_equal(prefix_out, conditioned_prefix_out)
    for cache_leaf, conditioned_cache_leaf in zip(
        jax.tree.leaves(prefix_cache), jax.tree.leaves(conditioned_prefix_cache), strict=True
    ):
        np.testing.assert_array_equal(cache_leaf, conditioned_cache_leaf)

    suffix_attention_mask = make_attn_mask(suffix_mask, suffix_ar_mask)
    prefix_cross_attention_mask = jnp.broadcast_to(prefix_mask[:, None, :], (1, 3, 4))
    full_attention_mask = jnp.concatenate([prefix_cross_attention_mask, suffix_attention_mask], axis=-1)
    suffix_positions = jnp.sum(prefix_mask, axis=-1)[:, None] + jnp.cumsum(suffix_mask, axis=1) - 1

    def suffix_forward(model_params, action_conditioning):
        (_, suffix_out), _ = model.apply(
            model_params,
            [None, suffix],
            mask=full_attention_mask,
            positions=suffix_positions,
            kv_cache=prefix_cache,
            action_conditioning=action_conditioning,
        )
        return suffix_out

    bypass_suffix = suffix_forward(params, None)
    zero_gate_suffix = suffix_forward(params, (zero_gates, zero_gates))
    conditioned_suffix = suffix_forward(params, (nonzero_q_gates, zero_gates))

    np.testing.assert_array_equal(bypass_suffix, zero_gate_suffix)
    assert not bool(jnp.allclose(zero_gate_suffix, conditioned_suffix))

    conditional_params = _conditional_param_leaves(params)
    assert set(conditional_params) == {
        "layers/attn/conditional_q_lora_1/lora_a",
        "layers/attn/conditional_q_lora_1/lora_b",
    }
    assert conditional_params["layers/attn/conditional_q_lora_1/lora_a"].shape == (
        config.depth,
        config.num_heads,
        config.width,
        2,
    )
    assert conditional_params["layers/attn/conditional_q_lora_1/lora_b"].shape == (
        config.depth,
        config.num_heads,
        2,
        config.head_dim,
    )

    grads = jax.grad(
        lambda model_params: jnp.mean(jnp.square(suffix_forward(model_params, (nonzero_q_gates, zero_gates))))
    )(params)
    conditional_grads = _conditional_param_leaves(grads)
    assert set(conditional_grads) == set(conditional_params)
    assert all(bool(jnp.all(jnp.isfinite(grad))) for grad in conditional_grads.values())
    assert all(bool(jnp.any(grad != 0)) for grad in conditional_grads.values())
