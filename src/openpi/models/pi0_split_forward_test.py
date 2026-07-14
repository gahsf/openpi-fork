import jax
import jax.numpy as jnp
import numpy as np

from openpi.models import gemma
from openpi.models.pi0 import make_attn_mask


def test_joint_and_cached_suffix_forward_match():
    config = gemma.get_config("dummy")
    model = gemma.Module(configs=[config, config], embed_dtype="float32", adarms=False)
    _, params = model.init_with_output(jax.random.key(0), use_adarms=[False, False], method=model.init)

    prefix_key, suffix_key = jax.random.split(jax.random.key(1))
    prefix = jax.random.normal(prefix_key, (2, 5, config.width))
    suffix = jax.random.normal(suffix_key, (2, 3, config.width))
    prefix_mask = jnp.array([[True, True, True, True, True], [True, True, True, False, False]])
    suffix_mask = jnp.ones((2, 3), dtype=jnp.bool_)
    prefix_ar_mask = jnp.zeros((5,), dtype=jnp.bool_)
    suffix_ar_mask = jnp.array([True, True, False])

    input_mask = jnp.concatenate([prefix_mask, suffix_mask], axis=1)
    ar_mask = jnp.concatenate([prefix_ar_mask, suffix_ar_mask], axis=0)
    joint_mask = make_attn_mask(input_mask, ar_mask)
    joint_positions = jnp.cumsum(input_mask, axis=1) - 1
    (_, joint_suffix), joint_cache = model.apply(
        params,
        [prefix, suffix],
        mask=joint_mask,
        positions=joint_positions,
    )

    prefix_attention_mask = make_attn_mask(prefix_mask, prefix_ar_mask)
    prefix_positions = jnp.cumsum(prefix_mask, axis=1) - 1
    suffix_attention_mask = make_attn_mask(suffix_mask, suffix_ar_mask)
    prefix_cross_attention_mask = jnp.broadcast_to(prefix_mask[:, None, :], (2, 3, 5))
    full_attention_mask = jnp.concatenate([prefix_cross_attention_mask, suffix_attention_mask], axis=-1)
    suffix_positions = jnp.sum(prefix_mask, axis=-1)[:, None] + jnp.cumsum(suffix_mask, axis=1) - 1

    def cached_forward(model_params):
        _, prefix_cache = model.apply(
            model_params,
            [prefix, None],
            mask=prefix_attention_mask,
            positions=prefix_positions,
        )
        (_, cached_suffix), cached_joint_cache = model.apply(
            model_params,
            [None, suffix],
            mask=full_attention_mask,
            positions=suffix_positions,
            kv_cache=prefix_cache,
        )
        return cached_suffix, cached_joint_cache

    cached_suffix, cached_joint_cache = cached_forward(params)

    np.testing.assert_allclose(joint_suffix, cached_suffix, rtol=5e-3, atol=5e-3)
    valid_cache_mask = jnp.broadcast_to(input_mask[None, :, :, None, None], joint_cache[0].shape)
    for joint_cache_leaf, cached_cache_leaf in zip(
        jax.tree.leaves(joint_cache), jax.tree.leaves(cached_joint_cache), strict=True
    ):
        np.testing.assert_allclose(
            joint_cache_leaf[valid_cache_mask], cached_cache_leaf[valid_cache_mask], rtol=5e-3, atol=5e-3
        )

    grads = jax.grad(lambda model_params: jnp.mean(jnp.square(cached_forward(model_params)[0])))(params)
    grad_leaves = jax.tree.leaves(grads)
    assert all(bool(jnp.all(jnp.isfinite(grad))) for grad in grad_leaves)
    assert any(bool(jnp.any(grad != 0)) for grad in grad_leaves)
