import flax.nnx as nnx
import jax.numpy as jnp
import numpy as np
import pytest

from openpi.models import overview_action_conditioning as oac
from openpi.models import pi0_config
from openpi.shared import nnx_utils


def _make_prefix() -> oac.PrefixEmbeddings:
    return oac.PrefixEmbeddings(
        tokens=jnp.zeros((1, 7, 4), dtype=jnp.float32),
        input_mask=jnp.ones((1, 7), dtype=jnp.bool_),
        ar_mask=jnp.zeros((7,), dtype=jnp.bool_),
        image_lengths=(2, 2),
        language_length=3,
    )


def _make_conditioning(*, use_state_context: bool = True, dtype: str = "float32") -> oac.OverviewActionConditioning:
    return oac.OverviewActionConditioning(
        oac.OverviewActionConditioningConfig(enabled=True, use_state_context=use_state_context),
        vlm_width=4,
        action_width=6,
        state_dim=5,
        num_views=2,
        num_layers=3,
        num_heads=2,
        dtype=dtype,
        rngs=nnx.Rngs(0),
    )


def test_overview_context_ignores_masked_tokens_and_is_finite():
    conditioning = _make_conditioning()
    prefix = _make_prefix()
    prefix_out = jnp.arange(28, dtype=jnp.float32).reshape(1, 7, 4)
    image_masks = (jnp.array([False]), jnp.array([False]))
    language_mask = jnp.array([[True, True, False]])

    expected = conditioning.encode_overview(prefix_out, prefix, image_masks, language_mask)
    changed = prefix_out.at[:, :4].set(1e6)
    changed = changed.at[:, 6].set(-1e6)
    actual = conditioning.encode_overview(changed, prefix, image_masks, language_mask)

    np.testing.assert_array_equal(expected, actual)
    assert bool(jnp.all(jnp.isfinite(actual)))


def test_overview_context_preserves_view_order():
    conditioning = _make_conditioning()
    prefix = _make_prefix()
    prefix_out = jnp.array(
        [
            [
                [1, 0, 0, 0],
                [0, 1, 0, 0],
                [0, 0, 1, 0],
                [0, 0, 0, 2],
                [1, 2, 3, 4],
                [2, 3, 4, 1],
                [3, 4, 1, 2],
            ]
        ],
        dtype=jnp.float32,
    )
    image_masks = (jnp.array([True]), jnp.array([True]))
    language_mask = jnp.ones((1, 3), dtype=jnp.bool_)

    original = conditioning.encode_overview(prefix_out, prefix, image_masks, language_mask)
    swapped = jnp.concatenate([prefix_out[:, 2:4], prefix_out[:, :2], prefix_out[:, 4:]], axis=1)
    reordered = conditioning.encode_overview(swapped, prefix, image_masks, language_mask)

    assert not bool(jnp.allclose(original, reordered))


def test_conditioning_shapes_zero_gates_and_jit():
    conditioning = _make_conditioning()
    prefix = _make_prefix()
    prefix_out = jnp.arange(28, dtype=jnp.float32).reshape(1, 7, 4)
    image_masks = (jnp.array([True]), jnp.array([False]))
    language_mask = jnp.array([[True, True, False]])
    state = jnp.ones((1, 5), dtype=jnp.float32)

    scene_context = conditioning.encode_overview(prefix_out, prefix, image_masks, language_mask)
    state_context = conditioning.encode_state(state)
    q_gates, o_gates = nnx_utils.module_jit(conditioning.__call__)(
        prefix_out, prefix, image_masks, language_mask, state
    )

    assert scene_context.shape == (1, 6)
    assert state_context.shape == (1, 6)
    assert q_gates.shape == (1, 3, 2)
    assert o_gates.shape == (1, 3, 2)
    assert scene_context.dtype == jnp.float32
    assert state_context.dtype == jnp.float32
    assert q_gates.dtype == jnp.float32
    assert o_gates.dtype == jnp.float32
    np.testing.assert_array_equal(q_gates, jnp.zeros_like(q_gates))
    np.testing.assert_array_equal(o_gates, jnp.zeros_like(o_gates))


def test_disabled_state_context_returns_zeros():
    conditioning = _make_conditioning(use_state_context=False)

    state_context = conditioning.encode_state(jnp.ones((2, 5), dtype=jnp.float32))

    np.testing.assert_array_equal(state_context, jnp.zeros((2, 6), dtype=jnp.float32))


def test_constant_reference_inputs_ignore_samples_and_remain_nonzero():
    prefix_out = jnp.arange(56, dtype=jnp.float32).reshape(2, 7, 4)
    image_masks = (jnp.array([True, False]), jnp.array([False, True]))
    language_mask = jnp.array([[True, True, False], [False, True, True]])
    state = jnp.arange(10, dtype=jnp.float32).reshape(2, 5)

    fixed_prefix, fixed_image_masks, fixed_language_mask, fixed_state = oac.make_constant_reference_inputs(
        prefix_out, image_masks, language_mask, state
    )

    np.testing.assert_array_equal(fixed_prefix[0], fixed_prefix[1])
    np.testing.assert_array_equal(fixed_state[0], fixed_state[1])
    assert bool(jnp.any(fixed_prefix != 0))
    assert bool(jnp.any(fixed_state != 0))
    assert all(bool(jnp.all(mask)) for mask in fixed_image_masks)
    assert bool(jnp.all(fixed_language_mask))


def test_conditioning_respects_bfloat16_dtype():
    conditioning = _make_conditioning(dtype="bfloat16")
    prefix = _make_prefix()
    prefix_out = jnp.arange(28, dtype=jnp.bfloat16).reshape(1, 7, 4)
    image_masks = (jnp.array([True]), jnp.array([False]))
    language_mask = jnp.array([[True, True, False]])
    state = jnp.ones((1, 5), dtype=jnp.bfloat16)

    scene_context = conditioning.encode_overview(prefix_out, prefix, image_masks, language_mask)
    state_context = conditioning.encode_state(state)
    q_gates, o_gates = conditioning.make_gates(scene_context, state_context)

    assert scene_context.dtype == jnp.bfloat16
    assert state_context.dtype == jnp.bfloat16
    assert q_gates.dtype == jnp.bfloat16
    assert o_gates.dtype == jnp.bfloat16


def test_zero_gate_projection_has_nonzero_gradient():
    controller = oac.ActionConditioningController(
        action_width=6,
        num_layers=3,
        num_heads=2,
        dtype="float32",
        rngs=nnx.Rngs(0),
    )
    scene_context = jnp.arange(12, dtype=jnp.float32).reshape(2, 6)
    state_context = jnp.flip(scene_context, axis=-1)
    weights = jnp.arange(6, dtype=jnp.float32).reshape(1, 3, 2) + 1

    def loss_fn(module):
        q_gates, o_gates = module(scene_context, state_context)
        return jnp.sum(q_gates * weights) + jnp.sum(o_gates * weights)

    grads = nnx.grad(loss_fn)(controller)

    assert bool(jnp.any(grads.gate_proj.kernel != 0))
    assert bool(jnp.any(grads.gate_proj.bias != 0))


def test_config_validation_and_pi0_default():
    config = pi0_config.Pi0Config()
    assert not config.overview_action_conditioning.enabled

    with pytest.raises(ValueError, match="rank"):
        oac.OverviewActionConditioningConfig(rank=0)
    with pytest.raises(ValueError, match="lora_alpha"):
        oac.OverviewActionConditioningConfig(lora_alpha=0)
    with pytest.raises(ValueError, match="target"):
        oac.OverviewActionConditioningConfig(target="invalid")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="conditioning_mode"):
        oac.OverviewActionConditioningConfig(conditioning_mode="invalid")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="enabled"):
        oac.OverviewActionConditioningConfig(enabled=False, conditioning_mode="static")
    with pytest.raises(ValueError, match="pi05"):
        pi0_config.Pi0Config(pi05=True, overview_action_conditioning=oac.OverviewActionConditioningConfig(enabled=True))
    with pytest.raises(ValueError, match="LoRA"):
        pi0_config.Pi0Config(
            paligemma_variant="gemma_2b_lora",
            overview_action_conditioning=oac.OverviewActionConditioningConfig(enabled=True),
        )
    with pytest.raises(ValueError, match="LoRA"):
        pi0_config.Pi0Config(
            action_expert_variant="gemma_300m_lora",
            overview_action_conditioning=oac.OverviewActionConditioningConfig(enabled=True),
        )
