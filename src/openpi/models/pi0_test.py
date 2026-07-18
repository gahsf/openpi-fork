import flax.nnx as nnx
import jax
import pytest

import openpi.models.overview_action_conditioning as _overview_action_conditioning
import openpi.models.pi0_config as _pi0_config


def _get_frozen_state(config: _pi0_config.Pi0Config) -> nnx.State:
    abstract_model = nnx.eval_shape(config.create, jax.random.key(0))

    freeze_filter = config.get_freeze_filter()
    return nnx.state(abstract_model, nnx.All(nnx.Param, freeze_filter)).flat_state()


def _ocaev1_config(target):
    return _pi0_config.Pi0Config(
        paligemma_variant="dummy",
        action_expert_variant="dummy",
        overview_action_conditioning=_overview_action_conditioning.OverviewActionConditioningConfig(
            enabled=True,
            rank=2,
            lora_alpha=2.0,
            target=target,
        ),
    )


def test_pi0_full_finetune():
    config = _pi0_config.Pi0Config()
    state = _get_frozen_state(config)
    assert len(state) == 0


def test_pi0_gemma_lora():
    config = _pi0_config.Pi0Config(paligemma_variant="gemma_2b_lora")
    state = _get_frozen_state(config)
    assert len(state) == 9
    assert all("lora" not in p for p in state)
    assert all("llm" in p for p in state)
    assert all("_1" not in p for p in state)


def test_pi0_action_expert_lora():
    config = _pi0_config.Pi0Config(action_expert_variant="gemma_300m_lora")
    state = _get_frozen_state(config)
    # excluding embedder, rest of the params should be same as gemma_lora.
    assert len(state) == 8
    assert all("lora" not in p for p in state)
    assert all("llm" in p for p in state)
    # all frozen params should have _1 in their path since it's the action expert.
    assert all(any("_1" in p for p in path) for path in state)


def test_pi0_all_lora():
    config = _pi0_config.Pi0Config(paligemma_variant="gemma_2b_lora", action_expert_variant="gemma_300m_lora")
    state = _get_frozen_state(config)
    # sum of gemma_lora and action_expert_lora's frozen params.
    assert len(state) == 17
    assert all("lora" not in p for p in state)
    assert all("llm" in p for p in state)


@pytest.mark.parametrize(
    ("target", "expects_q", "expects_o"),
    [
        ("q", True, False),
        ("o", False, True),
        ("q_o", True, True),
    ],
)
def test_pi0_ocaev1_parameter_tree(target, expects_q, expects_o):
    config = _ocaev1_config(target)
    abstract_model = nnx.eval_shape(config.create, jax.random.key(0))
    paths = ["/".join(str(part) for part in path) for path in nnx.state(abstract_model).flat_state()]

    assert any("overview_action_conditioning" in path for path in paths)
    assert any("conditional_q_lora_1" in path for path in paths) is expects_q
    assert any("conditional_o_lora_1" in path for path in paths) is expects_o


def test_pi0_static_qo_has_only_conditional_lora_trainable_params():
    config = _pi0_config.Pi0Config(
        paligemma_variant="dummy",
        action_expert_variant="dummy",
        overview_action_conditioning=_overview_action_conditioning.OverviewActionConditioningConfig(
            enabled=True,
            rank=2,
            lora_alpha=2.0,
            target="q_o",
            conditioning_mode="static",
        ),
    )
    abstract_model = nnx.eval_shape(config.create, jax.random.key(0))
    trainable_paths = [
        "/".join(str(part) for part in path)
        for path in nnx.state(abstract_model, nnx.All(nnx.Param, nnx.Not(config.get_freeze_filter()))).flat_state()
    ]

    assert trainable_paths
    assert all("conditional_q_lora_1" in path or "conditional_o_lora_1" in path for path in trainable_paths)
    assert not any("overview_action_conditioning" in path for path in trainable_paths)


@pytest.mark.parametrize(
    ("target", "expects_q", "expects_o"),
    [
        ("q", True, False),
        ("o", False, True),
        ("q_o", True, True),
    ],
)
def test_pi0_ocaev1_freezes_everything_except_new_params(target, expects_q, expects_o):
    config = _ocaev1_config(target)
    abstract_model = nnx.eval_shape(config.create, jax.random.key(0))
    trainable_state = nnx.state(
        abstract_model,
        nnx.All(nnx.Param, nnx.Not(config.get_freeze_filter())),
    ).flat_state()
    trainable_paths = ["/".join(str(part) for part in path) for path in trainable_state]
    frozen_paths = ["/".join(str(part) for part in path) for path in _get_frozen_state(config)]

    assert trainable_paths
    assert all(
        "overview_action_conditioning" in path or "conditional_q_lora_1" in path or "conditional_o_lora_1" in path
        for path in trainable_paths
    )
    assert any("overview_action_conditioning" in path for path in trainable_paths)
    assert any("conditional_q_lora_1" in path for path in trainable_paths) is expects_q
    assert any("conditional_o_lora_1" in path for path in trainable_paths) is expects_o
    assert not any("action_out_proj" in path for path in trainable_paths)
    assert not any("state_proj" in path for path in trainable_paths)
    assert not any(
        "overview_action_conditioning" in path or "conditional_q_lora_1" in path or "conditional_o_lora_1" in path
        for path in frozen_paths
    )
    assert any("PaliGemma/img" in path for path in frozen_paths)
    assert any("PaliGemma/llm/layers/attn/q_einsum_1" in path for path in frozen_paths)
    assert any("state_proj" in path for path in frozen_paths)
    assert any("action_in_proj" in path for path in frozen_paths)
    assert any("action_out_proj" in path for path in frozen_paths)
