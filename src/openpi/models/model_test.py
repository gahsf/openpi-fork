from flax import nnx
import jax
import jax.numpy as jnp
import numpy as np
import optax
import orbax.checkpoint as ocp
import pytest

from openpi.models import model as _model
from openpi.models import overview_action_conditioning
from openpi.models import pi0_config
from openpi.models import pi0_fast
from openpi.shared import download
from openpi.shared import nnx_utils
from openpi.training import config as training_config
from openpi.training import optimizer as training_optimizer
from openpi.training import weight_loaders


def _ocaev1_dummy_config():
    return pi0_config.Pi0Config(
        paligemma_variant="dummy",
        action_expert_variant="dummy",
        overview_action_conditioning=overview_action_conditioning.OverviewActionConditioningConfig(
            enabled=True,
            rank=2,
            lora_alpha=2.0,
            target="q_o",
        ),
    )


def test_pi0_model():
    key = jax.random.key(0)
    config = pi0_config.Pi0Config()
    model = config.create(key)

    batch_size = 2
    obs, act = config.fake_obs(batch_size), config.fake_act(batch_size)

    loss = nnx_utils.module_jit(model.compute_loss)(key, obs, act)
    assert loss.shape == (batch_size, config.action_horizon)

    actions = nnx_utils.module_jit(model.sample_actions)(key, obs, num_steps=10)
    assert actions.shape == (batch_size, model.action_horizon, model.action_dim)


def test_pi0_lora_model():
    key = jax.random.key(0)
    config = pi0_config.Pi0Config(paligemma_variant="gemma_2b_lora")
    model = config.create(key)

    batch_size = 2
    obs, act = config.fake_obs(batch_size), config.fake_act(batch_size)

    loss = nnx_utils.module_jit(model.compute_loss)(key, obs, act)
    assert loss.shape == (batch_size, config.action_horizon)

    actions = nnx_utils.module_jit(model.sample_actions)(key, obs, num_steps=10)
    assert actions.shape == (batch_size, model.action_horizon, model.action_dim)


def test_pi0_ocaev1_qo_model():
    key = jax.random.key(0)
    config = _ocaev1_dummy_config()
    model = config.create(key)

    batch_size = 1
    obs, act = config.fake_obs(batch_size), config.fake_act(batch_size)

    loss, aux = nnx_utils.module_jit(model.compute_loss_with_aux)(key, obs, act)
    assert loss.shape == (batch_size, config.action_horizon)
    assert set(aux) == {
        "gate_regularization_loss",
        "o_gate_abs_gt_0_9",
        "o_gate_abs_max",
        "o_raw_gate_abs_max",
        "q_gate_abs_gt_0_9",
        "q_gate_abs_max",
        "q_raw_gate_abs_max",
    }
    assert all(bool(jnp.isfinite(value)) for value in aux.values())

    actions = nnx_utils.module_jit(model.sample_actions)(key, obs, num_steps=2)
    assert actions.shape == (batch_size, model.action_horizon, model.action_dim)


def test_pi0_ocaev1_training_config():
    config = training_config.get_config("pi0_ocaev1_debug")

    assert isinstance(config.model, pi0_config.Pi0Config)
    assert config.model.overview_action_conditioning == overview_action_conditioning.OverviewActionConditioningConfig(
        enabled=True,
        rank=16,
        lora_alpha=16.0,
        target="q_o",
    )
    assert config.freeze_filter == config.model.get_freeze_filter()
    assert isinstance(config.weight_loader, weight_loaders.CheckpointWeightLoader)
    assert config.weight_loader.params_path == "gs://openpi-assets/checkpoints/pi0_base/params"
    assert config.batch_size == 1
    assert config.ema_decay is None


def test_pi0_ocaev1_smoke_training_config():
    config = training_config.get_config("pi0_ocaev1_smoke")

    assert isinstance(config.model, pi0_config.Pi0Config)
    assert config.model.overview_action_conditioning == overview_action_conditioning.OverviewActionConditioningConfig(
        enabled=True,
        rank=16,
        lora_alpha=16.0,
        target="q_o",
    )
    assert config.freeze_filter == config.model.get_freeze_filter()
    assert isinstance(config.weight_loader, weight_loaders.CheckpointWeightLoader)
    assert isinstance(config.lr_schedule, training_optimizer.CosineDecaySchedule)
    assert config.lr_schedule.warmup_steps == 5
    assert config.lr_schedule.peak_lr == 1e-5
    assert config.lr_schedule.decay_steps == 50
    assert config.lr_schedule.decay_lr == 1e-6
    assert config.batch_size == 1
    assert config.num_train_steps == 50
    assert config.log_interval == 1
    assert config.ema_decay is None


@pytest.mark.parametrize(
    ("name", "target", "conditioning_mode"),
    [
        ("pi0_libero_e1_static_qo", "q_o", "static"),
        ("pi0_libero_e2_constant_qo", "q_o", "constant"),
        ("pi0_libero_e5_ocae_q", "q", "sample"),
        ("pi0_libero_e6_ocae_o", "o", "sample"),
        ("pi0_libero_e7_ocae_qo", "q_o", "sample"),
    ],
)
def test_pi0_ocaev1_libero_experiment_configs(name, target, conditioning_mode):
    config = training_config.get_config(name)

    assert isinstance(config.model, pi0_config.Pi0Config)
    assert config.model.overview_action_conditioning.enabled
    assert config.model.overview_action_conditioning.target == target
    assert config.model.overview_action_conditioning.conditioning_mode == conditioning_mode
    assert config.model.overview_action_conditioning.rank == 16
    expected_gate_l2 = 3e-2 if name in {"pi0_libero_e2_constant_qo", "pi0_libero_e7_ocae_qo"} else 0.0
    assert config.model.overview_action_conditioning.gate_l2_regularization == expected_gate_l2

    assert config.freeze_filter == config.model.get_freeze_filter()
    assert isinstance(config.data, training_config.LeRobotLiberoDataConfig)
    assert config.data.repo_id == "physical-intelligence/libero"
    assert config.data.extra_delta_transform
    assert config.data.assets.assets_dir == "./assets/pi0_libero_e7_ocae_qo"
    assert config.data.assets.asset_id == "physical-intelligence/libero"
    assert isinstance(config.weight_loader, weight_loaders.CheckpointWeightLoader)
    assert config.batch_size == 32
    assert config.num_train_steps == 30_000
    assert config.ema_decay is None


def test_pi0_libero_e0_action_lora_config():
    config = training_config.get_config("pi0_libero_e0_action_lora")

    assert isinstance(config.model, pi0_config.Pi0Config)
    assert config.model.action_expert_variant == "gemma_300m_lora"
    assert not config.model.overview_action_conditioning.enabled
    assert isinstance(config.data, training_config.LeRobotLiberoDataConfig)
    assert config.data.repo_id == "physical-intelligence/libero"
    assert config.data.extra_delta_transform
    assert config.data.assets.assets_dir == "./assets/pi0_libero_e7_ocae_qo"
    assert config.batch_size == 32
    assert config.num_train_steps == 30_000
    assert config.ema_decay is None


def test_pi0_libero_e7_gate025_config():
    config = training_config.get_config("pi0_libero_e7_ocae_qo_gate025")

    assert isinstance(config.model, pi0_config.Pi0Config)
    assert config.model.overview_action_conditioning.gate_logit_scale == 0.25
    assert config.model.overview_action_conditioning.conditioning_mode == "sample"
    assert config.freeze_filter == config.model.get_freeze_filter()
    assert config.batch_size == 32
    assert config.num_train_steps == 30_000


def test_pi0_libero_e2_gate025_config():
    config = training_config.get_config("pi0_libero_e2_constant_qo_gate025")

    assert isinstance(config.model, pi0_config.Pi0Config)
    assert config.model.overview_action_conditioning.gate_logit_scale == 0.25
    assert config.model.overview_action_conditioning.conditioning_mode == "constant"
    assert config.freeze_filter == config.model.get_freeze_filter()
    assert config.batch_size == 32
    assert config.num_train_steps == 30_000


def test_pi0_libero_e7_gate_l2_config():
    config = training_config.get_config("pi0_libero_e7_ocae_qo_gate_l2_1e3")

    assert isinstance(config.model, pi0_config.Pi0Config)
    assert config.model.overview_action_conditioning.gate_logit_scale == 1.0
    assert config.model.overview_action_conditioning.gate_l2_regularization == 1e-3
    assert config.model.overview_action_conditioning.conditioning_mode == "sample"
    assert config.freeze_filter == config.model.get_freeze_filter()
    assert config.batch_size == 32
    assert config.num_train_steps == 30_000


def test_pi0_libero_e7_gate_l2_1e2_config():
    config = training_config.get_config("pi0_libero_e7_ocae_qo_gate_l2_1e2")

    assert isinstance(config.model, pi0_config.Pi0Config)
    assert config.model.overview_action_conditioning.gate_logit_scale == 1.0
    assert config.model.overview_action_conditioning.gate_l2_regularization == 1e-2
    assert config.model.overview_action_conditioning.conditioning_mode == "sample"
    assert config.freeze_filter == config.model.get_freeze_filter()
    assert config.batch_size == 32
    assert config.num_train_steps == 30_000


def test_pi0_libero_e7_gate_l2_3e2_pilot_config():
    config = training_config.get_config("pi0_libero_e7_ocae_qo_gate_l2_3e2")

    assert isinstance(config.model, pi0_config.Pi0Config)
    assert config.model.overview_action_conditioning.gate_logit_scale == 1.0
    assert config.model.overview_action_conditioning.gate_l2_regularization == 3e-2
    assert config.model.overview_action_conditioning.conditioning_mode == "sample"
    assert config.freeze_filter == config.model.get_freeze_filter()
    assert config.batch_size == 32
    assert config.num_train_steps == 1_000
    assert config.save_interval == 1_000


def test_pi0_libero_e2_gate_l2_3e2_pilot_config():
    config = training_config.get_config("pi0_libero_e2_constant_qo_gate_l2_3e2")

    assert isinstance(config.model, pi0_config.Pi0Config)
    assert config.model.overview_action_conditioning.gate_logit_scale == 1.0
    assert config.model.overview_action_conditioning.gate_l2_regularization == 3e-2
    assert config.model.overview_action_conditioning.conditioning_mode == "constant"
    assert config.freeze_filter == config.model.get_freeze_filter()
    assert config.batch_size == 32
    assert config.num_train_steps == 1_000
    assert config.save_interval == 1_000


def test_pi0_ocaev1_train_sample_and_checkpoint_roundtrip(tmp_path):
    key = jax.random.key(0)
    config = _ocaev1_dummy_config()
    model = config.create(key)
    obs, act = config.fake_obs(batch_size=1), config.fake_act(batch_size=1)
    trainable_filter = nnx.All(nnx.Param, nnx.Not(config.get_freeze_filter()))

    @nnx.jit
    def train_step(module, rng, observation, actions):
        def loss_fn(model):
            return jnp.mean(model.compute_loss(rng, observation, actions, train=True))

        loss, grads = nnx.value_and_grad(loss_fn, argnums=nnx.DiffState(0, trainable_filter))(module)
        params = nnx.state(module, trainable_filter)
        updates = jax.tree.map(lambda grad: -1e-3 * grad, grads)
        nnx.update(module, optax.apply_updates(params, updates))
        return loss, grads

    loss, grads = train_step(model, key, obs, act)
    assert bool(jnp.isfinite(loss))
    assert all(bool(jnp.all(jnp.isfinite(grad))) for grad in jax.tree.leaves(grads))
    assert any(bool(jnp.any(grad != 0)) for grad in jax.tree.leaves(grads))

    noise = jax.random.normal(key, (1, config.action_horizon, config.action_dim))
    expected_actions = nnx_utils.module_jit(model.sample_actions)(key, obs, num_steps=2, noise=noise)

    params_path = tmp_path / "params"
    with ocp.PyTreeCheckpointer() as checkpointer:
        checkpointer.save(params_path, {"params": nnx.state(model).to_pure_dict()})
    restored_model = config.load(_model.restore_params(params_path))
    actual_actions = nnx_utils.module_jit(restored_model.sample_actions)(key, obs, num_steps=2, noise=noise)

    np.testing.assert_array_equal(actual_actions, expected_actions)


def test_pi0_fast_model():
    key = jax.random.key(0)
    config = pi0_fast.Pi0FASTConfig()
    model = config.create(key)

    batch_size = 2
    obs, act = config.fake_obs(batch_size), config.fake_act(batch_size)

    loss = nnx_utils.module_jit(model.compute_loss)(key, obs, act)
    assert loss.shape == (batch_size,)

    actions = nnx_utils.module_jit(model.sample_actions)(key, obs)
    assert actions.shape == (batch_size, 256)


def test_pi0_fast_lora_model():
    key = jax.random.key(0)
    config = pi0_fast.Pi0FASTConfig(paligemma_variant="gemma_2b_lora")
    model = config.create(key)

    batch_size = 2
    obs, act = config.fake_obs(batch_size), config.fake_act(batch_size)

    loss = nnx_utils.module_jit(model.compute_loss)(key, obs, act)
    assert loss.shape == (batch_size,)

    actions = nnx_utils.module_jit(model.sample_actions)(key, obs)
    assert actions.shape == (batch_size, 256)

    lora_filter = nnx_utils.PathRegex(".*lora.*")
    model_state = nnx.state(model)

    lora_state_elems = list(model_state.filter(lora_filter))
    assert len(lora_state_elems) > 0


@pytest.mark.manual
def test_model_restore():
    key = jax.random.key(0)
    config = pi0_config.Pi0Config()

    batch_size = 2
    obs, act = config.fake_obs(batch_size), config.fake_act(batch_size)

    model = config.load(
        _model.restore_params(download.maybe_download("gs://openpi-assets/checkpoints/pi0_base/params"))
    )

    loss = model.compute_loss(key, obs, act)
    assert loss.shape == (batch_size, config.action_horizon)

    actions = model.sample_actions(key, obs, num_steps=10)
    assert actions.shape == (batch_size, model.action_horizon, model.action_dim)
