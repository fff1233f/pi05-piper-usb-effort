import dataclasses

from flax import nnx
import jax
import numpy as np
import pytest

from openpi.models import model as _model
from openpi.models import pi0_config
from openpi.models import pi0_fast
from openpi.shared import download
from openpi.shared import nnx_utils


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


def test_pi05_effort_token_is_inserted_before_action_tokens():
    key = jax.random.key(0)
    config = pi0_config.Pi0Config(
        pi05=True,
        action_dim=8,
        action_horizon=3,
        paligemma_variant="dummy",
        action_expert_variant="dummy",
        effort_dim=6,
        effort_history_length=2,
    )
    model = config.create(key)
    obs, actions = config.fake_obs(batch_size=2), config.fake_act(batch_size=2)

    tokens, input_mask, ar_mask, adarms_cond = model.embed_suffix(obs, actions, jax.numpy.ones((2,)))

    assert tokens.shape == (2, config.action_horizon + 1, 64)
    assert input_mask.shape == (2, config.action_horizon + 1)
    np.testing.assert_array_equal(ar_mask, np.array([True, True, False, False]))
    assert adarms_cond.shape == (2, 64)

    with pytest.raises(ValueError, match="observation.effort is missing"):
        model.embed_suffix(dataclasses.replace(obs, effort=None), actions, jax.numpy.ones((2,)))


def test_pi05_without_effort_keeps_original_suffix_layout():
    key = jax.random.key(0)
    config = pi0_config.Pi0Config(
        pi05=True,
        action_dim=8,
        action_horizon=3,
        paligemma_variant="dummy",
        action_expert_variant="dummy",
    )
    model = config.create(key)
    obs, actions = config.fake_obs(batch_size=2), config.fake_act(batch_size=2)

    tokens, input_mask, ar_mask, _ = model.embed_suffix(obs, actions, jax.numpy.ones((2,)))

    assert obs.effort is None
    assert not hasattr(model, "effort_proj_in")
    assert not hasattr(model, "effort_proj_out")
    assert tokens.shape == (2, config.action_horizon, 64)
    assert input_mask.shape == (2, config.action_horizon)
    np.testing.assert_array_equal(ar_mask, np.array([True, False, False]))


def test_effort_projection_receives_gradients():
    config = pi0_config.Pi0Config(
        pi05=True,
        action_dim=8,
        action_horizon=3,
        paligemma_variant="dummy",
        action_expert_variant="dummy",
        effort_dim=6,
    )
    model = config.create(jax.random.key(0))
    effort = jax.numpy.ones((2, 1, 6))

    def loss_fn(model, effort):
        return jax.numpy.mean(jax.numpy.square(model._embed_effort(effort)))  # noqa: SLF001

    diff_state = nnx.DiffState(0, nnx_utils.PathRegex(".*effort_proj_.*"))
    _, grads = nnx.value_and_grad(loss_fn, argnums=diff_state)(model, effort)

    assert set(grads.flat_state()) == {
        ("effort_proj_in", "bias"),
        ("effort_proj_in", "kernel"),
        ("effort_proj_out", "bias"),
        ("effort_proj_out", "kernel"),
    }
    assert all(np.linalg.norm(np.asarray(variable.value)) > 0 for variable in grads.flat_state().values())


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
