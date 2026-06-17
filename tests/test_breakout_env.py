import numpy as np
from breakout_rl.env.breakout_env import BreakoutEnv
from breakout_rl.env.observation import OBS_DIM, ObservationBuilder
from breakout_rl.constants import ACTION_NOOP, ACTION_WEST, DT


def test_reset_returns_obs_of_correct_shape():
    env = BreakoutEnv()
    obs, info = env.reset(seed=0)
    assert obs.shape == (OBS_DIM,)
    assert env.action_space.n == 3


def test_step_returns_5_tuple_and_moves_paddle():
    env = BreakoutEnv()
    env.reset(seed=0)
    x0 = env.game.paddle_x
    obs, r, term, trunc, info = env.step(ACTION_WEST)
    assert obs.shape == (OBS_DIM,)
    assert isinstance(r, float)
    assert env.game.paddle_x <= x0  # moved west (or already at wall)


def test_observation_built_from_wire_state_not_internal_velocity():
    # Guards train/deploy skew: the obs the env RETURNS from step() must be reproducible
    # by an independent ObservationBuilder fed the same get_state() frame sequence — i.e.
    # the env derives ball velocity from successive wire states, never from ball_vx/vy.
    # (The builder is stateful, so we replay the exact frames the env consumed rather than
    # re-calling _build_obs, which would double-consume the same frame and zero velocity.)
    env = BreakoutEnv()
    env.reset(seed=1)
    s_reset = env._prev_state_for_test  # frame the env's builder consumed at reset
    obs_step, _, _, _, _ = env.step(ACTION_NOOP)
    s_after = env.game.get_state()  # frame the env's builder consumed in step()

    ref_builder = ObservationBuilder()
    ref_builder.reset()
    ref_builder.build(s_reset, DT)  # replay reset frame
    expected = ref_builder.build(s_after, DT)  # replay post-step frame
    assert np.allclose(obs_step, expected)
    assert obs_step[3] != 0.0 or obs_step[4] != 0.0  # velocity actually recovered


def test_truncates_at_max_steps():
    env = BreakoutEnv(max_steps=3)
    env.reset(seed=0)
    trunc = False
    for _ in range(3):
        _, _, term, trunc, _ = env.step(ACTION_NOOP)
        if term:
            break
    assert trunc or term
