import numpy as np
from server.logic import Breakout
from breakout_rl.env.observation import ObservationBuilder
from breakout_rl.constants import DT


def test_env_path_and_deploy_path_produce_identical_vectors():
    """The env and the deployed agent both build observations from get_state() via the
    same ObservationBuilder. Feeding an identical state sequence through two independent
    builders must yield byte-identical vectors (guards train/deploy skew)."""
    g = Breakout()
    g.ball_x, g.ball_y = 250.0, 150.0
    s1 = g.get_state()
    g.update(DT)
    s2 = g.get_state()

    env_builder = ObservationBuilder()
    deploy_builder = ObservationBuilder()
    env_builder.build(s1, DT); deploy_builder.build(s1, DT)
    v_env = env_builder.build(s2, DT)
    v_deploy = deploy_builder.build(s2, DT)

    assert np.array_equal(v_env, v_deploy)
