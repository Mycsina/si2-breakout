from breakout_rl.env.high_level_env import HighLevelEnv, HIGH_OBS_DIM
from breakout_rl.constants import REGION_CENTER


def test_reset_returns_high_obs():
    env = HighLevelEnv(max_option_steps=2000)
    obs, info = env.reset(seed=0)
    assert obs.shape == (HIGH_OBS_DIM,)


def test_step_returns_gamma_k_and_k():
    env = HighLevelEnv(max_option_steps=2000)
    env.reset(seed=0)
    obs, R, term, trunc, info = env.step(REGION_CENTER)
    assert "gamma_k" in info and "k" in info
    assert info["k"] >= 1
    assert 0.0 < info["gamma_k"] <= 1.0
    # gamma_k == gamma ** k
    assert abs(info["gamma_k"] - env.gamma ** info["k"]) < 1e-9
