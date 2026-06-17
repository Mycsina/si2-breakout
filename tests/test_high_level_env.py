from breakout_rl.env.high_level_env import (
    HighLevelEnv,
    HIGH_OBS_DIM,
    crossed_decision_line,
)
from breakout_rl.env.observation import OBS_DIM
from breakout_rl.env.rewards import RewardConfig
from breakout_rl.constants import REGION_CENTER, DECISION_LINE_Y


def test_crossed_decision_line_fires_only_on_downward_line_crossing():
    # the shared predicate used by both training (HighLevelEnv) and deploy: it must fire
    # exactly when the ball crosses the decision line moving DOWN (vy > 0).
    above, below = DECISION_LINE_Y - 5.0, DECISION_LINE_Y + 5.0
    # crossing the line downward -> decision point
    assert crossed_decision_line(above, below, ball_vy=300.0)
    # crossing the line upward (vy < 0) -> not a decision point
    assert not crossed_decision_line(below, above, ball_vy=-300.0)
    # descending but already past the line (no crossing this frame) -> no
    assert not crossed_decision_line(below, below + 5.0, ball_vy=300.0)
    # descending but still above the line (brick field) -> no
    assert not crossed_decision_line(above - 5.0, above, ball_vy=300.0)


def test_reset_returns_high_obs():
    env = HighLevelEnv(max_option_steps=2000)
    obs, info = env.reset(seed=0)
    assert obs.shape == (HIGH_OBS_DIM,)


def test_high_obs_is_base_plus_brick_centroid_feature():
    # one extra feature on top of the shared flat observation
    assert HIGH_OBS_DIM == OBS_DIM + 1
    env = HighLevelEnv(max_option_steps=2000)
    obs, _ = env.reset(seed=0)
    assert obs.shape == (OBS_DIM + 1,)


def test_step_reports_bricks_broken():
    env = HighLevelEnv(max_option_steps=2000)
    env.reset(seed=0)
    _, _, _, _, info = env.step(REGION_CENTER)
    assert "broken" in info and info["broken"] >= 0


def test_waste_penalty_applied_only_on_zero_brick_volley():
    # same seed + same (no-brick) trajectory: the only difference between the two runs is
    # the waste penalty, so the return must differ by exactly that penalty.
    def run(penalty):
        env = HighLevelEnv(
            max_option_steps=2000, reward_cfg=RewardConfig(waste_penalty=penalty)
        )
        env.reset(seed=0)
        for b in env.game.bricks:  # remove all bricks -> every volley breaks zero
            b.active = False
        env.game._sync_bricks_to_numpy()
        _, R, _, _, info = env.step(REGION_CENTER)
        return R, info["broken"]

    r0, broken0 = run(0.0)
    rp, brokenp = run(-100.0)
    assert broken0 == 0 and brokenp == 0
    assert abs((r0 - rp) - 100.0) < 1e-6


def test_step_returns_gamma_k_and_k():
    env = HighLevelEnv(max_option_steps=2000)
    env.reset(seed=0)
    obs, R, term, trunc, info = env.step(REGION_CENTER)
    assert "gamma_k" in info and "k" in info
    assert info["k"] >= 1
    assert 0.0 < info["gamma_k"] <= 1.0
    # gamma_k == gamma ** k
    assert abs(info["gamma_k"] - env.gamma ** info["k"]) < 1e-9
