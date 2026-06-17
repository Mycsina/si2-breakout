import random

from breakout_rl.agents.planner import MCPlanner
from breakout_rl.env.high_level_env import HighLevelEnv
from breakout_rl.constants import REGION_LEFT, REGION_CENTER, REGION_RIGHT


def test_plan_returns_a_valid_region():
    env = HighLevelEnv()
    env.reset(seed=0)  # reset advances to the first decision point
    planner = MCPlanner(n_samples=4)
    region = planner.plan(env.game)
    assert region in (REGION_LEFT, REGION_CENTER, REGION_RIGHT)


def test_plan_does_not_perturb_the_global_rng_stream():
    # the real episode's bounces must be reproducible regardless of how much the planner
    # samples, so plan() must leave the global random stream exactly as it found it.
    env = HighLevelEnv()
    env.reset(seed=1)
    planner = MCPlanner(n_samples=8)
    before = random.getstate()
    planner.plan(env.game)
    after = random.getstate()
    assert before == after


def test_expected_broken_is_nonnegative():
    env = HighLevelEnv()
    env.reset(seed=2)
    planner = MCPlanner(n_samples=4)
    for region in (REGION_LEFT, REGION_CENTER, REGION_RIGHT):
        assert planner._expected_broken(env.game, region) >= 0.0
