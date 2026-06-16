import numpy as np
from breakout_rl.eval.metrics import evaluate_policy


def test_evaluate_policy_returns_expected_keys():
    # a trivial always-NOOP policy
    def policy(obs):
        return 0
    stats = evaluate_policy(policy, episodes=2, max_steps=200,
                            paddle_width=80.0, ball_speed=300.0, seed=0)
    for k in ["mean_score", "mean_clears", "mean_survival_steps",
              "mean_bricks_per_life", "std_score"]:
        assert k in stats
    assert stats["mean_survival_steps"] > 0
