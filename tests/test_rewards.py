import math
from breakout_rl.env.rewards import base_reward, potential, RewardConfig


def test_brick_score_delta_is_positive_reward():
    before = {"lives": 3, "score": 0, "game_over": False}
    after = {"lives": 3, "score": 3, "game_over": False}
    cfg = RewardConfig(score_scale=1.0, step_cost=0.0)
    assert base_reward(before, after, cfg) == 3.0


def test_life_loss_uses_penalty_not_score_delta():
    before = {"lives": 3, "score": 75, "game_over": False}
    after = {"lives": 2, "score": 50, "game_over": False}  # score reset to checkpoint
    cfg = RewardConfig(life_loss_penalty=-30.0, step_cost=0.0)
    assert base_reward(before, after, cfg) == -30.0


def test_game_over_adds_terminal_penalty():
    before = {"lives": 1, "score": 10, "game_over": False}
    after = {"lives": 0, "score": 10, "game_over": True}
    cfg = RewardConfig(life_loss_penalty=-30.0, game_over_penalty=-50.0, step_cost=0.0)
    assert base_reward(before, after, cfg) == -80.0


def test_potential_zero_when_ascending_or_terminal():
    s = {"width": 600, "paddle_x": 100.0, "paddle_width": 80.0}
    assert potential(s, None, is_terminal=False) == 0.0          # ascending -> landing None
    assert potential(s, 300.0, is_terminal=True) == 0.0          # terminal


def test_potential_is_negative_normalized_distance():
    s = {"width": 600, "paddle_x": 100.0, "paddle_width": 80.0}  # center = 140
    # landing at 200 -> distance 60 -> -60/600
    assert abs(potential(s, 200.0, is_terminal=False) - (-60.0 / 600.0)) < 1e-9


def test_pbrs_discounted_shaping_telescopes_to_terminal_minus_start():
    # F_t = gamma*phi(s_{t+1}) - phi(s_t); sum_t gamma^t F_t == gamma^T*phi_T - phi_0
    gamma = 0.99
    phis = [-0.5, -0.3, -0.1, 0.0]  # phi_T = 0 (terminal)
    total = 0.0
    for t in range(len(phis) - 1):
        f = gamma * phis[t + 1] - phis[t]
        total += (gamma ** t) * f
    expected = (gamma ** (len(phis) - 1)) * phis[-1] - phis[0]
    assert abs(total - expected) < 1e-12
