from dataclasses import dataclass
from typing import Optional


@dataclass
class RewardConfig:
    score_scale: float = 1.0
    life_loss_penalty: float = -30.0
    game_over_penalty: float = -50.0
    step_cost: float = -0.01


def base_reward(before: dict, after: dict, cfg: RewardConfig) -> float:
    """Event reward computed from get_state() dicts captured before/after one update.
    On a non-death step the score only increases (+3/brick, +100/clear), so the score
    delta is a clean positive signal. On a death step the score resets to checkpoint,
    so we branch on the lives delta and ignore the (negative) score delta."""
    r = cfg.step_cost
    if after["lives"] < before["lives"]:
        r += cfg.life_loss_penalty
        if after["game_over"]:
            r += cfg.game_over_penalty
    else:
        r += (after["score"] - before["score"]) * cfg.score_scale
    return r


def potential(state: dict, predicted_landing_x: Optional[float], is_terminal: bool) -> float:
    """PBRS potential. MUST be a pure function of state, and MUST be 0 on terminal
    states for policy invariance. `predicted_landing_x` is None when ascending (the
    predictor returns None), which yields potential 0 -> a valid state function."""
    if is_terminal or predicted_landing_x is None:
        return 0.0
    w = float(state["width"])
    paddle_center = float(state["paddle_x"]) + float(state["paddle_width"]) / 2.0
    return -abs(paddle_center - predicted_landing_x) / w
