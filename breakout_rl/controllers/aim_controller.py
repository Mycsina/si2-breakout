from typing import Optional

from server.logic import Breakout
from breakout_rl.physics.predictor import predict_landing
from breakout_rl.constants import (REGION_LEFT, REGION_CENTER, REGION_RIGHT,
                                   ACTION_NOOP, ACTION_WEST, ACTION_EAST)

_REGION_FRAC = {REGION_LEFT: 1.0 / 6.0, REGION_CENTER: 0.5, REGION_RIGHT: 5.0 / 6.0}
_HALF_STEP = 12.5  # half of the 25px paddle step


def target_paddle_x(game: Breakout, region: int) -> Optional[float]:
    """Paddle_x that makes the ball contact the chosen third of the paddle.
    Returns None while ascending (no aim possible). Falls back to a center intercept
    when the chosen region is infeasible (ball too near a wall)."""
    landing = predict_landing(game)
    if landing is None:
        return None
    w = game.width
    pw = game.paddle_width  # read from game (curriculum may change it)
    tx = landing - _REGION_FRAC[region] * pw
    clamped = min(max(tx, 0.0), w - pw)
    if abs(clamped - tx) > 1e-6:  # region infeasible -> center intercept
        tx = landing - 0.5 * pw
        clamped = min(max(tx, 0.0), w - pw)
    return clamped


def choose_action(game: Breakout, region: int) -> int:
    tx = target_paddle_x(game, region)
    if tx is None:
        return ACTION_NOOP  # hold while ascending
    dx = tx - game.paddle_x
    if dx > _HALF_STEP:
        return ACTION_EAST
    if dx < -_HALF_STEP:
        return ACTION_WEST
    return ACTION_NOOP
