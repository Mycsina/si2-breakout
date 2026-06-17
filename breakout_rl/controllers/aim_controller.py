from typing import Optional

from server.logic import Breakout
from breakout_rl.physics.predictor import predict_landing
from breakout_rl.constants import (
    REGION_LEFT,
    REGION_CENTER,
    REGION_RIGHT,
    ACTION_NOOP,
    ACTION_WEST,
    ACTION_EAST,
)

_REGION_FRAC = {REGION_LEFT: 1.0 / 6.0, REGION_CENTER: 0.5, REGION_RIGHT: 5.0 / 6.0}
_HALF_STEP = 12.5  # half of the 25px paddle step


def brick_mass_offset(game: Breakout) -> float:
    """Signed horizontal offset of the surviving-brick centroid from the ball, normalized
    by board width. >0 means most remaining bricks are to the RIGHT of the ball (so aim
    right), <0 to the left, ~0 when balanced or none left. This is the 'which way are the
    bricks' signal the high-level policy needs to aim proactively in the endgame, instead
    of relying on corner-ricochet setups. Computed identically in training and deploy."""
    arr = game.brick_array
    if arr.shape[0] == 0:
        return 0.0
    active = arr[:, 4] == 1.0
    if not active.any():
        return 0.0
    centers_x = (arr[active, 0] + arr[active, 2]) * 0.5
    centroid = float(centers_x.mean())
    return float((centroid - game.ball_x) / game.width)


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


def action_for_target(paddle_x: float, target_x: float) -> int:
    """Primitive action that nudges the paddle toward ``target_x`` (a 25px step, so we
    use a half-step deadband to avoid oscillating around the target)."""
    dx = target_x - paddle_x
    if dx > _HALF_STEP:
        return ACTION_EAST
    if dx < -_HALF_STEP:
        return ACTION_WEST
    return ACTION_NOOP


def choose_action(game: Breakout, region: int) -> int:
    tx = target_paddle_x(game, region)
    if tx is None:
        return ACTION_NOOP  # hold while ascending
    return action_for_target(game.paddle_x, tx)
