from typing import Optional

from server.logic import Breakout
from breakout_rl.physics.clone import clone_game
from breakout_rl.constants import DT

_MAX_ROLLOUT_STEPS = 500


def predict_landing(game: Breakout) -> Optional[float]:
    """Roll the game forward (no paddle input) until the ball reaches the paddle line,
    and return the ball_x at that crossing. Returns None if the ball is ascending.

    Bit-exact with the real environment by construction: it reuses Breakout.update,
    so wall reflections and brick collisions are identical. Deterministic up to the
    paddle line (launch/paddle randomness is not reached). Does not mutate `game`."""
    if game.ball_vy <= 0:
        return None
    g = clone_game(game)
    for _ in range(_MAX_ROLLOUT_STEPS):
        g.update(DT)
        if g.game_over:
            return g.ball_x
        if g.ball_y + g.ball_radius >= g.paddle_y:
            return g.ball_x
    return g.ball_x  # safety fallback; should not be reached
