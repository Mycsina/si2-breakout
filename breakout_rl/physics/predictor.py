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
    # If the ball has already reached the paddle line, the landing is here -- return now.
    # This is not just an optimization: rolling a clone PAST the line lets update() take the
    # ball-death branch, whose reset_ball() draws from the GLOBAL random stream. That both
    # perturbs the real game's RNG (breaking seed reproducibility) and makes the rollout
    # non-deterministic (it would return a random respawn's landing). Stopping at the line
    # keeps predict_landing a pure, RNG-safe function of the current state.
    if game.ball_y + game.ball_radius >= game.paddle_y:
        return game.ball_x
    g = clone_game(game)
    # Move the clone's paddle off-screen so the rollout never triggers the *stochastic*
    # paddle bounce (update() draws a random bounce angle from the global RNG when the ball
    # contacts the paddle). The paddle is irrelevant to *where* the ball crosses the line,
    # so removing it leaves the predicted landing identical while keeping the rollout pure
    # and RNG-safe. With a 10px max step and a 20px line->death gap, the crossing check below
    # always fires before update() could reach the (also RNG-drawing) ball-death path.
    g.paddle_x = g.width + 100.0
    for _ in range(_MAX_ROLLOUT_STEPS):
        g.update(DT)
        # paddle-line crossing is checked first: from above the line a single ~10px step
        # cannot skip the ~20px gap to the death line, so we always return at the crossing
        # before update() can ever trigger a death (and thus never touch the RNG).
        if g.ball_y + g.ball_radius >= g.paddle_y:
            return g.ball_x
        if g.game_over:
            return g.ball_x
    return g.ball_x  # safety fallback; should not be reached
