from server.logic import Breakout, Brick


def _copy_brick(b: Brick) -> Brick:
    nb = Brick.__new__(Brick)
    nb.__dict__.update(b.__dict__)
    return nb


def clone_game(game: Breakout) -> Breakout:
    """Return a deep-enough copy of `game` whose mutation never affects the original.

    Built with ``__new__`` to bypass ``Breakout.__init__`` — the constructor rebuilds the
    16 bricks and, crucially, calls ``reset_ball()`` which draws from the global ``random``
    stream. Cloning is done thousands of times per episode (once per ``predict_landing``),
    so going through ``__init__`` both wasted work and *perturbed the RNG that drives the
    real game's paddle bounces*, breaking seed reproducibility. Here we copy state only.

    Only ``active`` flags and the numpy mirror are mutated by ``update()``, so the brick
    objects are shallow-copied (independent ``__dict__``) and the array is copied."""
    c = Breakout.__new__(Breakout)
    c.width = game.width
    c.height = game.height
    c.high_score = game.high_score
    c.paddle_width = game.paddle_width
    c.paddle_height = game.paddle_height
    c.paddle_y = game.paddle_y
    c.paddle_x = game.paddle_x
    c.ball_radius = game.ball_radius
    c.ball_speed = game.ball_speed
    c.ball_x = game.ball_x
    c.ball_y = game.ball_y
    c.ball_vx = game.ball_vx
    c.ball_vy = game.ball_vy
    c.lives = game.lives
    c.score = game.score
    c.checkpoint_score = game.checkpoint_score
    c.game_over = game.game_over
    c.bricks_need_respawn = game.bricks_need_respawn
    c.bricks = [_copy_brick(b) for b in game.bricks]
    c.brick_array = game.brick_array.copy()
    return c
