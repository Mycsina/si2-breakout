from server.logic import Breakout


def clone_game(game: Breakout) -> Breakout:
    """Return a deep-enough copy of `game` whose mutation never affects the original.
    Geometry of bricks (positions/sizes) is immutable and shared; only active flags
    and the numpy mirror are copied."""
    c = Breakout(width=game.width, height=game.height)
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
    c.high_score = game.high_score
    c.game_over = game.game_over
    c.bricks_need_respawn = game.bricks_need_respawn
    for src, dst in zip(game.bricks, c.bricks):
        dst.active = src.active
    c.brick_array = game.brick_array.copy()
    return c
