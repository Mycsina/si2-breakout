from server.logic import Breakout
from breakout_rl.physics.clone import clone_game

def test_clone_is_independent_and_equal():
    g = Breakout()
    g.ball_x, g.ball_y, g.ball_vx, g.ball_vy = 123.0, 200.0, 50.0, 100.0
    g.paddle_x, g.score, g.lives = 77.0, 42, 2
    g.bricks[0].active = False
    g._sync_bricks_to_numpy()

    c = clone_game(g)
    # equal snapshot
    assert (c.ball_x, c.ball_y, c.ball_vx, c.ball_vy) == (123.0, 200.0, 50.0, 100.0)
    assert (c.paddle_x, c.score, c.lives) == (77.0, 42, 2)
    assert c.bricks[0].active is False
    assert c.brick_array[0, 4] == 0.0

    # independence: mutating clone does not touch original
    c.ball_x = -999.0
    c.bricks[1].active = False
    assert g.ball_x == 123.0
    assert g.bricks[1].active is True

def test_clone_preserves_curriculum_overrides():
    g = Breakout()
    g.paddle_width = 120.0
    g.ball_speed = 200.0
    c = clone_game(g)
    assert c.paddle_width == 120.0
    assert c.ball_speed == 200.0
