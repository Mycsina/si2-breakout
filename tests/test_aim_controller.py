from server.logic import Breakout
from breakout_rl.controllers.aim_controller import (
    target_paddle_x,
    choose_action,
    brick_mass_offset,
)
from breakout_rl.constants import REGION_CENTER, REGION_RIGHT, ACTION_EAST


def _descending_midboard():
    g = Breakout()
    for b in g.bricks:
        b.active = False
    g._sync_bricks_to_numpy()
    g.ball_x, g.ball_y, g.ball_vx, g.ball_vy = 300.0, 250.0, 0.0, 150.0
    return g


def test_target_none_when_ascending():
    g = _descending_midboard()
    g.ball_vy = -150.0
    assert target_paddle_x(g, REGION_CENTER) is None


def test_center_region_centers_paddle_on_landing():
    g = _descending_midboard()
    tx = target_paddle_x(g, REGION_CENTER)
    # ball goes straight down (vx=0) -> landing ~300; paddle center at 300 -> tx ~ 300 - 40
    assert tx is not None
    assert abs(tx - (300.0 - g.paddle_width / 2.0)) < 5.0


def test_choose_action_moves_toward_target():
    g = _descending_midboard()
    g.paddle_x = 0.0  # far left; target is near center -> should move EAST
    assert choose_action(g, REGION_CENTER) == ACTION_EAST


def test_region_infeasible_near_wall_falls_back_to_center():
    g = _descending_midboard()
    g.ball_x, g.ball_vx = 30.0, 0.0  # lands near left wall
    tx = target_paddle_x(g, REGION_RIGHT)  # right region impossible near left wall
    assert tx == 0.0  # clamped fallback intercept at left wall


def test_brick_mass_offset_zero_when_no_bricks():
    g = Breakout()
    for b in g.bricks:
        b.active = False
    g._sync_bricks_to_numpy()
    assert brick_mass_offset(g) == 0.0


def test_brick_mass_offset_points_toward_surviving_bricks():
    g = Breakout()
    # keep only a far-right brick (index 4: x 425..495, center 460); ball on the left
    for b in g.bricks:
        b.active = b.index == 4
    g._sync_bricks_to_numpy()
    g.ball_x = 100.0
    off_right = brick_mass_offset(g)
    assert off_right > 0.0  # bricks are to the right -> aim right

    # mirror: keep only a far-left brick (index 5: x 65..135, center 100); ball on the right
    for b in g.bricks:
        b.active = b.index == 5
    g._sync_bricks_to_numpy()
    g.ball_x = 500.0
    assert brick_mass_offset(g) < 0.0  # bricks to the left -> aim left
