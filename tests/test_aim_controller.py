from server.logic import Breakout
from breakout_rl.controllers.aim_controller import target_paddle_x, choose_action
from breakout_rl.constants import (REGION_LEFT, REGION_CENTER, REGION_RIGHT,
                                   ACTION_NOOP, ACTION_WEST, ACTION_EAST)


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
