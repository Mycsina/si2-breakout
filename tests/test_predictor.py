from server.logic import Breakout
from breakout_rl.physics.clone import clone_game
from breakout_rl.physics.predictor import predict_landing
from breakout_rl.constants import DT


def test_returns_none_when_ascending():
    g = Breakout()
    g.ball_y, g.ball_vx, g.ball_vy = 200.0, 0.0, -100.0
    assert predict_landing(g) is None


def test_matches_manual_rollout_no_bricks_in_path():
    g = Breakout()
    # clear bricks so the descent is clean ballistics
    for b in g.bricks:
        b.active = False
    g._sync_bricks_to_numpy()
    g.ball_x, g.ball_y, g.ball_vx, g.ball_vy = 300.0, 200.0, 40.0, 150.0

    pred = predict_landing(g)

    # manual reference rollout on an independent clone
    ref = clone_game(g)
    for _ in range(500):
        ref.update(DT)
        if ref.ball_y + ref.ball_radius >= ref.paddle_y or ref.game_over:
            break
    assert pred is not None
    assert abs(pred - ref.ball_x) < 1e-6


def test_does_not_mutate_input_game():
    g = Breakout()
    g.ball_x, g.ball_y, g.ball_vx, g.ball_vy = 300.0, 200.0, 40.0, 150.0
    before = (g.ball_x, g.ball_y, g.ball_vx, g.ball_vy)
    predict_landing(g)
    assert (g.ball_x, g.ball_y, g.ball_vx, g.ball_vy) == before
