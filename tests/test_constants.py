from breakout_rl import constants as C


def test_action_and_region_enums_distinct():
    assert {C.ACTION_NOOP, C.ACTION_WEST, C.ACTION_EAST} == {0, 1, 2}
    assert {C.REGION_LEFT, C.REGION_CENTER, C.REGION_RIGHT} == {0, 1, 2}


def test_fixed_constants():
    assert abs(C.DT - 1.0 / 30.0) < 1e-9
    assert C.NUM_BRICKS == 16
    assert C.BALL_SPEED_NORM == 300.0
    assert C.LOWEST_BRICK_BOTTOM == 125.0
    assert C.DECISION_LINE_Y == 140.0
