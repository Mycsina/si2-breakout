import numpy as np
from server.logic import Breakout
from breakout_rl.env.observation import ObservationBuilder, OBS_DIM
from breakout_rl.constants import DT


def _state(**over):
    g = Breakout()
    s = g.get_state()
    s.update(over)
    return s


def test_dimension_and_first_frame_zero_velocity():
    ob = ObservationBuilder()
    v = ob.build(_state(), DT)
    assert v.shape == (OBS_DIM,)
    assert v.dtype == np.float32
    # velocity features (indices 3,4) are zero on the first frame
    assert v[3] == 0.0 and v[4] == 0.0


def test_velocity_uses_measured_dt():
    ob = ObservationBuilder()
    ob.build(_state(ball_x=100.0, ball_y=100.0), DT)
    v = ob.build(_state(ball_x=100.0 + 300.0 * DT, ball_y=100.0), DT)
    # moved exactly ball_speed*dt in x over dt -> normalized vx == 1.0
    assert abs(v[3] - 1.0) < 1e-5
    assert abs(v[4] - 0.0) < 1e-6


def test_brick_occupancy_reconstructed_from_active_only_wire():
    g = Breakout()
    g.bricks[0].active = False
    g.bricks[5].active = False
    s = g.get_state()  # bricks list excludes inactive ones
    ob = ObservationBuilder()
    v = ob.build(s, DT)
    occ = v[-16:]
    assert occ[0] == 0.0 and occ[5] == 0.0
    assert occ[1] == 1.0 and occ[15] == 1.0


def test_reset_clears_velocity_history():
    ob = ObservationBuilder()
    ob.build(_state(ball_x=100.0), DT)
    ob.reset()
    v = ob.build(_state(ball_x=400.0), DT)
    assert v[3] == 0.0  # no carry-over velocity after reset
