from typing import Optional
import numpy as np

from breakout_rl.constants import NUM_BRICKS, BALL_SPEED_NORM

OBS_DIM = 7 + NUM_BRICKS  # 23


class ObservationBuilder:
    """Builds the policy observation from a get_state() dict ONLY (the WebSocket wire
    format), so the training env and the deployed agent produce identical vectors.
    Velocity is recovered by finite difference over the *measured* dt."""

    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        self._prev_ball_x: Optional[float] = None
        self._prev_ball_y: Optional[float] = None

    def build(self, state: dict, dt: float) -> np.ndarray:
        w = float(state["width"])
        h = float(state["height"])
        pw = float(state["paddle_width"])
        px = float(state["paddle_x"])
        bx = float(state["ball_x"])
        by = float(state["ball_y"])

        if self._prev_ball_x is None or dt <= 0:
            vx = 0.0
            vy = 0.0
        else:
            vx = ((bx - self._prev_ball_x) / dt) / BALL_SPEED_NORM
            vy = ((by - self._prev_ball_y) / dt) / BALL_SPEED_NORM
        self._prev_ball_x = bx
        self._prev_ball_y = by

        occ = np.zeros(NUM_BRICKS, dtype=np.float32)
        for b in state.get("bricks", []):
            idx = int(b["index"])
            if 0 <= idx < NUM_BRICKS:
                occ[idx] = 1.0

        paddle_range = max(w - pw, 1.0)
        scalars = np.array([
            px / paddle_range,
            bx / w,
            by / h,
            vx,
            vy,
            float(state["lives"]) / 3.0,
            float(occ.sum()) / NUM_BRICKS,
        ], dtype=np.float32)
        return np.concatenate([scalars, occ]).astype(np.float32)
