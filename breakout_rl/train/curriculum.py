from typing import Tuple

STAGE1 = (120.0, 200.0)  # (paddle_width, ball_speed) — easy
STAGE2 = (80.0, 300.0)  # real difficulty


def curriculum_params(step: int, switch_step: int) -> Tuple[float, float]:
    """Two-stage curriculum: easy until `switch_step`, then real settings.
    Stage 2 must run long enough that final eval reflects real difficulty."""
    return STAGE1 if step < switch_step else STAGE2
