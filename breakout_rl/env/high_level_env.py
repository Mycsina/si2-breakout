from typing import Optional
import random
import numpy as np
import gymnasium as gym
from gymnasium import spaces

from server.logic import Breakout
from breakout_rl.env.observation import ObservationBuilder, OBS_DIM
from breakout_rl.env.rewards import RewardConfig, base_reward
from breakout_rl.controllers.aim_controller import (
    target_paddle_x,
    action_for_target,
    brick_mass_offset,
)
from breakout_rl.constants import (
    DT,
    DECISION_LINE_Y,
    ACTION_NOOP,
    ACTION_WEST,
    ACTION_EAST,
)

# Shared 23-dim observation (brick grid + ball + paddle + velocity) plus one extra
# high-level feature: the signed offset of the surviving-brick centroid from the ball,
# so the policy can aim toward where bricks remain (see brick_mass_offset).
HIGH_OBS_DIM = OBS_DIM + 1


def crossed_decision_line(prev_ball_y: float, ball_y: float, ball_vy: float) -> bool:
    """A high-level decision point: the ball crosses the decision line downward, entering
    the clean descent toward the paddle (committing to a contact region only matters here).
    Shared by HighLevelEnv (training/eval) and the deploy agent so the two trigger on the
    exact same event and can never silently diverge."""
    return prev_ball_y <= DECISION_LINE_Y < ball_y and ball_vy > 0


class HighLevelEnv(gym.Env):
    """Semi-Markov (options) wrapper. The high level picks a contact region at each
    decision point (clean descent below the brick field); the low-level aim controller
    executes it. One high-level step = one option = the whole volley until the next
    decision point / life loss / game over. Returns the option-discounted reward R,
    plus gamma**k and k in info for SMDP bootstrapping."""

    metadata = {"render_modes": []}

    def __init__(
        self,
        max_option_steps: int = 3000,
        gamma: float = 0.99,
        reward_cfg: Optional[RewardConfig] = None,
        paddle_width: float = 80.0,
        ball_speed: float = 300.0,
    ) -> None:
        super().__init__()
        self.max_option_steps = max_option_steps
        self.gamma = gamma
        self.reward_cfg = reward_cfg or RewardConfig()
        self.curr_paddle_width = paddle_width
        self.curr_ball_speed = ball_speed
        self.game = Breakout()
        self.obs_builder = ObservationBuilder()
        self.action_space = spaces.Discrete(3)  # LEFT, CENTER, RIGHT
        self.observation_space = spaces.Box(
            -np.inf, np.inf, (HIGH_OBS_DIM,), np.float32
        )

    def set_curriculum(self, paddle_width: float, ball_speed: float) -> None:
        self.curr_paddle_width = paddle_width
        self.curr_ball_speed = ball_speed

    def _obs(self) -> np.ndarray:
        # Base = the observation built from the most recent primitive frame. We must NOT
        # rebuild from the current state here: the builder is stateful and already consumed
        # this frame in _advance_one_primitive, so rebuilding would diff a frame against
        # itself and zero out the recovered ball velocity. We append the brick-centroid
        # offset (a pure function of the current state, cheap, computed once per option).
        return np.concatenate([self._last_obs, [brick_mass_offset(self.game)]]).astype(
            np.float32
        )

    def _advance_one_primitive(self, action: int):
        before = self.game.get_state()
        if action == ACTION_WEST:
            self.game.move_paddle("WEST")
        elif action == ACTION_EAST:
            self.game.move_paddle("EAST")
        self.game.update(DT)
        after = self.game.get_state()
        self._last_obs = self.obs_builder.build(
            after, DT
        )  # advance velocity history + keep obs
        return base_reward(before, after, self.reward_cfg)

    def _at_decision_point(self, prev_ball_y: float) -> bool:
        return crossed_decision_line(prev_ball_y, self.game.ball_y, self.game.ball_vy)

    def _advance_to_decision_point(self, region: Optional[int]):
        """Advance with the controller until a new decision point / terminal / life loss.
        region=None means 'no aim yet' (used only during reset bootstrapping).

        The landing target is constant during a clean descent (no bricks below the
        decision line), so it is computed once per descent and reused — this avoids
        re-running predict_landing every primitive step (an O(descent_len^2) waste).
        Behaviour is identical to recomputing because the target only depends on the
        ball's (constant) landing x and the paddle width, not the moving paddle_x."""
        R, k, discount = 0.0, 0, 1.0
        prev_ball_y = self.game.ball_y
        lives0 = self.game.lives
        cached_target: Optional[float] = None
        bricks_at_cache = -1
        prev_bricks = int(self.game.brick_array[:, 4].sum())
        broken = 0  # bricks broken during this option (respawn jumps are not counted)
        while k < self.max_option_steps:
            if region is None:
                a = ACTION_NOOP
            elif self.game.ball_vy <= 0:
                cached_target = None  # ascending: no aim possible, invalidate cache
                a = ACTION_NOOP
            else:
                bricks_now = int(self.game.brick_array[:, 4].sum())
                if cached_target is None or bricks_now != bricks_at_cache:
                    cached_target = target_paddle_x(self.game, region)
                    bricks_at_cache = bricks_now
                a = (
                    ACTION_NOOP
                    if cached_target is None
                    else action_for_target(self.game.paddle_x, cached_target)
                )
            r = self._advance_one_primitive(a)
            R += discount * r
            discount *= self.gamma
            k += 1
            now_bricks = int(self.game.brick_array[:, 4].sum())
            if now_bricks < prev_bricks:  # ignore respawn jumps (now > prev)
                broken += prev_bricks - now_bricks
            prev_bricks = now_bricks
            if self.game.game_over:
                return R, k, True, False, broken
            if self.game.lives < lives0:
                return R, k, False, False, broken  # option boundary, not terminal
            if self._at_decision_point(prev_ball_y):
                return R, k, False, False, broken
            prev_ball_y = self.game.ball_y
        return R, k, False, True, broken  # truncated

    def reset(self, *, seed: Optional[int] = None, options=None):
        super().reset(seed=seed)
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)
        self.game.reset_game()
        self.game.paddle_width = self.curr_paddle_width
        self.game.ball_speed = self.curr_ball_speed
        self.game.paddle_x = (self.game.width - self.game.paddle_width) / 2.0
        self.game.reset_ball()
        self.obs_builder.reset()
        self._last_obs = self.obs_builder.build(
            self.game.get_state(), DT
        )  # initial frame (v=0)
        # advance (NOOP) to the first decision point so step() always starts aimed
        self._advance_to_decision_point(region=None)
        return self._obs(), {}

    def step(self, region: int):
        R, k, terminated, truncated, broken = self._advance_to_decision_point(region)
        if broken == 0:  # unproductive volley: nudge the policy to aim at the bricks
            R += self.reward_cfg.waste_penalty
        info = {
            "gamma_k": self.gamma**k,
            "k": k,
            "score": self.game.score,
            "bricks_left": int(self.game.brick_array[:, 4].sum()),
            "broken": broken,
        }
        return self._obs(), float(R), terminated, truncated, info
