from typing import Optional
import random
import numpy as np
import gymnasium as gym
from gymnasium import spaces

from server.logic import Breakout
from breakout_rl.env.observation import ObservationBuilder, OBS_DIM
from breakout_rl.env.rewards import RewardConfig, base_reward
from breakout_rl.controllers.aim_controller import choose_action
from breakout_rl.constants import DT, DECISION_LINE_Y, ACTION_WEST, ACTION_EAST

HIGH_OBS_DIM = OBS_DIM  # reuse the shared observation (brick grid + ball + paddle + velocity)


class HighLevelEnv(gym.Env):
    """Semi-Markov (options) wrapper. The high level picks a contact region at each
    decision point (clean descent below the brick field); the low-level aim controller
    executes it. One high-level step = one option = the whole volley until the next
    decision point / life loss / game over. Returns the option-discounted reward R,
    plus gamma**k and k in info for SMDP bootstrapping."""

    metadata = {"render_modes": []}

    def __init__(self, max_option_steps: int = 3000, gamma: float = 0.99,
                 reward_cfg: Optional[RewardConfig] = None,
                 paddle_width: float = 80.0, ball_speed: float = 300.0) -> None:
        super().__init__()
        self.max_option_steps = max_option_steps
        self.gamma = gamma
        self.reward_cfg = reward_cfg or RewardConfig()
        self.curr_paddle_width = paddle_width
        self.curr_ball_speed = ball_speed
        self.game = Breakout()
        self.obs_builder = ObservationBuilder()
        self.action_space = spaces.Discrete(3)  # LEFT, CENTER, RIGHT
        self.observation_space = spaces.Box(-np.inf, np.inf, (HIGH_OBS_DIM,), np.float32)

    def set_curriculum(self, paddle_width: float, ball_speed: float) -> None:
        self.curr_paddle_width = paddle_width
        self.curr_ball_speed = ball_speed

    def _obs(self) -> np.ndarray:
        return self.obs_builder.build(self.game.get_state(), DT)

    def _advance_one_primitive(self, action: int):
        before = self.game.get_state()
        if action == ACTION_WEST:
            self.game.move_paddle("WEST")
        elif action == ACTION_EAST:
            self.game.move_paddle("EAST")
        self.game.update(DT)
        after = self.game.get_state()
        self.obs_builder.build(after, DT)  # keep velocity history warm
        return base_reward(before, after, self.reward_cfg)

    def _at_decision_point(self, prev_ball_y: float) -> bool:
        # decision point = ball crosses the decision line downward, entering the clean
        # descent toward the paddle (this is when committing to a contact region matters).
        return (prev_ball_y <= DECISION_LINE_Y < self.game.ball_y
                and self.game.ball_vy > 0)

    def _advance_to_decision_point(self, region: Optional[int]):
        """Advance with the controller until a new decision point / terminal / life loss.
        region=None means 'no aim yet' (used only during reset bootstrapping)."""
        R, k, discount = 0.0, 0, 1.0
        prev_ball_y = self.game.ball_y
        lives0 = self.game.lives
        while k < self.max_option_steps:
            a = 0 if region is None else choose_action(self.game, region)
            r = self._advance_one_primitive(a)
            R += discount * r
            discount *= self.gamma
            k += 1
            if self.game.game_over:
                return R, k, True, False
            if self.game.lives < lives0:
                return R, k, False, False  # option boundary, not terminal
            if self._at_decision_point(prev_ball_y):
                return R, k, False, False
            prev_ball_y = self.game.ball_y
        return R, k, False, True  # truncated

    def reset(self, *, seed: Optional[int] = None, options=None):
        super().reset(seed=seed)
        if seed is not None:
            random.seed(seed); np.random.seed(seed)
        self.game.reset_game()
        self.game.paddle_width = self.curr_paddle_width
        self.game.ball_speed = self.curr_ball_speed
        self.game.paddle_x = (self.game.width - self.game.paddle_width) / 2.0
        self.game.reset_ball()
        self.obs_builder.reset()
        # advance (NOOP) to the first decision point so step() always starts aimed
        self._advance_to_decision_point(region=None)
        return self._obs(), {}

    def step(self, region: int):
        R, k, terminated, truncated = self._advance_to_decision_point(region)
        info = {"gamma_k": self.gamma ** k, "k": k,
                "score": self.game.score, "bricks_left": int(self.game.brick_array[:, 4].sum())}
        return self._obs(), float(R), terminated, truncated, info
