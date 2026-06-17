from typing import Optional
import random
import numpy as np
import gymnasium as gym
from gymnasium import spaces

from server.logic import Breakout
from breakout_rl.env.observation import ObservationBuilder, OBS_DIM
from breakout_rl.env.rewards import RewardConfig, base_reward, potential
from breakout_rl.physics.predictor import predict_landing
from breakout_rl.constants import ACTION_WEST, ACTION_EAST, DT


class BreakoutEnv(gym.Env):
    """Headless Gymnasium wrapper around server.logic.Breakout.
    One step = apply action -> game.update(DT) -> base reward + PBRS shaping."""

    metadata = {"render_modes": []}

    def __init__(
        self,
        max_steps: int = 4000,
        gamma: float = 0.99,
        reward_cfg: Optional[RewardConfig] = None,
        paddle_width: float = 80.0,
        ball_speed: float = 300.0,
    ) -> None:
        super().__init__()
        self.max_steps = max_steps
        self.gamma = gamma
        self.reward_cfg = reward_cfg or RewardConfig()
        self.curr_paddle_width = paddle_width
        self.curr_ball_speed = ball_speed

        self.game = Breakout()
        self.obs_builder = ObservationBuilder()
        self.action_space = spaces.Discrete(3)
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(OBS_DIM,), dtype=np.float32
        )
        self._prev_state_for_test: dict = {}

    def set_curriculum(self, paddle_width: float, ball_speed: float) -> None:
        self.curr_paddle_width = paddle_width
        self.curr_ball_speed = ball_speed

    def _build_obs(self, state: dict) -> np.ndarray:
        return self.obs_builder.build(state, DT)

    def reset(self, *, seed: Optional[int] = None, options=None):
        super().reset(seed=seed)
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)
        self.game.reset_game()
        # apply curriculum AFTER reset_game (which hard-codes defaults), then relaunch
        self.game.paddle_width = self.curr_paddle_width
        self.game.ball_speed = self.curr_ball_speed
        self.game.paddle_x = (self.game.width - self.game.paddle_width) / 2.0
        self.game.reset_ball()

        self.obs_builder.reset()
        self.steps = 0
        # descent cache: predict_landing is constant during a clean descent, so we recompute
        # it only when the descent (re)starts or a brick changes — exact, but avoids the
        # O(descent_len^2) cost of re-rolling the predictor every descending step.
        self._cached_landing = None
        self._bricks_at_cache = int(self.game.brick_array[:, 4].sum())
        self._last_vy = self.game.ball_vy
        state = self.game.get_state()
        self._prev_state_for_test = state
        landing = predict_landing(self.game)
        self.prev_phi = potential(state, landing, is_terminal=False)
        return self._build_obs(state), {}

    def step(self, action: int):
        self._prev_state_for_test = self.game.get_state()
        if action == ACTION_WEST:
            self.game.move_paddle("WEST")
        elif action == ACTION_EAST:
            self.game.move_paddle("EAST")
        before = self.game.get_state()
        self.game.update(DT)
        after = self.game.get_state()

        terminated = bool(self.game.game_over)
        base = base_reward(before, after, self.reward_cfg)

        descending = self.game.ball_vy > 0
        bricks_now = int(self.game.brick_array[:, 4].sum())
        if not descending:
            self._cached_landing = None
        else:
            # The "landing is constant during a clean descent" premise only holds while the
            # ball is ABOVE the paddle line. Once ball_y + r >= paddle_y, predict_landing
            # degenerates to a ~1-step rollout (~current ball_x) as the ball falls past a
            # missed paddle, so a cached long-descent value would be stale. Recompute there
            # (it's cheap) and whenever the descent (re)starts or a brick changes -> stays
            # bit-exact with per-step prediction.
            near_paddle = self.game.ball_y + self.game.ball_radius >= self.game.paddle_y
            if (
                self._cached_landing is None
                or bricks_now != self._bricks_at_cache
                or self._last_vy <= 0
                or near_paddle
            ):
                self._cached_landing = predict_landing(self.game)
                self._bricks_at_cache = bricks_now
        self._last_vy = self.game.ball_vy
        landing = self._cached_landing
        cur_phi = potential(after, landing, is_terminal=terminated)
        reward = base + self.gamma * cur_phi - self.prev_phi
        self.prev_phi = 0.0 if terminated else cur_phi

        self.steps += 1
        truncated = self.steps >= self.max_steps
        obs = self._build_obs(after)
        info = {
            "score": self.game.score,
            "lives": self.game.lives,
            "bricks_left": int(self.game.brick_array[:, 4].sum()),
        }
        return obs, float(reward), terminated, truncated, info
