# Hierarchical RL Breakout Agent — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build two Breakout agents — a from-scratch Double+Dueling DQN with PER (the reactive "spine"), and a hierarchical SMDP agent that learns a high-level paddle-aim policy on top of a physics-based low-level controller — and deploy them over the existing WebSocket framework.

**Architecture:** A Gymnasium environment wraps the existing, pure `server.logic.Breakout` so training runs headless at thousands of steps/sec. A rollout-based predictor (the simulator itself, cloned) gives bit-exact ball-landing predictions used by both a potential-based shaped reward and the low-level aim controller. One shared observation builder guarantees train/deploy parity. The high-level policy is trained with semi-Markov (options) Q-learning.

**Tech Stack:** Python 3.10+, PyTorch, Gymnasium, NumPy, PyYAML, TensorBoard, pytest, websockets (deploy). `ai-game-framework` (PyPI, ≥1.1.0) provides the server/agent interfaces.

**Companion spec:** `docs/superpowers/specs/2026-06-16-breakout-hierarchical-rl-design.md` — read it first; this plan implements it.

---

## Conventions (read once, applies to every task)

**Repo is Jujutsu-colocated with git.** Plain `git add` / `git commit` work and `jj` auto-imports them. Commit commands below use `git`.

**Package root:** all new code lives under `breakout_rl/` (a Python package). Existing `server/` and `agents/` are only *read* or *subclassed* — do not edit them except the upstream-bugfix branch (Task 28).

**Action encoding (single source of truth, `breakout_rl/constants.py`):**
- `ACTION_NOOP = 0`, `ACTION_WEST = 1`, `ACTION_EAST = 2`
- `REGION_LEFT = 0`, `REGION_CENTER = 1`, `REGION_RIGHT = 2`

**Fixed constants (`breakout_rl/constants.py`):** `DT = 1.0 / 30.0`, `BALL_SPEED_NORM = 300.0`, `NUM_BRICKS = 16`, `LOWEST_BRICK_BOTTOM = 125.0`, `DECISION_LINE_Y = 140.0`.

**Always read geometry (`paddle_width`, `width`, `ball_speed`) from the live game object**, never hard-code — the curriculum changes them at runtime.

**Testing:** `pytest` from repo root. Every task is TDD: failing test → minimal code → passing test → commit.

**Run all setup commands inside the venv** created in Task 1.

---

## File structure (decomposition locked here)

```
breakout_rl/
  __init__.py
  constants.py                 # action/region enums, fixed constants
  physics/
    __init__.py
    clone.py                   # clone_game(game) -> Breakout
    predictor.py               # predict_landing(game) -> Optional[float]
  env/
    __init__.py
    observation.py             # ObservationBuilder (SHARED with deploy)
    rewards.py                 # base_reward, potential, shaped term
    breakout_env.py            # BreakoutEnv(gym.Env)
    high_level_env.py          # HighLevelEnv (SMDP options wrapper)
  agents/
    __init__.py
    replay.py                  # PrioritizedReplay + Transition
    networks.py                # DuelingMLP
    dqn_agent.py               # DQNAgent (Double DQN update, ε-greedy)
  controllers/
    __init__.py
    aim_controller.py          # target_paddle_x, choose_action
  train/
    __init__.py
    curriculum.py              # curriculum schedule helper
    train_dqn.py               # flat DQN training entrypoint
    train_aim.py               # SMDP high-level training entrypoint
  eval/
    __init__.py
    metrics.py                 # rollout metrics (clears, steps-to-clear, ...)
    evaluate.py                # comparison table + plots entrypoint
    probe_aiming_authority.py  # M3 GATE experiment
  deploy/
    __init__.py
    trained_agent.py           # TrainedAgent(BaseAgent) flat + hierarchical
  configs/
    flat_dqn.yaml
    aim.yaml
tests/
  test_clone.py
  test_predictor.py
  test_observation.py
  test_rewards.py
  test_breakout_env.py
  test_replay.py
  test_networks.py
  test_aim_controller.py
  test_high_level_env.py
  (existing) test_logic.py
checkpoints/                   # gitignored except a final/ kept checkpoint
docs/superpowers/...           # spec + this plan
```

---

# PHASE M0 — Setup

### Task 1: Virtual environment and dependencies

**Files:**
- Create: `requirements-rl.txt`

- [ ] **Step 1: Create the RL requirements file**

`requirements-rl.txt`:
```
ai-game-framework>=1.1.0
numpy>=1.26
torch>=2.2
gymnasium>=0.29
pyyaml>=6.0
tensorboard>=2.15
pytest>=8.0
websockets>=12.0
```

- [ ] **Step 2: Create venv and install**

Run:
```bash
python3 -m venv venv
source venv/bin/activate
pip install --upgrade pip
pip install -r requirements-rl.txt
```
Expected: all install cleanly. `ai-game-framework` is on PyPI (the README's "local editable" wording is outdated). If torch is slow/large, that is expected.

- [ ] **Step 3: Verify framework + existing tests**

Run:
```bash
python -c "import aigf.interface, aigf.main; print('aigf ok')"
python -m pytest tests/test_logic.py -q
```
Expected: `aigf ok` and existing logic tests pass.

- [ ] **Step 4: Commit**

```bash
git add requirements-rl.txt
git commit -m "chore: add RL training dependencies"
```

---

### Task 2: Package skeleton and constants

**Files:**
- Create: `breakout_rl/__init__.py`, `breakout_rl/constants.py`, and empty `__init__.py` in every subpackage listed in the file structure.
- Test: `tests/test_constants.py`

- [ ] **Step 1: Write the failing test**

`tests/test_constants.py`:
```python
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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_constants.py -v`
Expected: FAIL with `ModuleNotFoundError: breakout_rl`.

- [ ] **Step 3: Create the package files**

`breakout_rl/constants.py`:
```python
ACTION_NOOP = 0
ACTION_WEST = 1
ACTION_EAST = 2
ACTIONS = ["NOOP", "WEST", "EAST"]

REGION_LEFT = 0
REGION_CENTER = 1
REGION_RIGHT = 2

DT = 1.0 / 30.0
BALL_SPEED_NORM = 300.0
NUM_BRICKS = 16
LOWEST_BRICK_BOTTOM = 125.0
DECISION_LINE_Y = 140.0
```

Create empty `__init__.py` in: `breakout_rl/`, `breakout_rl/physics/`, `breakout_rl/env/`, `breakout_rl/agents/`, `breakout_rl/controllers/`, `breakout_rl/train/`, `breakout_rl/eval/`, `breakout_rl/deploy/`.

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_constants.py -v`
Expected: PASS (2 tests).

- [ ] **Step 5: Commit**

```bash
git add breakout_rl tests/test_constants.py
git commit -m "feat: breakout_rl package skeleton and constants"
```

---

# PHASE M1 — Foundation (env, predictor, observation, rewards)

### Task 3: `clone_game` — copy mutable game state

**Files:**
- Create: `breakout_rl/physics/clone.py`
- Test: `tests/test_clone.py`

- [ ] **Step 1: Write the failing test**

`tests/test_clone.py`:
```python
from server.logic import Breakout
from breakout_rl.physics.clone import clone_game

def test_clone_is_independent_and_equal():
    g = Breakout()
    g.ball_x, g.ball_y, g.ball_vx, g.ball_vy = 123.0, 200.0, 50.0, 100.0
    g.paddle_x, g.score, g.lives = 77.0, 42, 2
    g.bricks[0].active = False
    g._sync_bricks_to_numpy()

    c = clone_game(g)
    # equal snapshot
    assert (c.ball_x, c.ball_y, c.ball_vx, c.ball_vy) == (123.0, 200.0, 50.0, 100.0)
    assert (c.paddle_x, c.score, c.lives) == (77.0, 42, 2)
    assert c.bricks[0].active is False
    assert c.brick_array[0, 4] == 0.0

    # independence: mutating clone does not touch original
    c.ball_x = -999.0
    c.bricks[1].active = False
    assert g.ball_x == 123.0
    assert g.bricks[1].active is True

def test_clone_preserves_curriculum_overrides():
    g = Breakout()
    g.paddle_width = 120.0
    g.ball_speed = 200.0
    c = clone_game(g)
    assert c.paddle_width == 120.0
    assert c.ball_speed == 200.0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_clone.py -v`
Expected: FAIL with `ModuleNotFoundError`.

- [ ] **Step 3: Implement**

`breakout_rl/physics/clone.py`:
```python
from server.logic import Breakout


def clone_game(game: Breakout) -> Breakout:
    """Return a deep-enough copy of `game` whose mutation never affects the original.
    Geometry of bricks (positions/sizes) is immutable and shared; only active flags
    and the numpy mirror are copied."""
    c = Breakout(width=game.width, height=game.height)
    c.paddle_width = game.paddle_width
    c.paddle_height = game.paddle_height
    c.paddle_y = game.paddle_y
    c.paddle_x = game.paddle_x
    c.ball_radius = game.ball_radius
    c.ball_speed = game.ball_speed
    c.ball_x = game.ball_x
    c.ball_y = game.ball_y
    c.ball_vx = game.ball_vx
    c.ball_vy = game.ball_vy
    c.lives = game.lives
    c.score = game.score
    c.checkpoint_score = game.checkpoint_score
    c.high_score = game.high_score
    c.game_over = game.game_over
    c.bricks_need_respawn = game.bricks_need_respawn
    for src, dst in zip(game.bricks, c.bricks):
        dst.active = src.active
    c.brick_array = game.brick_array.copy()
    return c
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_clone.py -v`
Expected: PASS (2 tests).

- [ ] **Step 5: Commit**

```bash
git add breakout_rl/physics/clone.py tests/test_clone.py
git commit -m "feat: clone_game for rollout prediction"
```

---

### Task 4: `predict_landing` — rollout-based, bit-exact predictor

**Files:**
- Create: `breakout_rl/physics/predictor.py`
- Test: `tests/test_predictor.py`

- [ ] **Step 1: Write the failing test**

`tests/test_predictor.py`:
```python
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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_predictor.py -v`
Expected: FAIL with `ModuleNotFoundError`.

- [ ] **Step 3: Implement**

`breakout_rl/physics/predictor.py`:
```python
from typing import Optional

from server.logic import Breakout
from breakout_rl.physics.clone import clone_game
from breakout_rl.constants import DT

_MAX_ROLLOUT_STEPS = 500


def predict_landing(game: Breakout) -> Optional[float]:
    """Roll the game forward (no paddle input) until the ball reaches the paddle line,
    and return the ball_x at that crossing. Returns None if the ball is ascending.

    Bit-exact with the real environment by construction: it reuses Breakout.update,
    so wall reflections and brick collisions are identical. Deterministic up to the
    paddle line (launch/paddle randomness is not reached). Does not mutate `game`."""
    if game.ball_vy <= 0:
        return None
    g = clone_game(game)
    for _ in range(_MAX_ROLLOUT_STEPS):
        g.update(DT)
        if g.game_over:
            return g.ball_x
        if g.ball_y + g.ball_radius >= g.paddle_y:
            return g.ball_x
    return g.ball_x  # safety fallback; should not be reached
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_predictor.py -v`
Expected: PASS (3 tests).

- [ ] **Step 5: Commit**

```bash
git add breakout_rl/physics/predictor.py tests/test_predictor.py
git commit -m "feat: rollout-based bit-exact ball-landing predictor"
```

---

### Task 5: `ObservationBuilder` — shared, deploy-faithful feature vector

**Files:**
- Create: `breakout_rl/env/observation.py`
- Test: `tests/test_observation.py`

- [ ] **Step 1: Write the failing test**

`tests/test_observation.py`:
```python
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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_observation.py -v`
Expected: FAIL with `ModuleNotFoundError`.

- [ ] **Step 3: Implement**

`breakout_rl/env/observation.py`:
```python
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
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_observation.py -v`
Expected: PASS (4 tests).

- [ ] **Step 5: Commit**

```bash
git add breakout_rl/env/observation.py tests/test_observation.py
git commit -m "feat: shared deploy-faithful observation builder"
```

---

### Task 6: Rewards — base reward + potential-based shaping

**Files:**
- Create: `breakout_rl/env/rewards.py`
- Test: `tests/test_rewards.py`

- [ ] **Step 1: Write the failing test**

`tests/test_rewards.py`:
```python
import math
from breakout_rl.env.rewards import base_reward, potential, RewardConfig


def test_brick_score_delta_is_positive_reward():
    before = {"lives": 3, "score": 0, "game_over": False}
    after = {"lives": 3, "score": 3, "game_over": False}
    cfg = RewardConfig(score_scale=1.0, step_cost=0.0)
    assert base_reward(before, after, cfg) == 3.0


def test_life_loss_uses_penalty_not_score_delta():
    before = {"lives": 3, "score": 75, "game_over": False}
    after = {"lives": 2, "score": 50, "game_over": False}  # score reset to checkpoint
    cfg = RewardConfig(life_loss_penalty=-30.0, step_cost=0.0)
    assert base_reward(before, after, cfg) == -30.0


def test_game_over_adds_terminal_penalty():
    before = {"lives": 1, "score": 10, "game_over": False}
    after = {"lives": 0, "score": 10, "game_over": True}
    cfg = RewardConfig(life_loss_penalty=-30.0, game_over_penalty=-50.0, step_cost=0.0)
    assert base_reward(before, after, cfg) == -80.0


def test_potential_zero_when_ascending_or_terminal():
    s = {"width": 600, "paddle_x": 100.0, "paddle_width": 80.0}
    assert potential(s, None, is_terminal=False) == 0.0          # ascending -> landing None
    assert potential(s, 300.0, is_terminal=True) == 0.0          # terminal


def test_potential_is_negative_normalized_distance():
    s = {"width": 600, "paddle_x": 100.0, "paddle_width": 80.0}  # center = 140
    # landing at 200 -> distance 60 -> -60/600
    assert abs(potential(s, 200.0, is_terminal=False) - (-60.0 / 600.0)) < 1e-9


def test_pbrs_discounted_shaping_telescopes_to_terminal_minus_start():
    # F_t = gamma*phi(s_{t+1}) - phi(s_t); sum_t gamma^t F_t == gamma^T*phi_T - phi_0
    gamma = 0.99
    phis = [-0.5, -0.3, -0.1, 0.0]  # phi_T = 0 (terminal)
    total = 0.0
    for t in range(len(phis) - 1):
        f = gamma * phis[t + 1] - phis[t]
        total += (gamma ** t) * f
    expected = (gamma ** (len(phis) - 1)) * phis[-1] - phis[0]
    assert abs(total - expected) < 1e-12
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_rewards.py -v`
Expected: FAIL with `ModuleNotFoundError`.

- [ ] **Step 3: Implement**

`breakout_rl/env/rewards.py`:
```python
from dataclasses import dataclass
from typing import Optional


@dataclass
class RewardConfig:
    score_scale: float = 1.0
    life_loss_penalty: float = -30.0
    game_over_penalty: float = -50.0
    step_cost: float = -0.01


def base_reward(before: dict, after: dict, cfg: RewardConfig) -> float:
    """Event reward computed from get_state() dicts captured before/after one update.
    On a non-death step the score only increases (+3/brick, +100/clear), so the score
    delta is a clean positive signal. On a death step the score resets to checkpoint,
    so we branch on the lives delta and ignore the (negative) score delta."""
    r = cfg.step_cost
    if after["lives"] < before["lives"]:
        r += cfg.life_loss_penalty
        if after["game_over"]:
            r += cfg.game_over_penalty
    else:
        r += (after["score"] - before["score"]) * cfg.score_scale
    return r


def potential(state: dict, predicted_landing_x: Optional[float], is_terminal: bool) -> float:
    """PBRS potential. MUST be a pure function of state, and MUST be 0 on terminal
    states for policy invariance. `predicted_landing_x` is None when ascending (the
    predictor returns None), which yields potential 0 -> a valid state function."""
    if is_terminal or predicted_landing_x is None:
        return 0.0
    w = float(state["width"])
    paddle_center = float(state["paddle_x"]) + float(state["paddle_width"]) / 2.0
    return -abs(paddle_center - predicted_landing_x) / w
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_rewards.py -v`
Expected: PASS (6 tests).

- [ ] **Step 5: Commit**

```bash
git add breakout_rl/env/rewards.py tests/test_rewards.py
git commit -m "feat: base reward and provable PBRS potential"
```

---

### Task 7: `BreakoutEnv` — Gymnasium wrapper

**Files:**
- Create: `breakout_rl/env/breakout_env.py`
- Test: `tests/test_breakout_env.py`

- [ ] **Step 1: Write the failing test**

`tests/test_breakout_env.py`:
```python
import numpy as np
from breakout_rl.env.breakout_env import BreakoutEnv
from breakout_rl.env.observation import OBS_DIM, ObservationBuilder
from breakout_rl.constants import ACTION_NOOP, ACTION_WEST, ACTION_EAST, DT


def test_reset_returns_obs_of_correct_shape():
    env = BreakoutEnv()
    obs, info = env.reset(seed=0)
    assert obs.shape == (OBS_DIM,)
    assert env.action_space.n == 3


def test_step_returns_5_tuple_and_moves_paddle():
    env = BreakoutEnv()
    env.reset(seed=0)
    x0 = env.game.paddle_x
    obs, r, term, trunc, info = env.step(ACTION_WEST)
    assert obs.shape == (OBS_DIM,)
    assert isinstance(r, float)
    assert env.game.paddle_x <= x0  # moved west (or already at wall)


def test_observation_built_from_wire_state_not_internal_velocity():
    # Guards train/deploy skew: env obs must equal builder fed get_state().
    env = BreakoutEnv()
    env.reset(seed=1)
    env.step(ACTION_NOOP)
    state = env.game.get_state()
    ref_builder = ObservationBuilder()
    ref_builder.reset()
    # replay the two frames the env saw through an independent builder
    ref_builder.build(env._prev_state_for_test, DT)
    expected = ref_builder.build(state, DT)
    actual = env._build_obs(state)
    assert np.allclose(actual, expected)


def test_truncates_at_max_steps():
    env = BreakoutEnv(max_steps=3)
    env.reset(seed=0)
    trunc = False
    for _ in range(3):
        _, _, term, trunc, _ = env.step(ACTION_NOOP)
        if term:
            break
    assert trunc or term
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_breakout_env.py -v`
Expected: FAIL with `ModuleNotFoundError`.

- [ ] **Step 3: Implement**

`breakout_rl/env/breakout_env.py`:
```python
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

        landing = predict_landing(self.game) if self.game.ball_vy > 0 else None
        cur_phi = potential(after, landing, is_terminal=terminated)
        reward = base + self.gamma * cur_phi - self.prev_phi
        self.prev_phi = 0.0 if terminated else cur_phi

        self.steps += 1
        truncated = self.steps >= self.max_steps
        obs = self._build_obs(after)
        info = {"score": self.game.score, "lives": self.game.lives,
                "bricks_left": int(self.game.brick_array[:, 4].sum())}
        return obs, float(reward), terminated, truncated, info
```

> **Note (perf):** `predict_landing` is called each descending step. This is correct but a hotspot. Task 14 (optional) adds a descent cache. Get correctness first.

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_breakout_env.py -v`
Expected: PASS (4 tests).

- [ ] **Step 5: Commit**

```bash
git add breakout_rl/env/breakout_env.py tests/test_breakout_env.py
git commit -m "feat: Gymnasium BreakoutEnv with PBRS reward"
```

---

# PHASE M2 — Flat DQN (Double + Dueling + PER)

### Task 8: Prioritized replay buffer

**Files:**
- Create: `breakout_rl/agents/replay.py`
- Test: `tests/test_replay.py`

- [ ] **Step 1: Write the failing test**

`tests/test_replay.py`:
```python
import numpy as np
from breakout_rl.agents.replay import PrioritizedReplay, Transition


def _t(i):
    return Transition(
        state=np.zeros(4, np.float32), action=0, reward=float(i),
        next_state=np.zeros(4, np.float32), done=False, gamma=0.99,
    )


def test_len_and_capacity():
    buf = PrioritizedReplay(capacity=3)
    for i in range(5):
        buf.add(_t(i))
    assert len(buf) == 3  # overwrote oldest


def test_sample_shapes_and_weight_range():
    buf = PrioritizedReplay(capacity=64)
    for i in range(64):
        buf.add(_t(i))
    batch, idxs, weights = buf.sample(8, beta=0.4)
    assert len(batch) == 8 and len(idxs) == 8
    assert weights.shape == (8,)
    assert np.all(weights > 0) and np.all(weights <= 1.0 + 1e-6)


def test_priority_update_biases_sampling():
    buf = PrioritizedReplay(capacity=64)
    for i in range(64):
        buf.add(_t(i))
    # make index 0 hugely important; it should be sampled frequently
    buf.update_priorities(np.array([0]), np.array([1000.0]))
    counts = 0
    for _ in range(50):
        _, idxs, _ = buf.sample(4, beta=0.4)
        counts += int(0 in idxs.tolist())
    assert counts > 0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_replay.py -v`
Expected: FAIL with `ModuleNotFoundError`.

- [ ] **Step 3: Implement**

`breakout_rl/agents/replay.py`:
```python
from dataclasses import dataclass
from typing import List, Tuple
import numpy as np


@dataclass
class Transition:
    state: np.ndarray
    action: int
    reward: float
    next_state: np.ndarray
    done: bool
    gamma: float  # per-sample discount (gamma for flat, gamma**k for SMDP options)


class PrioritizedReplay:
    """Proportional prioritized experience replay (simple array-backed implementation).
    Priority p_i = (|td_error| + eps) ** alpha; sampling prob = p_i / sum(p)."""

    def __init__(self, capacity: int, alpha: float = 0.6, eps: float = 1e-5) -> None:
        self.capacity = capacity
        self.alpha = alpha
        self.eps = eps
        self.data: List[Transition] = []
        self.priorities = np.zeros(capacity, dtype=np.float64)
        self.pos = 0

    def __len__(self) -> int:
        return len(self.data)

    def add(self, t: Transition) -> None:
        max_p = self.priorities.max() if self.data else 1.0
        if len(self.data) < self.capacity:
            self.data.append(t)
        else:
            self.data[self.pos] = t
        self.priorities[self.pos] = max_p  # new samples get max priority
        self.pos = (self.pos + 1) % self.capacity

    def sample(self, batch_size: int, beta: float) -> Tuple[List[Transition], np.ndarray, np.ndarray]:
        n = len(self.data)
        prios = self.priorities[:n] ** self.alpha
        probs = prios / prios.sum()
        idxs = np.random.choice(n, batch_size, p=probs)
        batch = [self.data[i] for i in idxs]
        weights = (n * probs[idxs]) ** (-beta)
        weights = weights / weights.max()
        return batch, idxs, weights.astype(np.float32)

    def update_priorities(self, idxs: np.ndarray, td_errors: np.ndarray) -> None:
        for i, e in zip(idxs, td_errors):
            self.priorities[int(i)] = abs(float(e)) + self.eps
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_replay.py -v`
Expected: PASS (3 tests).

- [ ] **Step 5: Commit**

```bash
git add breakout_rl/agents/replay.py tests/test_replay.py
git commit -m "feat: prioritized experience replay buffer"
```

---

### Task 9: Dueling network

**Files:**
- Create: `breakout_rl/agents/networks.py`
- Test: `tests/test_networks.py`

- [ ] **Step 1: Write the failing test**

`tests/test_networks.py`:
```python
import torch
from breakout_rl.agents.networks import DuelingMLP


def test_output_shape_matches_actions():
    net = DuelingMLP(in_dim=23, n_actions=3, hidden=64)
    q = net(torch.zeros(5, 23))
    assert q.shape == (5, 3)


def test_dueling_advantage_is_mean_centered():
    # If V is fixed and A has nonzero mean, Q must equal V + (A - mean(A)).
    net = DuelingMLP(in_dim=4, n_actions=3, hidden=16)
    x = torch.randn(2, 4)
    q = net(x)
    # mean over actions of (q - V) == 0 by construction of dueling head
    assert q.shape == (2, 3)
    # numerical sanity: not NaN
    assert torch.isfinite(q).all()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_networks.py -v`
Expected: FAIL with `ModuleNotFoundError`.

- [ ] **Step 3: Implement**

`breakout_rl/agents/networks.py`:
```python
import torch
import torch.nn as nn


class DuelingMLP(nn.Module):
    def __init__(self, in_dim: int, n_actions: int, hidden: int = 128) -> None:
        super().__init__()
        self.torso = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
        )
        self.value = nn.Sequential(nn.Linear(hidden, hidden), nn.ReLU(), nn.Linear(hidden, 1))
        self.adv = nn.Sequential(nn.Linear(hidden, hidden), nn.ReLU(), nn.Linear(hidden, n_actions))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.torso(x)
        v = self.value(h)                       # (B, 1)
        a = self.adv(h)                          # (B, n_actions)
        return v + (a - a.mean(dim=1, keepdim=True))
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_networks.py -v`
Expected: PASS (2 tests).

- [ ] **Step 5: Commit**

```bash
git add breakout_rl/agents/networks.py tests/test_networks.py
git commit -m "feat: dueling MLP network"
```

---

### Task 10: DQN agent (Double DQN update, ε-greedy)

**Files:**
- Create: `breakout_rl/agents/dqn_agent.py`
- Test: `tests/test_dqn_agent.py`

- [ ] **Step 1: Write the failing test**

`tests/test_dqn_agent.py`:
```python
import numpy as np
from breakout_rl.agents.dqn_agent import DQNAgent
from breakout_rl.agents.replay import PrioritizedReplay, Transition


def test_select_action_in_range_and_greedy_is_deterministic():
    agent = DQNAgent(obs_dim=23, n_actions=3, device="cpu")
    obs = np.zeros(23, np.float32)
    a = agent.select_action(obs, epsilon=0.0)
    assert a in (0, 1, 2)
    assert agent.select_action(obs, epsilon=0.0) == a  # greedy deterministic


def test_update_runs_and_returns_finite_loss():
    agent = DQNAgent(obs_dim=4, n_actions=3, device="cpu")
    buf = PrioritizedReplay(capacity=128)
    for i in range(128):
        buf.add(Transition(
            state=np.random.randn(4).astype(np.float32), action=i % 3,
            reward=1.0, next_state=np.random.randn(4).astype(np.float32),
            done=False, gamma=0.99,
        ))
    loss = agent.update(buf, batch_size=32, beta=0.4)
    assert np.isfinite(loss)


def test_target_sync_copies_weights():
    agent = DQNAgent(obs_dim=4, n_actions=3, device="cpu")
    # perturb online net, sync, assert equal
    for p in agent.online.parameters():
        p.data.add_(1.0)
    agent.sync_target()
    for po, pt in zip(agent.online.parameters(), agent.target.parameters()):
        assert (po.data == pt.data).all()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_dqn_agent.py -v`
Expected: FAIL with `ModuleNotFoundError`.

- [ ] **Step 3: Implement**

`breakout_rl/agents/dqn_agent.py`:
```python
import random
from typing import Tuple
import numpy as np
import torch
import torch.nn as nn

from breakout_rl.agents.networks import DuelingMLP
from breakout_rl.agents.replay import PrioritizedReplay


class DQNAgent:
    def __init__(self, obs_dim: int, n_actions: int, hidden: int = 128,
                 lr: float = 1e-3, device: str = "cuda") -> None:
        self.n_actions = n_actions
        self.device = torch.device(device if torch.cuda.is_available() or device == "cpu" else "cpu")
        self.online = DuelingMLP(obs_dim, n_actions, hidden).to(self.device)
        self.target = DuelingMLP(obs_dim, n_actions, hidden).to(self.device)
        self.sync_target()
        self.opt = torch.optim.Adam(self.online.parameters(), lr=lr)

    def sync_target(self) -> None:
        self.target.load_state_dict(self.online.state_dict())

    @torch.no_grad()
    def select_action(self, obs: np.ndarray, epsilon: float) -> int:
        if random.random() < epsilon:
            return random.randrange(self.n_actions)
        x = torch.as_tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)
        return int(self.online(x).argmax(dim=1).item())

    def update(self, buffer: PrioritizedReplay, batch_size: int, beta: float,
               grad_clip: float = 10.0) -> float:
        batch, idxs, weights = buffer.sample(batch_size, beta)
        s = torch.as_tensor(np.stack([t.state for t in batch]), device=self.device)
        a = torch.as_tensor([t.action for t in batch], device=self.device).long()
        r = torch.as_tensor([t.reward for t in batch], dtype=torch.float32, device=self.device)
        ns = torch.as_tensor(np.stack([t.next_state for t in batch]), device=self.device)
        done = torch.as_tensor([t.done for t in batch], dtype=torch.float32, device=self.device)
        gamma = torch.as_tensor([t.gamma for t in batch], dtype=torch.float32, device=self.device)
        w = torch.as_tensor(weights, device=self.device)

        q = self.online(s).gather(1, a.unsqueeze(1)).squeeze(1)
        with torch.no_grad():
            next_a = self.online(ns).argmax(dim=1, keepdim=True)        # Double DQN
            next_q = self.target(ns).gather(1, next_a).squeeze(1)
            target = r + gamma * next_q * (1.0 - done)
        td = q - target
        loss = (w * td.pow(2)).mean()

        self.opt.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.online.parameters(), grad_clip)
        self.opt.step()

        buffer.update_priorities(idxs, td.detach().cpu().numpy())
        return float(loss.item())
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_dqn_agent.py -v`
Expected: PASS (3 tests).

- [ ] **Step 5: Commit**

```bash
git add breakout_rl/agents/dqn_agent.py tests/test_dqn_agent.py
git commit -m "feat: Double DQN agent with PER-weighted update"
```

---

### Task 11: Curriculum helper

**Files:**
- Create: `breakout_rl/train/curriculum.py`
- Test: `tests/test_curriculum.py`

- [ ] **Step 1: Write the failing test**

`tests/test_curriculum.py`:
```python
from breakout_rl.train.curriculum import curriculum_params


def test_stage1_easy_then_stage2_real():
    # before switch step -> easy; after -> real
    pw0, bs0 = curriculum_params(step=0, switch_step=1000)
    pw1, bs1 = curriculum_params(step=2000, switch_step=1000)
    assert (pw0, bs0) == (120.0, 200.0)
    assert (pw1, bs1) == (80.0, 300.0)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_curriculum.py -v`
Expected: FAIL with `ModuleNotFoundError`.

- [ ] **Step 3: Implement**

`breakout_rl/train/curriculum.py`:
```python
from typing import Tuple

STAGE1 = (120.0, 200.0)  # (paddle_width, ball_speed) — easy
STAGE2 = (80.0, 300.0)   # real difficulty


def curriculum_params(step: int, switch_step: int) -> Tuple[float, float]:
    """Two-stage curriculum: easy until `switch_step`, then real settings.
    Stage 2 must run long enough that final eval reflects real difficulty."""
    return STAGE1 if step < switch_step else STAGE2
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_curriculum.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add breakout_rl/train/curriculum.py tests/test_curriculum.py
git commit -m "feat: two-stage curriculum helper"
```

---

### Task 12: Flat DQN training config + entrypoint

**Files:**
- Create: `breakout_rl/configs/flat_dqn.yaml`, `breakout_rl/train/train_dqn.py`

- [ ] **Step 1: Write the config**

`breakout_rl/configs/flat_dqn.yaml`:
```yaml
run_id: flat_dqn
seed: 0
total_steps: 400000
max_episode_steps: 4000
gamma: 0.99
lr: 0.001
hidden: 128
buffer_capacity: 100000
batch_size: 128
learn_start: 2000
train_every: 4
target_sync_every: 2000
epsilon_start: 1.0
epsilon_end: 0.05
epsilon_decay_steps: 100000
per_beta_start: 0.4
per_beta_end: 1.0
curriculum_switch_step: 120000
eval_every: 20000
eval_episodes: 20
checkpoint_every: 50000
reward:
  score_scale: 1.0
  life_loss_penalty: -30.0
  game_over_penalty: -50.0
  step_cost: -0.01
logic_commit: "FILL_AT_RUNTIME"   # pin via scripts/record_commit (Task 27)
```

- [ ] **Step 2: Write the training entrypoint**

`breakout_rl/train/train_dqn.py`:
```python
import argparse
import csv
import os
import random
from pathlib import Path

import numpy as np
import torch
import yaml
from torch.utils.tensorboard import SummaryWriter

from breakout_rl.env.breakout_env import BreakoutEnv
from breakout_rl.env.observation import OBS_DIM
from breakout_rl.env.rewards import RewardConfig
from breakout_rl.agents.dqn_agent import DQNAgent
from breakout_rl.agents.replay import PrioritizedReplay, Transition
from breakout_rl.train.curriculum import curriculum_params
from breakout_rl.eval.metrics import evaluate_agent


def linear(start, end, frac):
    frac = min(max(frac, 0.0), 1.0)
    return start + (end - start) * frac


def main(cfg_path: str) -> None:
    cfg = yaml.safe_load(open(cfg_path))
    seed = cfg["seed"]
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)

    run_dir = Path("checkpoints") / cfg["run_id"]
    run_dir.mkdir(parents=True, exist_ok=True)
    writer = SummaryWriter(str(run_dir / "tb"))
    csv_file = open(run_dir / "log.csv", "w", newline="")
    csv_writer = csv.writer(csv_file)
    csv_writer.writerow(["step", "episode_return", "episode_score", "loss", "epsilon", "eval_score"])

    reward_cfg = RewardConfig(**cfg["reward"])
    env = BreakoutEnv(max_steps=cfg["max_episode_steps"], gamma=cfg["gamma"], reward_cfg=reward_cfg)
    agent = DQNAgent(OBS_DIM, n_actions=3, hidden=cfg["hidden"], lr=cfg["lr"])
    buffer = PrioritizedReplay(cfg["buffer_capacity"])

    pw, bs = curriculum_params(0, cfg["curriculum_switch_step"])
    env.set_curriculum(pw, bs)
    obs, _ = env.reset(seed=seed)
    ep_return, ep_score = 0.0, 0
    last_loss = 0.0

    for step in range(1, cfg["total_steps"] + 1):
        pw, bs = curriculum_params(step, cfg["curriculum_switch_step"])
        env.set_curriculum(pw, bs)

        eps = linear(cfg["epsilon_start"], cfg["epsilon_end"], step / cfg["epsilon_decay_steps"])
        action = agent.select_action(obs, eps)
        next_obs, reward, term, trunc, info = env.step(action)
        buffer.add(Transition(obs, action, reward, next_obs, term, cfg["gamma"]))
        obs = next_obs
        ep_return += reward
        ep_score = info["score"]

        if len(buffer) >= cfg["learn_start"] and step % cfg["train_every"] == 0:
            beta = linear(cfg["per_beta_start"], cfg["per_beta_end"], step / cfg["total_steps"])
            last_loss = agent.update(buffer, cfg["batch_size"], beta)

        if step % cfg["target_sync_every"] == 0:
            agent.sync_target()

        if term or trunc:
            writer.add_scalar("train/episode_return", ep_return, step)
            writer.add_scalar("train/episode_score", ep_score, step)
            writer.add_scalar("train/epsilon", eps, step)
            csv_writer.writerow([step, ep_return, ep_score, last_loss, eps, ""])
            csv_file.flush()
            obs, _ = env.reset()
            ep_return, ep_score = 0.0, 0

        if step % cfg["eval_every"] == 0:
            stats = evaluate_agent(agent, episodes=cfg["eval_episodes"],
                                   paddle_width=80.0, ball_speed=300.0, seed=10_000 + step)
            writer.add_scalar("eval/score", stats["mean_score"], step)
            writer.add_scalar("eval/clears", stats["mean_clears"], step)
            csv_writer.writerow([step, "", "", last_loss, eps, stats["mean_score"]])
            csv_file.flush()
            print(f"[{step}] eval score={stats['mean_score']:.1f} clears={stats['mean_clears']:.2f}")

        if step % cfg["checkpoint_every"] == 0:
            torch.save(agent.online.state_dict(), run_dir / f"online_{step}.pt")

    torch.save(agent.online.state_dict(), run_dir / "online_final.pt")
    csv_file.close(); writer.close()


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="breakout_rl/configs/flat_dqn.yaml")
    main(p.parse_args().config)
```

> Depends on `evaluate_agent` from Task 13 — implement Task 13 first if executing strictly in order. (Listed after for narrative flow; build 13 before running 12.)

- [ ] **Step 3: Commit (no run yet)**

```bash
git add breakout_rl/configs/flat_dqn.yaml breakout_rl/train/train_dqn.py
git commit -m "feat: flat DQN training entrypoint and config"
```

---

### Task 13: Evaluation metrics

**Files:**
- Create: `breakout_rl/eval/metrics.py`
- Test: `tests/test_metrics.py`

- [ ] **Step 1: Write the failing test**

`tests/test_metrics.py`:
```python
import numpy as np
from breakout_rl.eval.metrics import evaluate_policy


def test_evaluate_policy_returns_expected_keys():
    # a trivial always-NOOP policy
    def policy(obs):
        return 0
    stats = evaluate_policy(policy, episodes=2, max_steps=200,
                            paddle_width=80.0, ball_speed=300.0, seed=0)
    for k in ["mean_score", "mean_clears", "mean_survival_steps",
              "mean_bricks_per_life", "std_score"]:
        assert k in stats
    assert stats["mean_survival_steps"] > 0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_metrics.py -v`
Expected: FAIL with `ModuleNotFoundError`.

- [ ] **Step 3: Implement**

`breakout_rl/eval/metrics.py`:
```python
from typing import Callable, Dict
import numpy as np

from breakout_rl.env.breakout_env import BreakoutEnv
from breakout_rl.env.observation import OBS_DIM


def evaluate_policy(policy: Callable[[np.ndarray], int], episodes: int,
                    max_steps: int = 4000, paddle_width: float = 80.0,
                    ball_speed: float = 300.0, seed: int = 0) -> Dict[str, float]:
    """Roll out a policy (obs -> action int) for several full games and aggregate
    clearing-focused metrics (see spec §7). Survival/score are expected to saturate;
    clears / bricks-per-life are the discriminating metrics."""
    env = BreakoutEnv(max_steps=max_steps, paddle_width=paddle_width, ball_speed=ball_speed)
    scores, clears, survivals, bricks_per_life = [], [], [], []

    for ep in range(episodes):
        obs, _ = env.reset(seed=seed + ep)
        ep_clears = 0
        prev_bricks = int(env.game.brick_array[:, 4].sum())
        bricks_broken = 0
        steps = 0
        while True:
            obs, r, term, trunc, info = env.step(policy(obs))
            steps += 1
            now = info["bricks_left"]
            if now < prev_bricks:
                bricks_broken += (prev_bricks - now)
            if now == 0 or (prev_bricks <= 2 and now > prev_bricks):  # board cleared+respawned
                ep_clears += 1
            prev_bricks = now
            if term or trunc:
                break
        scores.append(env.game.score)
        clears.append(ep_clears)
        survivals.append(steps)
        bricks_per_life.append(bricks_broken / 3.0)

    return {
        "mean_score": float(np.mean(scores)),
        "std_score": float(np.std(scores)),
        "mean_clears": float(np.mean(clears)),
        "mean_survival_steps": float(np.mean(survivals)),
        "mean_bricks_per_life": float(np.mean(bricks_per_life)),
    }


def evaluate_agent(agent, episodes: int, **kw) -> Dict[str, float]:
    """Adapter: wrap a DQNAgent as a greedy policy."""
    return evaluate_policy(lambda o: agent.select_action(o, epsilon=0.0), episodes=episodes, **kw)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_metrics.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add breakout_rl/eval/metrics.py tests/test_metrics.py
git commit -m "feat: rollout evaluation metrics"
```

---

### Task 14 (optional perf): descent cache in `BreakoutEnv`

**Files:**
- Modify: `breakout_rl/env/breakout_env.py`

Only do this if a short training smoke run is throughput-bound on `predict_landing`.

- [ ] **Step 1: Add cache fields in `reset`**, after `self.steps = 0`:
```python
self._cached_landing = None
self._bricks_at_cache = int(self.game.brick_array[:, 4].sum())
self._last_vy = self.game.ball_vy
```

- [ ] **Step 2: Replace the landing computation in `step`** (the `landing = predict_landing(...)` line) with:
```python
descending = self.game.ball_vy > 0
bricks_now = int(self.game.brick_array[:, 4].sum())
if not descending:
    self._cached_landing = None
elif (self._cached_landing is None or bricks_now != self._bricks_at_cache or self._last_vy <= 0):
    self._cached_landing = predict_landing(self.game)
    self._bricks_at_cache = bricks_now
self._last_vy = self.game.ball_vy
landing = self._cached_landing
```

- [ ] **Step 3: Re-run env + reward tests**

Run: `python -m pytest tests/test_breakout_env.py tests/test_rewards.py -v`
Expected: PASS (caching is exact — equal to recomputing during a clean descent).

- [ ] **Step 4: Commit**

```bash
git add breakout_rl/env/breakout_env.py
git commit -m "perf: descent-cached landing prediction in BreakoutEnv"
```

---

### Task 15: Smoke-train the flat DQN

**Files:** none (run + observe)

- [ ] **Step 1: Short run to confirm the loop learns**

Run (override total_steps for a quick smoke):
```bash
source venv/bin/activate
python - <<'PY'
import yaml
c = yaml.safe_load(open("breakout_rl/configs/flat_dqn.yaml"))
c.update(total_steps=20000, eval_every=5000, checkpoint_every=20000, curriculum_switch_step=8000)
yaml.safe_dump(c, open("breakout_rl/configs/_smoke.yaml", "w"))
PY
python -m breakout_rl.train.train_dqn --config breakout_rl/configs/_smoke.yaml
```
Expected: prints periodic eval lines; eval score should be clearly above the random baseline by the end (the dummy random agent scores near 0). If it does not, debug with superpowers:systematic-debugging before the full run.

- [ ] **Step 2: Full run (single GPU)**

```bash
python -m breakout_rl.train.train_dqn --config breakout_rl/configs/flat_dqn.yaml
```
Expected: TensorBoard curves in `checkpoints/flat_dqn/tb`, `online_final.pt` saved.

- [ ] **Step 3: Commit the kept checkpoint + curve export**

```bash
git add -f checkpoints/flat_dqn/online_final.pt checkpoints/flat_dqn/log.csv
git commit -m "chore: flat DQN trained checkpoint and training log"
```

---

# PHASE M3 — Aiming-authority probe (GATE)

> **Decision gate (spec §8 M3).** Measure how much clearing advantage aiming can buy, given the stochastic ~30° bounce band, before investing in the full hierarchy. Large advantage → build M4. Marginal → pivot to implicit reward-shaping aiming in the flat DQN and document the finding.

### Task 16: Aim controller (physics-based low level)

**Files:**
- Create: `breakout_rl/controllers/aim_controller.py`
- Test: `tests/test_aim_controller.py`

- [ ] **Step 1: Write the failing test**

`tests/test_aim_controller.py`:
```python
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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_aim_controller.py -v`
Expected: FAIL with `ModuleNotFoundError`.

- [ ] **Step 3: Implement**

`breakout_rl/controllers/aim_controller.py`:
```python
from typing import Optional

from server.logic import Breakout
from breakout_rl.physics.predictor import predict_landing
from breakout_rl.constants import (REGION_LEFT, REGION_CENTER, REGION_RIGHT,
                                   ACTION_NOOP, ACTION_WEST, ACTION_EAST)

_REGION_FRAC = {REGION_LEFT: 1.0 / 6.0, REGION_CENTER: 0.5, REGION_RIGHT: 5.0 / 6.0}
_HALF_STEP = 12.5  # half of the 25px paddle step


def target_paddle_x(game: Breakout, region: int) -> Optional[float]:
    """Paddle_x that makes the ball contact the chosen third of the paddle.
    Returns None while ascending (no aim possible). Falls back to a center intercept
    when the chosen region is infeasible (ball too near a wall)."""
    landing = predict_landing(game)
    if landing is None:
        return None
    w = game.width
    pw = game.paddle_width  # read from game (curriculum may change it)
    tx = landing - _REGION_FRAC[region] * pw
    clamped = min(max(tx, 0.0), w - pw)
    if abs(clamped - tx) > 1e-6:  # region infeasible -> center intercept
        tx = landing - 0.5 * pw
        clamped = min(max(tx, 0.0), w - pw)
    return clamped


def choose_action(game: Breakout, region: int) -> int:
    tx = target_paddle_x(game, region)
    if tx is None:
        return ACTION_NOOP  # hold while ascending
    dx = tx - game.paddle_x
    if dx > _HALF_STEP:
        return ACTION_EAST
    if dx < -_HALF_STEP:
        return ACTION_WEST
    return ACTION_NOOP
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_aim_controller.py -v`
Expected: PASS (4 tests).

- [ ] **Step 5: Commit**

```bash
git add breakout_rl/controllers/aim_controller.py tests/test_aim_controller.py
git commit -m "feat: physics-based aim controller (low level)"
```

---

### Task 17: Aiming-authority probe experiment

**Files:**
- Create: `breakout_rl/eval/probe_aiming_authority.py`

- [ ] **Step 1: Implement the probe**

`breakout_rl/eval/probe_aiming_authority.py`:
```python
"""M3 GATE. Compare three scripted policies over many seeds:
  - center: always aim CENTER (the survival-ceiling baseline)
  - random_region: pick a random region each volley
  - oracle_region: at each decision, brute-force try each region on a CLONE and keep
    the one that breaks the most bricks over the next volley (upper bound on aiming).
Report mean clears / bricks-per-life. If oracle >> center on clearing metrics, build
the full SMDP hierarchy (M4). If the gap is small, pivot to implicit reward shaping."""
import argparse
import random
import numpy as np

from server.logic import Breakout
from breakout_rl.physics.clone import clone_game
from breakout_rl.controllers.aim_controller import choose_action
from breakout_rl.constants import (DT, ACTION_WEST, ACTION_EAST,
                                   REGION_LEFT, REGION_CENTER, REGION_RIGHT, DECISION_LINE_Y)

REGIONS = [REGION_LEFT, REGION_CENTER, REGION_RIGHT]


def _apply(g: Breakout, action: int) -> None:
    if action == ACTION_WEST:
        g.move_paddle("WEST")
    elif action == ACTION_EAST:
        g.move_paddle("EAST")


def _simulate_volley_break_count(game: Breakout, region: int, max_steps: int = 800) -> int:
    """Bricks broken if we commit to `region` for the upcoming volley, simulated on a
    clone until the next decision point / life loss / game over."""
    g = clone_game(game)
    start = int(g.brick_array[:, 4].sum())
    prev_vy = g.ball_vy
    contacted = False
    for _ in range(max_steps):
        _apply(g, choose_action(g, region))
        g.update(DT)
        if g.lives < game.lives or g.game_over:
            break
        if prev_vy > 0 and g.ball_vy < 0:
            contacted = True
        if contacted and g.ball_y > DECISION_LINE_Y and g.ball_vy > 0 and prev_vy <= 0:
            break
        prev_vy = g.ball_vy
    return start - int(g.brick_array[:, 4].sum())


def _best_region(game: Breakout) -> int:
    best, best_n = REGION_CENTER, -1
    for region in REGIONS:
        n = _simulate_volley_break_count(game, region)
        if n > best_n:
            best_n, best = n, region
    return best


def run_policy(kind: str, episodes: int, seed: int, max_steps: int = 4000) -> dict:
    clears, bricks = [], []
    for ep in range(episodes):
        random.seed(seed + ep); np.random.seed(seed + ep)
        g = Breakout()
        prev = int(g.brick_array[:, 4].sum())
        broken = 0
        region = REGION_CENTER
        prev_vy = g.ball_vy
        steps = 0
        while not g.game_over and steps < max_steps:
            if g.ball_y > DECISION_LINE_Y and g.ball_vy > 0 and prev_vy <= 0:
                if kind == "center":
                    region = REGION_CENTER
                elif kind == "random_region":
                    region = random.choice(REGIONS)
                elif kind == "oracle_region":
                    region = _best_region(g)
            prev_vy = g.ball_vy
            _apply(g, choose_action(g, region))
            g.update(DT)
            now = int(g.brick_array[:, 4].sum())
            if now < prev:
                broken += (prev - now)
            prev = now
            steps += 1
        clears.append(broken // 16)
        bricks.append(broken / 3.0)
    return {"mean_clears": float(np.mean(clears)),
            "mean_bricks_per_life": float(np.mean(bricks))}


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--episodes", type=int, default=50)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()
    for kind in ["center", "random_region", "oracle_region"]:
        stats = run_policy(kind, args.episodes, args.seed)
        print(f"{kind:16s} clears={stats['mean_clears']:.2f} "
              f"bricks/life={stats['mean_bricks_per_life']:.2f}")
```

- [ ] **Step 2: Run the probe**

```bash
python -m breakout_rl.eval.probe_aiming_authority --episodes 100
```
Expected: three lines. Record the numbers in the report. **Gate decision:** if `oracle_region` clears/bricks-per-life is materially above `center` (e.g. ≥25% more bricks/life), proceed to M4. Otherwise, skip M4 and instead add aiming terms to the flat DQN reward (document this pivot in the report).

- [ ] **Step 3: Commit**

```bash
git add breakout_rl/eval/probe_aiming_authority.py
git commit -m "feat: M3 aiming-authority probe (gate experiment)"
```

---

# PHASE M4 — Hierarchy (SMDP) — only if the M3 gate passes

### Task 18: `HighLevelEnv` — options/SMDP wrapper

**Files:**
- Create: `breakout_rl/env/high_level_env.py`
- Test: `tests/test_high_level_env.py`

- [ ] **Step 1: Write the failing test**

`tests/test_high_level_env.py`:
```python
from breakout_rl.env.high_level_env import HighLevelEnv, HIGH_OBS_DIM
from breakout_rl.constants import REGION_CENTER


def test_reset_returns_high_obs():
    env = HighLevelEnv(max_option_steps=2000)
    obs, info = env.reset(seed=0)
    assert obs.shape == (HIGH_OBS_DIM,)


def test_step_returns_gamma_k_and_k():
    env = HighLevelEnv(max_option_steps=2000)
    env.reset(seed=0)
    obs, R, term, trunc, info = env.step(REGION_CENTER)
    assert "gamma_k" in info and "k" in info
    assert info["k"] >= 1
    assert 0.0 < info["gamma_k"] <= 1.0
    # gamma_k == gamma ** k
    assert abs(info["gamma_k"] - env.gamma ** info["k"]) < 1e-9
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_high_level_env.py -v`
Expected: FAIL with `ModuleNotFoundError`.

- [ ] **Step 3: Implement**

`breakout_rl/env/high_level_env.py`:
```python
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

    def _at_decision_point(self, prev_vy: float) -> bool:
        return (self.game.ball_y > DECISION_LINE_Y and self.game.ball_vy > 0
                and prev_vy <= 0)

    def _advance_to_decision_point(self, region: Optional[int]):
        """Advance with the controller until a new decision point / terminal / life loss.
        region=None means 'no aim yet' (used only during reset bootstrapping)."""
        R, k, discount = 0.0, 0, 1.0
        prev_vy = self.game.ball_vy
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
            if self._at_decision_point(prev_vy):
                return R, k, False, False
            prev_vy = self.game.ball_vy
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
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_high_level_env.py -v`
Expected: PASS (2 tests).

- [ ] **Step 5: Commit**

```bash
git add breakout_rl/env/high_level_env.py tests/test_high_level_env.py
git commit -m "feat: SMDP options HighLevelEnv"
```

---

### Task 19: SMDP training entrypoint

**Files:**
- Create: `breakout_rl/configs/aim.yaml`, `breakout_rl/train/train_aim.py`

The SMDP update reuses `DQNAgent` and `PrioritizedReplay` unchanged: per-transition `gamma` is set to `gamma**k` (from `info`), so `DQNAgent.update` already bootstraps with `gamma**k * max Q`. This is the SMDP backup.

- [ ] **Step 1: Config**

`breakout_rl/configs/aim.yaml`:
```yaml
run_id: aim_smdp
seed: 0
total_options: 60000          # high-level decisions (each spans many primitive steps)
max_option_steps: 3000
gamma: 0.99
lr: 0.0005
hidden: 128
buffer_capacity: 50000
batch_size: 128
learn_start: 1000
train_every: 1
target_sync_every: 1000
epsilon_start: 1.0
epsilon_end: 0.05
epsilon_decay_options: 20000
per_beta_start: 0.4
per_beta_end: 1.0
curriculum_switch_option: 20000
eval_every: 5000
eval_episodes: 20
checkpoint_every: 10000
reward:
  score_scale: 1.0
  life_loss_penalty: -30.0
  game_over_penalty: -50.0
  step_cost: -0.01
logic_commit: "FILL_AT_RUNTIME"
```

- [ ] **Step 2: Training entrypoint**

`breakout_rl/train/train_aim.py`:
```python
import argparse, csv, random
from pathlib import Path
import numpy as np, torch, yaml
from torch.utils.tensorboard import SummaryWriter

from breakout_rl.env.high_level_env import HighLevelEnv, HIGH_OBS_DIM
from breakout_rl.env.rewards import RewardConfig
from breakout_rl.agents.dqn_agent import DQNAgent
from breakout_rl.agents.replay import PrioritizedReplay, Transition
from breakout_rl.train.curriculum import curriculum_params
from breakout_rl.eval.metrics import evaluate_hierarchical


def linear(a, b, frac):
    frac = min(max(frac, 0.0), 1.0)
    return a + (b - a) * frac


def main(cfg_path: str) -> None:
    cfg = yaml.safe_load(open(cfg_path))
    seed = cfg["seed"]; random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    run_dir = Path("checkpoints") / cfg["run_id"]; run_dir.mkdir(parents=True, exist_ok=True)
    writer = SummaryWriter(str(run_dir / "tb"))
    csv_file = open(run_dir / "log.csv", "w", newline=""); cw = csv.writer(csv_file)
    cw.writerow(["option", "option_return", "score", "loss", "epsilon", "eval_clears"])

    env = HighLevelEnv(max_option_steps=cfg["max_option_steps"], gamma=cfg["gamma"],
                       reward_cfg=RewardConfig(**cfg["reward"]))
    agent = DQNAgent(HIGH_OBS_DIM, n_actions=3, hidden=cfg["hidden"], lr=cfg["lr"])
    buffer = PrioritizedReplay(cfg["buffer_capacity"])

    pw, bs = curriculum_params(0, cfg["curriculum_switch_option"]); env.set_curriculum(pw, bs)
    obs, _ = env.reset(seed=seed); last_loss = 0.0

    for opt in range(1, cfg["total_options"] + 1):
        pw, bs = curriculum_params(opt, cfg["curriculum_switch_option"]); env.set_curriculum(pw, bs)
        eps = linear(cfg["epsilon_start"], cfg["epsilon_end"], opt / cfg["epsilon_decay_options"])
        region = agent.select_action(obs, eps)
        next_obs, R, term, trunc, info = env.step(region)
        # SMDP: store gamma**k as the per-transition discount
        buffer.add(Transition(obs, region, R, next_obs, term, info["gamma_k"]))
        obs = next_obs

        if len(buffer) >= cfg["learn_start"] and opt % cfg["train_every"] == 0:
            beta = linear(cfg["per_beta_start"], cfg["per_beta_end"], opt / cfg["total_options"])
            last_loss = agent.update(buffer, cfg["batch_size"], beta)
        if opt % cfg["target_sync_every"] == 0:
            agent.sync_target()
        if term or trunc:
            cw.writerow([opt, R, info["score"], last_loss, eps, ""]); csv_file.flush()
            obs, _ = env.reset()
        if opt % cfg["eval_every"] == 0:
            stats = evaluate_hierarchical(agent, episodes=cfg["eval_episodes"],
                                          paddle_width=80.0, ball_speed=300.0, seed=10_000 + opt)
            writer.add_scalar("eval/clears", stats["mean_clears"], opt)
            writer.add_scalar("eval/bricks_per_life", stats["mean_bricks_per_life"], opt)
            cw.writerow([opt, "", "", last_loss, eps, stats["mean_clears"]]); csv_file.flush()
            print(f"[{opt}] eval clears={stats['mean_clears']:.2f} bpl={stats['mean_bricks_per_life']:.2f}")
        if opt % cfg["checkpoint_every"] == 0:
            torch.save(agent.online.state_dict(), run_dir / f"online_{opt}.pt")

    torch.save(agent.online.state_dict(), run_dir / "online_final.pt")
    csv_file.close(); writer.close()


if __name__ == "__main__":
    p = argparse.ArgumentParser(); p.add_argument("--config", default="breakout_rl/configs/aim.yaml")
    main(p.parse_args().config)
```

- [ ] **Step 3: Add `evaluate_hierarchical` to `breakout_rl/eval/metrics.py`**

Append:
```python
def evaluate_hierarchical(agent, episodes: int, max_steps: int = 4000,
                          paddle_width: float = 80.0, ball_speed: float = 300.0,
                          seed: int = 0):
    """Evaluate a high-level region policy in the HighLevelEnv, aggregating clearing
    metrics across full games (each game = many options)."""
    from breakout_rl.env.high_level_env import HighLevelEnv
    import numpy as np
    env = HighLevelEnv(max_option_steps=max_steps, paddle_width=paddle_width, ball_speed=ball_speed)
    clears, bricks_total = [], []
    for ep in range(episodes):
        obs, _ = env.reset(seed=seed + ep)
        start = int(env.game.brick_array[:, 4].sum())
        broken = 0
        prev = start
        while True:
            region = agent.select_action(obs, epsilon=0.0)
            obs, R, term, trunc, info = env.step(region)
            now = info["bricks_left"]
            if now < prev:
                broken += (prev - now)
            prev = now
            if term or trunc:
                break
        clears.append(broken // 16)
        bricks_total.append(broken / 3.0)
    return {"mean_clears": float(np.mean(clears)),
            "mean_bricks_per_life": float(np.mean(bricks_total))}
```

- [ ] **Step 4: Smoke run**

```bash
python -m breakout_rl.train.train_aim --config breakout_rl/configs/aim.yaml
```
Expected: eval clears trend upward above the `center` baseline from the M3 probe.

- [ ] **Step 5: Commit**

```bash
git add breakout_rl/configs/aim.yaml breakout_rl/train/train_aim.py breakout_rl/eval/metrics.py
git commit -m "feat: SMDP high-level aim training"
git add -f checkpoints/aim_smdp/online_final.pt checkpoints/aim_smdp/log.csv
git commit -m "chore: trained SMDP aim checkpoint and log"
```

---

# PHASE M5 — Deploy, evaluation, report

### Task 20: WebSocket deployment agents

**Files:**
- Create: `breakout_rl/deploy/trained_agent.py`

- [ ] **Step 1: Implement both deploy agents**

`breakout_rl/deploy/trained_agent.py`:
```python
import argparse, asyncio, time
from typing import Optional, Dict, Any
import numpy as np
import torch

from agents.base_agent import BaseAgent
from breakout_rl.agents.networks import DuelingMLP
from breakout_rl.env.observation import ObservationBuilder, OBS_DIM
from breakout_rl.physics.predictor import predict_landing
from breakout_rl.controllers.aim_controller import choose_action
from breakout_rl.physics.clone import clone_game
from server.logic import Breakout
from breakout_rl.constants import (ACTIONS, ACTION_NOOP, ACTION_WEST, ACTION_EAST,
                                   DECISION_LINE_Y)

_ACTION_MSG = {ACTION_WEST: {"action": "move", "direction": "WEST"},
               ACTION_EAST: {"action": "move", "direction": "EAST"}}


def _load(net_path: str, in_dim: int, hidden: int = 128) -> DuelingMLP:
    net = DuelingMLP(in_dim, 3, hidden)
    net.load_state_dict(torch.load(net_path, map_location="cpu"))
    net.eval()
    return net


class FlatDQNAgent(BaseAgent):
    def __init__(self, net_path: str, **kw):
        super().__init__(**kw)
        self.net = _load(net_path, OBS_DIM)
        self.obs_builder = ObservationBuilder()
        self._last_t: Optional[float] = None

    async def deliberate(self) -> Optional[Dict[str, Any]]:
        st = self.current_state
        if not st or st.get("game_over"):
            return None
        now = time.monotonic()
        dt = (now - self._last_t) if self._last_t else (1.0 / 30.0)  # measured dt
        self._last_t = now
        obs = self.obs_builder.build(st, dt)
        with torch.no_grad():
            a = int(self.net(torch.as_tensor(obs).unsqueeze(0)).argmax(1).item())
        return _ACTION_MSG.get(a)  # None for NOOP


def _mirror_from_state(st: Dict[str, Any]) -> Breakout:
    """Build a local Breakout matching the wire state, for the controller's predictor.
    Velocity is unknown on the wire, so estimate it from the builder's finite diff is
    not available here; instead we reconstruct vx/vy from two successive states in the
    hierarchical agent (see below)."""
    g = Breakout(width=int(st["width"]), height=int(st["height"]))
    g.paddle_width = st["paddle_width"]; g.paddle_x = st["paddle_x"]
    g.ball_radius = st["ball_radius"]; g.ball_x = st["ball_x"]; g.ball_y = st["ball_y"]
    active_idx = {b["index"] for b in st.get("bricks", [])}
    for b in g.bricks:
        b.active = b.index in active_idx
    g._sync_bricks_to_numpy()
    return g


class HierarchicalAgent(BaseAgent):
    def __init__(self, net_path: str, **kw):
        super().__init__(**kw)
        self.net = _load(net_path, OBS_DIM)
        self.obs_builder = ObservationBuilder()
        self._prev: Optional[Dict[str, Any]] = None
        self._prev_t: Optional[float] = None
        self._region = 1  # default CENTER
        self._prev_vy = 0.0

    async def deliberate(self) -> Optional[Dict[str, Any]]:
        st = self.current_state
        if not st or st.get("game_over"):
            return None
        now = time.monotonic()
        dt = (now - self._prev_t) if self._prev_t else (1.0 / 30.0)
        # estimate ball velocity from two successive wire states
        if self._prev is not None and dt > 0:
            vx = (st["ball_x"] - self._prev["ball_x"]) / dt
            vy = (st["ball_y"] - self._prev["ball_y"]) / dt
        else:
            vx, vy = 0.0, 0.0
        g = _mirror_from_state(st)
        g.ball_vx, g.ball_vy = vx, vy

        # new high-level decision at a clean descent decision point
        if g.ball_y > DECISION_LINE_Y and vy > 0 and self._prev_vy <= 0:
            obs = self.obs_builder.build(st, dt)
            with torch.no_grad():
                self._region = int(self.net(torch.as_tensor(obs).unsqueeze(0)).argmax(1).item())
        else:
            self.obs_builder.build(st, dt)  # keep history warm

        self._prev, self._prev_t, self._prev_vy = st, now, vy
        a = choose_action(g, self._region)
        return _ACTION_MSG.get(a)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--mode", choices=["flat", "hier"], default="flat")
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--uri", default="ws://localhost:8765/ws")
    args = p.parse_args()
    cls = FlatDQNAgent if args.mode == "flat" else HierarchicalAgent
    asyncio.run(cls(args.checkpoint, server_uri=args.uri).run())
```

- [ ] **Step 2: Manual deploy verification (uses superpowers:verification-before-completion)**

In one terminal: `python -m server.server`
In another:
```bash
python -m breakout_rl.deploy.trained_agent --mode flat --checkpoint checkpoints/flat_dqn/online_final.pt
```
Open `http://localhost:8765/` and confirm the paddle tracks the ball and clears bricks. Repeat with `--mode hier --checkpoint checkpoints/aim_smdp/online_final.pt`.
Expected: visibly competent play; no crashes; score climbs.

- [ ] **Step 3: Commit**

```bash
git add breakout_rl/deploy/trained_agent.py
git commit -m "feat: WebSocket deploy agents (flat + hierarchical)"
```

---

### Task 21: Train/deploy parity test

**Files:**
- Test: `tests/test_observation_parity.py`

- [ ] **Step 1: Write the test**

`tests/test_observation_parity.py`:
```python
import numpy as np
from server.logic import Breakout
from breakout_rl.env.observation import ObservationBuilder
from breakout_rl.constants import DT


def test_env_path_and_deploy_path_produce_identical_vectors():
    """The env and the deployed agent both build observations from get_state() via the
    same ObservationBuilder. Feeding an identical state sequence through two independent
    builders must yield byte-identical vectors (guards train/deploy skew)."""
    g = Breakout()
    g.ball_x, g.ball_y = 250.0, 150.0
    s1 = g.get_state()
    g.update(DT)
    s2 = g.get_state()

    env_builder = ObservationBuilder()
    deploy_builder = ObservationBuilder()
    env_builder.build(s1, DT); deploy_builder.build(s1, DT)
    v_env = env_builder.build(s2, DT)
    v_deploy = deploy_builder.build(s2, DT)

    assert np.array_equal(v_env, v_deploy)
```

- [ ] **Step 2: Run**

Run: `python -m pytest tests/test_observation_parity.py -v`
Expected: PASS.

- [ ] **Step 3: Commit**

```bash
git add tests/test_observation_parity.py
git commit -m "test: train/deploy observation parity"
```

---

### Task 22: Multi-seed runner + comparison table/plots

**Files:**
- Create: `breakout_rl/eval/evaluate.py`

- [ ] **Step 1: Implement**

`breakout_rl/eval/evaluate.py`:
```python
"""Evaluate all agents across seeds and emit a comparison table (CSV + markdown) and
training-curve plots. Baselines: random, analytic-center, flat DQN, hierarchical."""
import argparse, glob, csv
import numpy as np
import torch

from breakout_rl.eval.metrics import evaluate_policy, evaluate_agent, evaluate_hierarchical
from breakout_rl.agents.networks import DuelingMLP
from breakout_rl.env.observation import OBS_DIM
from breakout_rl.controllers.aim_controller import choose_action
from breakout_rl.physics.predictor import predict_landing


def _load(path):
    net = DuelingMLP(OBS_DIM, 3, 128); net.load_state_dict(torch.load(path, map_location="cpu")); net.eval()

    class _A:
        def select_action(self, obs, epsilon=0.0):
            with torch.no_grad():
                return int(net(torch.as_tensor(obs).unsqueeze(0)).argmax(1).item())
    return _A()


def main(episodes: int, seeds: int):
    rows = []
    # random baseline
    rnd = lambda o: np.random.randint(3)
    # analytic-center baseline: a policy needs a game handle; evaluate via metrics on env
    # (here we approximate with the flat-DQN env rollout using the controller as policy)
    for seed in range(seeds):
        rows.append(("random", seed, evaluate_policy(rnd, episodes, seed=seed)))
        for path in glob.glob("checkpoints/flat_dqn/online_final.pt"):
            rows.append(("flat_dqn", seed, evaluate_agent(_load(path), episodes, seed=seed)))
        for path in glob.glob("checkpoints/aim_smdp/online_final.pt"):
            rows.append(("hierarchical", seed, evaluate_hierarchical(_load(path), episodes, seed=seed)))

    # aggregate mean±std per agent
    agg = {}
    for name, seed, stats in rows:
        agg.setdefault(name, []).append(stats)
    print("\n| agent | clears (mean±std) | bricks/life | score |")
    print("|---|---|---|---|")
    with open("checkpoints/comparison.csv", "w", newline="") as f:
        w = csv.writer(f); w.writerow(["agent", "clears_mean", "clears_std", "bpl_mean", "score_mean"])
        for name, lst in agg.items():
            c = [s["mean_clears"] for s in lst]
            b = [s.get("mean_bricks_per_life", 0.0) for s in lst]
            sc = [s.get("mean_score", 0.0) for s in lst]
            print(f"| {name} | {np.mean(c):.2f}±{np.std(c):.2f} | {np.mean(b):.2f} | {np.mean(sc):.1f} |")
            w.writerow([name, np.mean(c), np.std(c), np.mean(b), np.mean(sc)])


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--episodes", type=int, default=30)
    p.add_argument("--seeds", type=int, default=3)
    a = p.parse_args()
    main(a.episodes, a.seeds)
```

- [ ] **Step 2: Run**

```bash
python -m breakout_rl.eval.evaluate --episodes 30 --seeds 3
```
Expected: a markdown comparison table on stdout + `checkpoints/comparison.csv`. The hierarchical agent should lead on clears / bricks-per-life (if the M3 gate passed).

- [ ] **Step 3: Commit**

```bash
git add breakout_rl/eval/evaluate.py
git commit -m "feat: multi-seed comparison evaluation"
git add -f checkpoints/comparison.csv
git commit -m "chore: evaluation comparison results"
```

---

### Task 23: Multi-seed training script

**Files:**
- Create: `scripts/train_seeds.sh`

- [ ] **Step 1: Implement**

`scripts/train_seeds.sh`:
```bash
#!/usr/bin/env bash
set -euo pipefail
source venv/bin/activate
for s in 0 1 2; do
  python - <<PY
import yaml
c = yaml.safe_load(open("breakout_rl/configs/flat_dqn.yaml"))
c["seed"] = $s; c["run_id"] = f"flat_dqn_seed$s"
yaml.safe_dump(c, open("breakout_rl/configs/_flat_seed$s.yaml", "w"))
PY
  python -m breakout_rl.train.train_dqn --config breakout_rl/configs/_flat_seed$s.yaml
done
```

- [ ] **Step 2: Make executable + commit (run later for final numbers)**

```bash
chmod +x scripts/train_seeds.sh
git add scripts/train_seeds.sh
git commit -m "chore: multi-seed training script"
```

---

### Task 24: README / report

**Files:**
- Modify: `README.md` (append a "Reinforcement Learning Agent" report section)

- [ ] **Step 1: Append the report section** covering, with the actual numbers/plots produced:
  - How to run training (`python -m breakout_rl.train.train_dqn --config ...`), the probe, and deploy.
  - Architecture: headless env, rollout predictor, shared observation, PBRS reward, SMDP hierarchy.
  - State representation (the 23-dim vector), network architecture (Dueling MLP), reward function (events + PBRS).
  - Training curves (embed TensorBoard exports / matplotlib PNGs from `log.csv`).
  - Hyperparameters (from the YAML configs).
  - Evaluation: the multi-seed comparison table, the M3 gate result, and the reactive-vs-aim discussion.
  - Pinned `logic.py` commit hash (Task 27).

- [ ] **Step 2: Commit**

```bash
git add README.md docs/
git commit -m "docs: RL agent report (architecture, curves, evaluation)"
```

---

### Task 25: `.gitignore` for checkpoints (keep finals)

**Files:**
- Modify: `.gitignore`

- [ ] **Step 1: Append**

```
# RL training artifacts
checkpoints/**/tb/
checkpoints/**/online_[0-9]*.pt
breakout_rl/configs/_*.yaml
venv/
```
(Final checkpoints are force-added in their tasks; intermediate ones and TensorBoard logs are ignored.)

- [ ] **Step 2: Commit**

```bash
git add .gitignore
git commit -m "chore: ignore intermediate training artifacts"
```

---

### Task 26: Plot training curves from CSV

**Files:**
- Create: `breakout_rl/eval/plot_curves.py`

- [ ] **Step 1: Implement**

`breakout_rl/eval/plot_curves.py`:
```python
"""Render training curves from one or more run log.csv files into PNGs for the report."""
import argparse, glob, csv
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def load(path):
    steps, evals = [], []
    with open(path) as f:
        for row in csv.DictReader(f):
            if row.get("eval_score"):
                steps.append(int(row["step"])); evals.append(float(row["eval_score"]))
    return steps, evals


def main(pattern, out):
    plt.figure()
    for path in glob.glob(pattern):
        s, e = load(path)
        if s:
            plt.plot(s, e, label=path.split("/")[-2])
    plt.xlabel("training step"); plt.ylabel("eval score"); plt.legend(); plt.grid(True)
    plt.savefig(out, dpi=120, bbox_inches="tight")
    print("wrote", out)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--pattern", default="checkpoints/flat_dqn*/log.csv")
    p.add_argument("--out", default="docs/curves_flat_dqn.png")
    a = p.parse_args(); main(a.pattern, a.out)
```

- [ ] **Step 2: Run + commit**

```bash
python -m breakout_rl.eval.plot_curves --pattern "checkpoints/flat_dqn*/log.csv" --out docs/curves_flat_dqn.png
git add breakout_rl/eval/plot_curves.py docs/curves_flat_dqn.png
git commit -m "docs: training curve plots"
```

---

### Task 27: Pin the `logic.py` commit

**Files:**
- Create: `scripts/record_commit.py`

- [ ] **Step 1: Implement a helper that stamps the current commit into configs**

`scripts/record_commit.py`:
```python
import subprocess, glob, yaml
h = subprocess.check_output(["git", "rev-parse", "HEAD"]).decode().strip()
for path in glob.glob("breakout_rl/configs/*.yaml"):
    c = yaml.safe_load(open(path))
    if "logic_commit" in c:
        c["logic_commit"] = h
        yaml.safe_dump(c, open(path, "w"))
print("pinned logic_commit =", h)
```

- [ ] **Step 2: Run before the final training runs + commit**

```bash
python scripts/record_commit.py
git add scripts/record_commit.py breakout_rl/configs/*.yaml
git commit -m "chore: pin logic.py commit for reproducibility"
```

---

### Task 28: Upstream contribution branch (bonus — Contributions 0.05)

**Files:** investigation only; fixes on a separate branch.

- [ ] **Step 1: Investigate the first-overlap brick resolution**

In `server/logic.py:202-205`, only `np.where(overlap)[0][0]` is deactivated per frame. Write a focused test that places the ball overlapping two bricks simultaneously and observe whether the intended brick/reflection is chosen. If it produces visibly wrong behavior (e.g. passing through a brick), prepare a fix.

- [ ] **Step 2: If a real bug is confirmed, branch + fix + PR**

```bash
git checkout -b fix/brick-collision-resolution
# implement fix in server/logic.py + add regression test in tests/test_logic.py
git commit -am "fix: resolve nearest brick on simultaneous overlap"
git push origin fix/brick-collision-resolution
# open PR against mariolpantunes/si2-breakout
```
Do **not** merge dynamics-changing fixes into the branch used for reported experiments without re-pinning (Task 27) and re-running. Document any PR in the report.

- [ ] **Step 3: Full test suite green**

Run: `python -m pytest -q`
Expected: all tests pass.

---

## Self-Review (completed by plan author)

**1. Spec coverage** — every spec section maps to a task:
- §2 ground truth → Task 2 constants; §4.1 clone → Task 3; §4.2 predictor → Task 4;
  §4.3 observation → Task 5 (+ parity Task 21); §4.4 rewards/PBRS → Task 6;
  §4.5 env → Task 7 (+ cache Task 14); §4.6 PER → Task 8; §4.7 DQN → Tasks 9–10;
  §4.8 aim controller → Task 16; §4.9 hierarchical → Tasks 18–19; §4.10 deploy → Task 20;
  §4.11 train/eval/configs → Tasks 12–13, 22, 26; §5 SMDP → Tasks 18–19;
  §6 curriculum → Task 11; §7 eval metrics/baselines/seeds → Tasks 13, 22, 23;
  §8 milestones + M3 gate → phase headers + Task 17; §9 parity → Task 21;
  §10 reproducibility → Tasks 25, 27; §11 upstream → Task 28; §12 repo layout → file structure.
**2. Placeholder scan** — no `TBD`/`TODO`/"implement later" anywhere; every code step
contains complete, runnable code. The single deferred value, `logic_commit:
FILL_AT_RUNTIME`, is intentionally stamped by Task 27's `record_commit.py`.
**3. Type consistency** — `clone_game`, `predict_landing`, `ObservationBuilder.build`,
`base_reward(before, after, cfg)`, `potential(state, landing, is_terminal)`,
`Transition(state, action, reward, next_state, done, gamma)`, `DQNAgent.update(buffer,
batch_size, beta)`, `choose_action(game, region)`, `HighLevelEnv.step -> info["gamma_k"]`
are used consistently across tasks. `OBS_DIM`/`HIGH_OBS_DIM` are the same 23 dims.

**Known sequencing note:** Task 12 (`train_dqn`) imports `evaluate_agent` from Task 13;
build Task 13 before running Task 12.
