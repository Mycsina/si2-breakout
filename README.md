# <img src="server/viewer/favicon.svg" alt="logo" width="128" height="128" align="middle"> SI2 - Breakout

A Breakout game implementation using the `ai-game-framework`.

## Features
- Real-time backend server.
- Web-based viewer with Canvas API.
- Dummy agent (ball tracker).
- Manual agent (terminal-based A/D control).

## Setup & Running the Game

### 1. Prerequisites
- Python 3.10+ installed on your host.

### 2. Create and Activate Virtual Environment
Create a virtual environment (`venv`) to isolate dependencies:
```bash
python3 -m venv venv
source venv/bin/activate
```

### 3. Install Dependencies
Install the required packages (this will install the local `ai-game-framework` package in editable mode and `numpy`):
```bash
pip install -r requirements.txt
```

### 4. Run the Game Server
Start the backend server (which also serves the frontend web viewer):
```bash
python3 -m server.server
```

### 5. Open the Viewer
Open your web browser and navigate to:
```
http://localhost:8765/
```

### 6. Run the Agents
In a separate terminal (ensure the virtual environment is activated):

- **Dummy Agent (Ball Tracker)**:
  ```bash
  python3 -m agents.dummy_agent
  ```

- **Manual Agent (Terminal A/D control)**:
  ```bash
  python3 -m agents.manual_agent
  ```

## Development
The project structure:
- `server/`: Game logic, server implementation, and visualizer assets (inside `server/viewer/`).
- `agents/`: Autonomous and manual agent implementations.
- `tests/`: Game unit tests.

---

# Reinforcement Learning Agent — Project Report (SI2)

This section is the project report. It documents the autonomous Breakout agent(s)
we built, the architecture, the state/model/reward design, how to run everything,
and the experimental findings.

## 1. What we built

Two agents that share one codebase (`breakout_rl/`):

1. **Flat DQN** — a from-scratch **Double + Dueling DQN with Prioritized Experience
   Replay (PER)**. A reactive policy that maps the full game observation directly to a
   paddle action (`NOOP`/`WEST`/`EAST`). This is the baseline "spine".
2. **Hierarchical SMDP agent** — the "showpiece". A **learned high-level policy** chooses
   a *contact region* (LEFT / CENTER / RIGHT third of the paddle) once per ball descent;
   a **physics-based low-level controller** then moves the paddle to make the ball strike
   that region. Trained with semi-Markov (options) Q-learning.

Both reuse the same network, replay buffer, observation builder, and rollout predictor.

## 2. How to run

This project uses [`uv`](https://docs.astral.sh/uv/). The Python environment (incl.
`torch`, `gymnasium`, `tensorboard`, `pytest`) is declared in `pyproject.toml` /
`uv.lock`. A `requirements-rl.txt` mirrors the dependency list for pip users.

```bash
uv sync                                   # create .venv and install everything
.venv/bin/python -m pytest -q             # run the test suite (47 tests)
```

All commands below assume `.venv/bin/python` (no activation needed). On a GPU box the
flat-DQN trainer uses CUDA automatically if available; otherwise it runs on CPU.

```bash
# --- Train the reactive flat DQN (writes checkpoints/flat_dqn/) ---
.venv/bin/python -m breakout_rl.train.train_dqn --config breakout_rl/configs/flat_dqn.yaml

# --- M3 aiming-authority probe (the gate experiment; CPU, ~7 min for 100 episodes) ---
.venv/bin/python -m breakout_rl.eval.probe_aiming_authority --episodes 100

# --- Train the hierarchical SMDP aim policy (writes checkpoints/aim_smdp/) ---
.venv/bin/python -m breakout_rl.train.train_aim --config breakout_rl/configs/aim.yaml

# --- Multi-seed training + comparison table ---
bash scripts/train_seeds.sh
.venv/bin/python -m breakout_rl.eval.evaluate --episodes 30 --seeds 3
.venv/bin/python -m breakout_rl.eval.plot_curves --pattern "checkpoints/flat_dqn*/log.csv" --out docs/curves_flat_dqn.png

# --- Deploy a trained agent against the live WebSocket server ---
.venv/bin/python -m server.server                                   # terminal 1
.venv/bin/python -m breakout_rl.deploy.trained_agent --mode flat \
    --checkpoint checkpoints/flat_dqn/online_final.pt               # terminal 2
.venv/bin/python -m breakout_rl.deploy.trained_agent --mode hier \
    --checkpoint checkpoints/aim_smdp/online_final.pt               # hierarchical
# then open http://localhost:8765/
```

## 3. Architecture

The central design decision is to **train headless**: we import the pure, synchronous
`server.logic.Breakout` directly into a Gymnasium environment, so training runs at
thousands of steps/sec instead of being throttled by the 30 fps WebSocket loop. The
WebSocket `BaseAgent` is used only for deployment.

- **`breakout_rl/env/breakout_env.py`** — `BreakoutEnv(gym.Env)`. One step = apply action
  → `game.update(1/30)` → compute reward. `Discrete(3)` actions, `Box(23)` observation.
- **`breakout_rl/physics/predictor.py`** — `predict_landing(game)`. The ball-landing
  predictor *is the simulator itself*: it clones the game and rolls `update()` forward to
  the paddle line. This is **bit-exact** with the real environment (identical wall and
  brick collisions), so there is no separate, drift-prone physics model to maintain.
- **`breakout_rl/env/observation.py`** — `ObservationBuilder`, **shared by training and
  deployment** so the two cannot skew (guarded by `tests/test_observation_parity.py`).
  Velocity is unavailable on the wire, so it is recovered by finite difference over the
  **measured** `dt`.
- **`breakout_rl/env/high_level_env.py`** — `HighLevelEnv`, the SMDP options wrapper.
- **`breakout_rl/controllers/aim_controller.py`** — the physics-based low-level controller.

## 4. State representation (23-dim vector)

Built only from the `get_state()` wire format (positions, not internal velocity):

| idx | feature | normalization |
|---|---|---|
| 0 | paddle x | `paddle_x / (width - paddle_width)` |
| 1 | ball x | `ball_x / width` |
| 2 | ball y | `ball_y / height` |
| 3 | ball vx | finite-difference `Δx/dt`, scaled by `1/300` |
| 4 | ball vy | finite-difference `Δy/dt`, scaled by `1/300` |
| 5 | lives | `lives / 3` |
| 6 | bricks remaining | `count / 16` |
| 7–22 | brick occupancy | 16-dim 0/1 grid (which bricks are alive) |

## 5. Network and reward

- **Network** (`breakout_rl/agents/networks.py`): a **Dueling MLP** — a 2-layer ReLU torso
  feeding separate value `V(s)` and advantage `A(s,a)` heads, combined as
  `Q = V + (A − mean_a A)`. Trained with **Double DQN** targets and **PER**-weighted MSE.
- **Reward** (`breakout_rl/env/rewards.py`):
  - *Event reward*: `+score_delta` per step (i.e. +3/brick, +100/board clear), `−30` on
    losing a life, an extra `−50` on game over, and a small `−0.01` step cost.
  - *Potential-based shaping (PBRS)*: `Φ(s) = −|paddle_center − predicted_landing_x| / width`
    while descending, `0` while ascending and `0` on terminal states. Added as
    `F = γ·Φ(s') − Φ(s)`. PBRS is **policy-invariant** (it cannot change the optimal
    policy), so it accelerates learning without biasing the solution — the terminal-`0`
    rule and the telescoping identity are unit-tested.

## 6. Curriculum

Two stages (`breakout_rl/train/curriculum.py`): Stage 1 is *easy* (paddle width 120, ball
speed 200) to bootstrap; Stage 2 switches to the *real* difficulty (width 80, speed 300)
and runs long enough that the final evaluation reflects real conditions.

## 7. Evaluation metrics and the M3 aiming-authority gate

Plain survival/score **saturate** and fail to discriminate good agents, so we report
**board-clears** and **bricks-per-life** as the primary metrics, against baselines
(random, analytic-center, flat DQN, hierarchical), over multiple seeds.

Before committing to the full hierarchy we ran a **gate experiment** (`probe_aiming_authority.py`)
comparing three scripted policies — all of which *move the paddle to intercept the ball*,
differing only in **which third** of the paddle the ball contacts:

| policy | board clears | bricks/life |
|---|---|---|
| center (always bounce straight up) | 0.24 | 5.01 |
| random region | 2.31 | 15.79 |
| oracle region (greedy 1-volley brute force) | 2.09 | 14.57 |

*(100 episodes, seed 0, real difficulty.)*

**Findings.** (1) **Aiming authority is large**: steering the contact region gives ~**3×**
the bricks-per-life and ~**9×** the clears of the center-only baseline (oracle is **+191%**
over center, far above the +25% gate threshold). A straight-up bounce just re-breaks the
same vertical column; spreading the bounce angle covers the whole board. (2) The greedy
**oracle is *not* better than random** region choice — the stochastic ±bounce band
(LEFT ≈ −45°…−15°, CENTER ≈ ±5°, RIGHT ≈ 15°…45°) means precise per-volley optimization
does not pay off; what matters is simply *not always going straight up*. The implication
for the learned hierarchy: its value must come from **board-aware, multi-step credit
assignment** (it observes the 16-brick occupancy grid and is trained with SMDP backups),
not from myopic single-volley aiming.

> **Note on a corrected decision-point detector.** The original plan detected a "decision
> point" via a velocity sign-flip (`prev_vy ≤ 0 and vy > 0`). In this **gravity-free**
> physics the ball's velocity only flips at collisions (all above `y=140`), so that test
> *never fired* and the region was never re-chosen — every probe policy collapsed to
> "center". We replaced it with a **downward crossing of the decision line**
> (`prev_ball_y ≤ 140 < ball_y`, descending), matching the spec's intent ("clean descent
> below the brick field"). This is used in both the probe and `HighLevelEnv`; SMDP option
> lengths are now sane (k ≈ 60–80 primitive steps per option).

## 8. Hyperparameters

From `breakout_rl/configs/flat_dqn.yaml` (flat) and `aim.yaml` (SMDP):

| | flat DQN | SMDP aim |
|---|---|---|
| horizon | 400k steps | 60k options |
| γ | 0.99 | 0.99 |
| learning rate | 1e-3 | 5e-4 |
| hidden units | 128 | 128 |
| replay capacity | 100k | 50k |
| batch size | 128 | 128 |
| target sync | every 2000 | every 1000 |
| ε schedule | 1.0 → 0.05 over 100k | 1.0 → 0.05 over 20k |
| PER β | 0.4 → 1.0 | 0.4 → 1.0 |
| curriculum switch | step 120k | option 20k |
| reward | score / −30 life / −50 game-over / −0.01 step | same |

## 9. Training curves and final comparison

> **To be populated from a full GPU training run.** The code, configs, and plotting/eval
> scripts are complete and the pipeline is validated end-to-end (a 2.5k-step smoke run
> trains, evaluates, and checkpoints without error). The 400k-step flat-DQN run and the
> 60k-option SMDP run require a GPU and a few hours; run the commands in §2, then paste:
> - `docs/curves_flat_dqn.png` (training curves from `plot_curves.py`),
> - the multi-seed comparison table printed by `evaluate.py` (and `checkpoints/comparison.csv`),
> - a one-line note on the deploy visual check (paddle tracks ball, score climbs).

## 10. Reproducibility

- The exact `server/logic.py` commit used for an experiment is pinned in each config's
  `logic_commit` field (stamped by `scripts/record_commit.py`).
- Training seeds everything (`random`, `numpy`, `torch`); evaluation uses **held-out**
  seeds (`10_000 + step`) disjoint from training.
- Intermediate checkpoints / TensorBoard logs are gitignored; only `online_final.pt`
  checkpoints are kept (force-added).

## 11. Upstream contribution — PR: fix brick-collision resolution

A reviewing pass over `server/logic.py` found a real collision bug. Below is the proposed
pull request to `mariolpantunes/si2-breakout` (verified locally; **not** applied to the
branch used for the reported experiments — see the caveat at the end).

**Title:** `fix: resolve all overlapped bricks and prevent corner-graze tunnelling`

### Symptom
When the ball overlaps **two bricks in the same physics frame** — e.g. straddling the
10 px gap between two bricks in a row — only one brick is removed and, in a reproducible
case, **the ball passes straight through the second brick without breaking it or
bouncing**.

### Root cause (`server/logic.py`, the brick-collision block, ~lines 202–226)
Two issues compound:

1. **First-in-array resolution.** The hit brick is chosen as `np.where(overlap)[0][0]` —
   the *lowest index*, not the brick actually being contacted — and only that single
   brick is deactivated. A genuinely-overlapped neighbour survives the frame.
2. **Penetration-only axis selection.** The reflection axis is picked purely by
   `overlap_x < overlap_y`. On a corner graze the horizontal overlap can be a few pixels
   while the ball is travelling almost purely vertically (`ball_vx ≈ 0`). The code then
   flips `vx` (a no-op when `vx == 0`) and leaves `vy` untouched, so the ball continues
   upward through the surviving neighbour — a tunnelling artifact.

### Reproduction (deterministic; uses no RNG)
```python
from server.logic import Breakout
g = Breakout()
for b in g.bricks:
    b.active = (b.index in (0, 1))   # keep only row-1 bricks 0 (105–175) and 1 (185–255)
g._sync_bricks_to_numpy()
# ball centred in the 10px gap (x=180 spans 172–188 -> overlaps both), moving straight up
g.ball_x, g.ball_y, g.ball_vx, g.ball_vy = 180.0, 83.0, 0.0, -100.0
g.update(0.05)
print(g.bricks[0].active, g.bricks[1].active, g.ball_vy)
# BEFORE FIX: False  True  -100.0   <- brick 1 survives, ball keeps going up (tunnels)
# AFTER  FIX: False  False +100.0   <- both break, ball bounces down
```

### Fix
Resolve **every** overlapped brick in the frame, reflect off the **deepest** (true
contact) brick, and only flip a velocity component that is actually carrying the ball
into the brick:

```python
overlap_idxs = np.where(overlap)[0]

# deepest-penetration brick = the true contact (not array order)
ox_all = np.minimum(self.ball_x + self.ball_radius - lefts,
                    rights - (self.ball_x - self.ball_radius))
oy_all = np.minimum(self.ball_y + self.ball_radius - tops,
                    bottoms - (self.ball_y - self.ball_radius))
pen = np.minimum(ox_all, oy_all)
pen[~overlap] = -np.inf
hit_idx = int(np.argmax(pen))

# deactivate EVERY overlapped brick so the ball cannot tunnel through a neighbour
for hit in overlap_idxs:
    bidx = int(self.brick_array[hit, 5])
    self.brick_array[hit, 4] = 0.0
    self.bricks[bidx].active = False
    self.score += 3

b_left, b_right = lefts[hit_idx], rights[hit_idx]
b_top, b_bottom = tops[hit_idx], bottoms[hit_idx]
overlap_x = min(self.ball_x + self.ball_radius - b_left, b_right - (self.ball_x - self.ball_radius))
overlap_y = min(self.ball_y + self.ball_radius - b_top, b_bottom - (self.ball_y - self.ball_radius))

# only flip a component that is moving the ball INTO the brick (no corner-graze tunnel)
if overlap_x < overlap_y and self.ball_vx != 0.0:
    self.ball_vx = -self.ball_vx
else:
    self.ball_vy = -self.ball_vy
```

### Regression test (add to `tests/test_logic.py`)
```python
def test_simultaneous_two_brick_overlap_breaks_both_and_bounces():
    g = Breakout(width=600, height=400)
    for b in g.bricks:
        b.active = (b.index in (0, 1))
    g._sync_bricks_to_numpy()
    g.ball_x, g.ball_y, g.ball_vx, g.ball_vy = 180.0, 83.0, 0.0, -100.0
    g.update(0.05)
    assert g.bricks[0].active is False
    assert g.bricks[1].active is False          # neighbour no longer survives
    assert g.ball_vy > 0                         # ball bounces down, does not tunnel
```

### Verification & impact
Locally verified: the fix breaks both bricks and reflects the ball downward, and the
existing single-brick collision test (`test_brick_collision_and_scoring`) is unchanged
(score 103, brick deactivated, board-clear respawn). The change is small, has no extra
dependencies, and only alters behaviour in the multi-overlap / corner-graze edge case.

**Caveat (reproducibility):** this fix changes game dynamics, so it is **not** merged into
the branch used for the reported experiments. If it is accepted upstream and adopted here,
re-pin `logic_commit` (`scripts/record_commit.py`) and re-run training before updating the
numbers in §9.

## 12. Repository layout (RL)

```
breakout_rl/
  constants.py            action/region enums + fixed constants
  physics/clone.py        deep-enough game clone for rollouts
  physics/predictor.py    bit-exact ball-landing predictor
  env/observation.py      shared 23-dim observation builder
  env/rewards.py          event reward + PBRS potential
  env/breakout_env.py     Gymnasium env (flat DQN)
  env/high_level_env.py   SMDP options wrapper (hierarchy)
  agents/replay.py        prioritized experience replay
  agents/networks.py      dueling MLP
  agents/dqn_agent.py     Double-DQN agent (reused by both levels)
  controllers/aim_controller.py   physics-based low-level aim
  train/                  curriculum + flat/SMDP training entrypoints
  eval/                   metrics, multi-seed comparison, probe, plots
  deploy/trained_agent.py WebSocket deploy (flat + hierarchical)
  configs/                flat_dqn.yaml, aim.yaml
tests/                    47 tests covering every module
```