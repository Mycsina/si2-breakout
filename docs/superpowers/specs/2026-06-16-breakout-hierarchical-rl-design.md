# Hierarchical RL Breakout Agent — Design Spec

- **Status:** Approved design (post-review). Ready for implementation planning.
- **Date:** 2026-06-16
- **Course:** SI2 project (deadline 2026-06-22). Fork: `Mycsina/si2-breakout`.
- **Audience:** This spec is written to be handed to lower-capability implementation
  agents. Module boundaries, interfaces, and acceptance criteria are explicit so that
  individual tasks can be executed with minimal cross-context reasoning.

---

## 1. Objective and grading alignment

Build an autonomous Breakout agent that moves the paddle (WEST / EAST / NOOP) to keep
the ball in play, clear brick columns, and maximize surviving laps (board-clears).

We commit to **two agents** forming the report's narrative spine:

1. **Flat DQN (the spine):** pure reactive RL, *no physics knowledge*. Double + Dueling
   DQN with Prioritized Experience Replay (PER), implemented from scratch in PyTorch.
2. **Hierarchical agent (the showpiece):** a *learned* high-level "aim" policy
   (SMDP / options) sitting on top of a *physics-based* low-level controller that uses
   the game's own dynamics to position the paddle.

Mapping to grading: Solution 0.30 (two working agents + strong eval), Code 0.20
(modular, tested), Repository 0.20 (clean layout + checkpoints + configs), Complexity
0.15 (SMDP hierarchy + provable PBRS reward), Report 0.10 (multi-seed curves + the
reactive-vs-aim contrast), Contributions 0.05 (upstream bug-fix PRs).

---

## 2. Environment ground truth (from `server/logic.py` — do not re-derive)

All constants below are authoritative as of the pinned commit (see §11). Lower-capability
agents must **source these from the live game object, not hard-code literals**, because
the curriculum (§7) changes some of them at runtime.

- Board: `width = 600`, `height = 400`.
- Paddle: `paddle_width = 80`, `paddle_height = 10`, `paddle_y = 380`. Moves in discrete
  **25 px** steps per `move_paddle` call. `paddle_x` clamped to `[0, width - paddle_width]`
  = `[0, 520]` at default width.
- Ball: `ball_radius = 8`, `ball_speed = 300` px/s (speed magnitude is **conserved** —
  energy is constant; only direction changes).
- Tick: server `fps = 30` → `dt = 1/30 s`. Ball moves ~10 px/frame; max horizontal ball
  speed `300·sin(45°) ≈ 212 px/s` ≈ 7 px/frame, so the paddle (750 px/s) easily outruns
  the ball horizontally. Control is feasible.
- Bricks: 16 total, indices 0–15.
  - Row 1 (top): 5 bricks, `left = 105 + i·80`, `top = 60`, `w = 70`, `h = 15` (i=0..4).
  - Row 2 (mid): 6 bricks, `left = 65 + i·80`, `top = 85` (i=0..5).
  - Row 3 (bot): 5 bricks, `left = 105 + i·80`, `top = 110` (i=0..4).
  - Lowest brick bottom = 125. Cleared boards respawn when `ball_y > 140`.
- Scoring: **+3** per brick; **+100** on full board clear (then `checkpoint_score`
  is saved and bricks respawn); on death `score` resets to `checkpoint_score`.
- Lives: 3. Game over at 0 lives.
- **Paddle bounce is stochastic** and region-dependent (relative x = `ball_x - paddle_x`):
  - left third (`< W/3`): outgoing angle `~U(-45°, -15°)`
  - center third: `~U(-5°, 5°)`
  - right third (`> 2W/3`): `~U(+15°, +45°)`
  - then `vx = speed·sin(angle)`, `vy = -speed·cos(angle)`.
  - **Implication:** choosing a contact region selects a *distribution* over outgoing
    directions, not a precise angle. This caps aiming precision (see §9 probe).

### What is on the WebSocket wire (`get_state()` output)

`width, height, paddle_x, paddle_y, paddle_width, paddle_height, ball_x, ball_y,
ball_radius, lives, score, high_score, game_over, bricks, actions/valid_actions`,
plus `player_id` added by the server.

- `bricks` is the list of **active bricks only**, each `{index, left, top, width, height,
  active}`. The observation builder must reconstruct the full 16-dim occupancy by index.
- **`ball_vx` / `ball_vy` are NOT transmitted.** Velocity must be recovered from
  successive frames (see §4). This partial observability is intrinsic to the task.
- **No `dt` is transmitted.** The deploy agent must measure wall-clock inter-frame `dt`.

---

## 3. Architecture overview

```
                    logic.Breakout  (owned, fast, synchronous)
                          │  clone() / rollout
                          ▼
   physics/predictor.py ──── exact ball-landing prediction (rollout, bit-exact)
        │                         │
        │ (Φ, aim target)         │
        ▼                         ▼
   env/rewards.py            controllers/aim_controller.py ── target paddle_x → WEST/EAST/NOOP
        │                         ▲
        ▼                         │ chosen region {L,C,R}
   env/breakout_env.py  ◄── env/observation.py ──► deploy/trained_agent.py (WebSocket)
        │   (Gymnasium)        (SHARED, deploy-faithful)         │
        ▼                                                        ▼
   agents/flat_dqn.py (Double+Dueling+PER)        agents/hierarchical_agent.py (SMDP high-level
        │                                          reuses the same DQN class; low-level = aim_controller)
        ▼
   train/ , eval/ , configs/ , checkpoints/
```

Key principle: **the env and the deploy agent build observations through the same
module**, and **the predictor is the simulator itself** (no parallel physics model).

---

## 4. Components and interfaces

Each component is a separately testable unit. Signatures are normative.

### 4.1 `logic.Breakout.clone()` (small addition to game state handling)

The predictor needs to roll the game forward without mutating the live game. Provide a
**clone** that copies all mutable state: ball pos/vel, paddle_x, lives, score,
checkpoint_score, game_over, bricks_need_respawn, `np.copy(brick_array)`, and each
`Brick.active`. Geometry (brick positions, sizes) is immutable and may be shared.

- Preferred: add `Breakout.clone() -> Breakout` in our own code (a helper that constructs
  a new `Breakout` and copies fields) rather than editing upstream — keeps the upstream
  bugfix track (§12) independent. If a `clone()` is added upstream, depend on it.
- Optimization (optional, only if profiling shows a bottleneck): a `snapshot()/restore()`
  pair that saves/restores just the mutable fields in place, avoiding allocation. Default
  to `clone()` for correctness and clarity.

### 4.2 `physics/predictor.py` — rollout-based, bit-exact

```python
def predict_landing(game: Breakout) -> float | None:
    """Return the ball_x at which the ball will reach the paddle line
    (ball_y + ball_radius >= paddle_y), assuming NO paddle input.
    Returns None if the ball is currently ascending (vy <= 0)."""
```

- Implementation: `g = game.clone()`; loop `g.update(dt)` until
  `g.ball_y + g.ball_radius >= g.paddle_y` (descending) or `g.game_over`; return
  `g.ball_x`. Cap iterations (e.g. 200) as a safety net.
- **Bit-exact by construction:** it reuses `Breakout.update`, so wall reflections, brick
  AABB resolution, and contact geometry are identical to the real environment. There is
  no analytic model to keep in sync.
- The rollout is deterministic up to the paddle line: launch randomness already happened,
  paddle randomness is not reached (we stop before paddle contact), and brick reflections
  are deterministic. Brick interactions *are* accounted for because the rollout destroys
  bricks on the clone exactly as the real game would.
- **Re-plan trigger (caller responsibility):** even though the prediction is exact, the
  *high-level decision* must be re-evaluated when the board changes — i.e. on brick
  contact and at each new descent — because the best aim depends on remaining bricks.
- **Caching:** during a single uninterrupted descent the prediction is constant, so the
  env caches it and invalidates on (vy sign flip, brick destroyed, paddle contact, life
  loss). Caching == recomputing (deterministic), so it does not affect correctness of Φ.

Acceptance test: construct a known descending state with no bricks in the path; assert
`predict_landing` equals the value obtained by manually stepping `update` to the paddle
line. Construct a state with a brick in the path; assert the prediction reflects the
post-brick trajectory.

### 4.3 `env/observation.py` — SHARED, deploy-faithful

```python
class ObservationBuilder:
    def reset(self) -> None: ...
    def build(self, state: dict, dt: float) -> np.ndarray:
        """state is the get_state() dict (active-only bricks). dt is the real
        inter-frame interval (1/30 in training, measured at deploy)."""
```

Feature vector (all normalized to ~[-1, 1] or [0, 1]):

1. `paddle_x / (width - paddle_width)`
2. `ball_x / width`, `ball_y / height`
3. velocity via finite difference using the **measured dt**:
   `vx = ((ball_x - prev_ball_x) / dt) / ball_speed`, likewise `vy`. First frame → 0.
4. 16-dim brick occupancy: index `i` → 1.0 if a brick with that index is in `bricks`
   else 0.0 (reconstructs full grid from the active-only wire list).
5. `lives / 3`, `bricks_remaining / 16`.

≈ 23 dims. A small MLP suffices.

- **Why explicit velocity rather than raw frame-stack:** dividing Δposition by the
  *measured* dt makes the feature invariant to frame-timing jitter at deploy (the
  reviewer's point — do not assume exactly 1/30 on the wire).
- **Known limitation (documented):** the finite-difference velocity on the single frame
  that straddles a wall/paddle/brick bounce is wrong (it averages pre/post-bounce). This
  is a transient and generally harmless; note it when diagnosing odd post-contact behavior.

### 4.4 `env/rewards.py` — base reward + provable PBRS

```python
def base_reward(prev_state, state) -> float
def potential(state, predicted_landing_x, is_terminal: bool) -> float
def shaped_reward(prev_state, state, gamma, prev_potential, potential) -> float
```

- **Event/base reward:** `+3` per brick destroyed this step, `+100` on board clear,
  large negative on life loss (e.g. `-30`), terminal negative on game over (e.g. `-50`),
  small per-step cost (e.g. `-0.01`) to discourage dithering.
- **Potential-based shaping (PBRS), policy-invariant:**
  - `Φ(s)` is a pure function of state:
    - descending (`vy > 0`): `Φ(s) = -|paddle_center(s) - predict_landing_x(s)| / width`
    - ascending (`vy <= 0`): `Φ(s) = 0`
    - terminal: `Φ(s) = 0` (**mandatory** for policy invariance).
  - Shaping term `F = γ·Φ(s') - Φ(s)`, added to base reward.
- **Correctness conditions (must be enforced and unit-tested):**
  1. `Φ` depends only on state (descend/ascend is read from inferred `vy` sign, which is
     a state feature) — never on time or history.
  2. `Φ(terminal) = 0`; at the game-over transition compute `F = γ·0 - Φ(s) = -Φ(s)`, and
     do **not** bootstrap value past the terminal state.
  3. The descend/ascend gate switches Φ on/off, but because Φ=0 during ascent is itself a
     valid function of state, no non-potential term is smuggled in. Unit-test that the
     summed shaping over any closed loop of states telescopes to `γ^n·Φ(end) - Φ(start)`.

### 4.5 `env/breakout_env.py` — Gymnasium environment

- `action_space = Discrete(3)` → {WEST, EAST, NOOP}.
- `observation_space` = Box of the §4.3 vector.
- `step(action)`: `move_paddle` (unless NOOP) → `game.update(1/30)` → compute base + PBRS
  reward → build observation → `terminated = game.game_over`; `truncated` at `max_steps`.
- `reset()`: re-init game, apply curriculum params (§7) **after** `reset_game()`, reset
  the `ObservationBuilder`, reset the predictor cache.
- Holds the predictor cache and invalidation logic (§4.2).
- Supports vectorized envs (multiple game instances) for throughput.

### 4.6 `agents/replay.py` — Prioritized Experience Replay

Standard proportional PER (sum-tree), with importance-sampling weights and β-annealing.
Reused by both the flat DQN and the high-level SMDP buffer (the high-level stores
`γ^k` instead of a fixed γ — see §5).

### 4.7 `agents/flat_dqn.py` — Double + Dueling DQN (from scratch, PyTorch)

- MLP torso → dueling heads (value stream `V(s)` + advantage stream `A(s,a)`),
  combined as `Q = V + (A - mean_a A)`.
- Double-DQN target: action selected by online net, evaluated by target net.
- Target network with periodic/soft updates; ε-greedy exploration with annealing.
- Single GPU; this network is tiny (~23-dim input), so throughput is bounded by env
  stepping, not the net. Vectorize envs accordingly.

### 4.8 `controllers/aim_controller.py` — physics-based low-level

Given a target contact region `r ∈ {LEFT, CENTER, RIGHT}` and `predict_landing(game)`:

- region center offsets (relative to paddle left edge): LEFT = `W/6`, CENTER = `W/2`,
  RIGHT = `5W/6`, where `W = game.paddle_width` (**read from the game**, not a literal).
- target paddle_x = `landing_x - region_center_offset`, clamped to `[0, width - W]`.
- If the clamp moves the target (ball too near a wall to realize the region), **fall back
  to CENTER** (`landing_x - W/2`) so the paddle at least intercepts; if even that clamps,
  minimize the miss distance.
- Emit WEST / EAST / NOOP to step `paddle_x` toward the target (25 px steps; NOOP when
  within half a step).

### 4.9 `agents/hierarchical_agent.py` — SMDP high-level (§5)

Reuses the `flat_dqn` network class. Action space = {LEFT, CENTER, RIGHT}. Observation =
brick occupancy grid + ball state (it does not need raw paddle control features; the
low-level handles those).

### 4.10 `deploy/trained_agent.py` — WebSocket deployment

- `TrainedAgent(BaseAgent)` overriding `deliberate()`.
- Loads a checkpoint; instantiates the **same** `ObservationBuilder` as the env.
- Measures inter-frame `dt` with `time.monotonic()` between received states.
- For the flat agent: obs → argmax Q → action dict `{"action": "move", "direction": ...}`
  or no-op (send nothing for NOOP).
- For the hierarchical agent: maintain a lightweight local `Breakout` mirror seeded from
  the wire state to run `predict_landing` for the low-level controller (the controller
  needs a game object to roll out). The high-level net picks the region; the controller
  picks the primitive action.

### 4.11 `train/` , `eval/` , `configs/`

- `train/train_dqn.py`, `train/train_aim.py`: config-driven (YAML/dataclass), checkpoint
  every N steps to `checkpoints/`, log reward/score/loss/ε/eval-score to TensorBoard + CSV.
- `eval/evaluate.py`: runs the eval protocol (§8), writes a comparison table + plots.
- `configs/*.yaml`: hyperparameters, curriculum schedule, seeds, pinned-commit hash.

---

## 5. SMDP / options formulation (the substance of "hierarchical")

The high-level decides **once per ball-descent cycle**; that decision (an *option*) runs
for `k` primitive 30 Hz steps. Naive single-step Bellman backup on the high-level Q is
**wrong** — it must be SMDP Q-learning.

- **Option initiation:** at each decision point — when the ball begins descending toward
  the paddle (or immediately after a paddle contact). The high-level observes the board +
  ball state and picks region `o ∈ {LEFT, CENTER, RIGHT}`.
- **Option execution:** the low-level controller (§4.8) drives the paddle to realize `o`
  until the option terminates.
- **Option termination:** next paddle contact, life loss, or game over.
- **SMDP transition stored:** `(s_o, o, R_o, γ^k, s_o', done)` where
  - `R_o = Σ_{i=0}^{k-1} γ^i · r_{t+i}` — discounted sum of primitive rewards during the
    option (use the same per-step rewards from §4.4),
  - `k` = option duration in primitive steps,
  - `s_o'` = state at option end.
- **SMDP update:**
  `Q(s_o, o) ← Q(s_o, o) + α [ R_o + γ^k · max_{o'} Q(s_o', o') - Q(s_o, o) ]`
  (drop the bootstrap term when `done`). Double/Dueling applies as in §4.7; PER stores
  `γ^k` per sample.

This bookkeeping (accumulate discounted reward over the option, bootstrap with `γ^k`) is
the single highest-correctness-risk item and must be implemented and tested **before**
any hyperparameter tuning.

---

## 6. Curriculum

A staged schedule for a clean convergence story (set on the `game` object **after**
`reset_game()` each episode):

- **Stage 1 (easy):** widen paddle (e.g. `paddle_width = 120`) and/or slow the ball
  (e.g. `ball_speed = 200`). Interception is easy → the agent first learns to track.
- **Stage 2 (real):** anneal to real settings (`paddle_width = 80`, `ball_speed = 300`).
  **Stage 2 must run long enough that final eval reflects real difficulty**, not
  curriculum-inflated numbers.

All geometry-dependent formulas (aim offsets, clamps) read current `W`/`width` from the
game so they remain correct as Stage 1 changes them.

---

## 7. Evaluation protocol and metrics

**Trap to avoid:** survival saturates. An analytic agent that just centers the paddle on
`predict_landing` essentially never dies in single-ball Breakout, so survival/score will
not differentiate the hierarchical agent. The hierarchy can only win on **reaching
specific bricks faster**. Foreground:

- **Board-clears per game (laps)** — primary.
- **Steps-to-clear** (mean steps to clear a board) and **steps-to-first-clear**.
- **Bricks-cleared-per-life.**
- **Tunnel-strategy discovery** (heuristic detector: a full column cleared / ball reaching
  above the brick field) — bonus signal that aiming is being learned.
- Secondary (expected to saturate): mean game score, survival steps.

**Baselines for the comparison table:**

1. Random / dummy agent (floor).
2. **Analytic-centering controller** (`predict_landing` → center paddle) — the *survival
   ceiling*; its purpose is to show survival saturates and force the hierarchy to compete
   on clearing metrics.
3. Flat DQN.
4. Hierarchical agent.

**Statistics:** RL is high-variance. Train **3–5 seeds per agent**; report mean ± std with
error bars in the table and on curves. Keep the bounce stochastic during training; use
**held-out RNG seeds** for eval. ≥ 30 games per (agent, seed) at eval.

---

## 8. Milestones and risk gates

- **M0 — Setup:** install `ai-game-framework` (not currently installed), create venv, run
  `python -m server.server` + dummy agent, run `tests/test_logic.py` green.
- **M1 — Foundation:** `clone()` + `predict_landing` (+ tests), `ObservationBuilder` (+
  parity test §10), `BreakoutEnv` + reward/PBRS (+ PBRS telescoping test).
- **M2 — Flat DQN:** train to convergence on the curriculum; record baseline metrics and
  curves. Establishes the convergence story regardless of what happens to the hierarchy.
- **M3 — Aiming-authority probe (GATE, do before building the hierarchy):** a *scripted*
  experiment. With a brute-force "perfect" high-level (try each region, measure resulting
  clears under the stochastic bounce), quantify how much clearing advantage aiming buys
  over the centering baseline. The ~30° stochastic outgoing band caps placement precision,
  so the achievable advantage may be modest.
  - **Decision:** large advantage → build the full SMDP hierarchy (M4). Marginal advantage
    → pivot to *implicit* aiming via reward shaping inside the flat DQN, and document the
    finding (a legitimate, well-supported result for the report).
- **M4 — Hierarchy:** SMDP high-level (§5) + analytic low-level; train; compare on §7
  metrics across seeds.
- **M5 — Deploy & report:** `TrainedAgent` over WebSocket for both agents; record a demo;
  finalize README/report with multi-seed curves and the comparison table.

---

## 9. Train/deploy parity (anti-skew)

- The `ObservationBuilder` (§4.3) is the single source of truth, imported by both
  `env/` and `deploy/`.
- **Parity test:** feed one identical `get_state()` dict (and an identical preceding
  frame for velocity) through the env path and the deploy path; assert byte-identical
  observation vectors.
- Confirm at M1 that the wire actually carries the brick grid and ball position at the
  precision the features need (it does per §2; verify empirically against a running
  server), and that deploy velocity uses **measured** dt.

---

## 10. Reproducibility

- **Pin the `logic.py` commit** used for the reported experiments; record its hash in the
  config and the report. If the upstream-bug track (§11) changes dynamics we depend on,
  do not let it shift under trained checkpoints mid-project — adopt deliberately and
  re-run, or keep experiments on the pinned commit.
- Seed everything (Python `random`, NumPy, PyTorch); log seeds in run configs.
- Save checkpoints + the exact config per run under `checkpoints/<run-id>/`.

---

## 11. Upstream contributions track (parallel, Contributions 0.05)

Verify before PR-ing; keep on a separate branch; PR to `mariolpantunes/si2-breakout`:

- Brick collision resolves only the **first** overlapping brick per frame
  (`np.where(overlap)[0][0]`) — investigate corner/double-hit cases.
- Discrete-collision **tunneling** risk at higher ball speeds (not at 300, but note for
  robustness).
- Any clamp/edge or typo issues found while building.
- **Not** a bug: absence of `ball_vx/vy` from the wire — that is the intended partial
  observability; do **not** "fix" it.

---

## 12. Repository layout

Kept separate from the framework's `server/` and `agents/` (we only *read* `logic.py`
and subclass `BaseAgent`).

```
breakout_rl/
  env/         breakout_env.py, observation.py, rewards.py
  physics/     predictor.py
  agents/      flat_dqn.py, hierarchical_agent.py, replay.py
  controllers/ aim_controller.py
  train/       train_dqn.py, train_aim.py
  eval/        evaluate.py
  deploy/      trained_agent.py
  configs/     *.yaml
  checkpoints/
tests/         test_predictor.py, test_observation_parity.py, test_rewards_pbrs.py,
               test_smdp.py, test_aim_controller.py   (+ existing test_logic.py)
docs/superpowers/specs/   this file
```

---

## 13. Known risks (watch list)

1. **Predictor/PBRS/controller all depend on the predictor** — mitigated by making it the
   simulator itself (§4.2). This is the load-bearing decision.
2. **SMDP bookkeeping** (§5) — accumulate discounted option reward, bootstrap with `γ^k`,
   handle terminals. Highest-correctness-risk; test first.
3. **PBRS terminal handling** (§4.4) — `Φ(terminal)=0`, no bootstrap past terminal.
4. **Aiming authority cap** (§8 M3) — stochastic bounce may limit the hierarchy's edge;
   the M3 gate de-risks this before full investment.
5. **Train/deploy skew** (§9) — one observation module + a parity test.
6. **Curriculum literals** (§6) — read `W`/`width` from the game, not constants.
7. **Finite-difference velocity across a bounce** (§4.3) — known harmless transient.
8. **clone() cost** (§4.1) — profile; optimize to snapshot/restore only if needed.
```
