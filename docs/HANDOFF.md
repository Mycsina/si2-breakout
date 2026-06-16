# Get-up-to-speed handoff

This is a quick operational brief on the RL agent work: current state, what was
decided/changed and why, what's left, and the exact commands to finish. The full
methodology lives in the README "Reinforcement Learning Agent — Project Report" section.

## TL;DR status

- **All 28 plan tasks' code is implemented, tested, and committed.** `47/47` tests pass.
- The whole training/eval/deploy pipeline is **validated end-to-end** (a tiny 2.5k-step
  smoke run trains → evaluates → checkpoints with no errors).
- The **M3 aiming-authority gate experiment has been run** (results below).
- **What remains is the heavy GPU training + pasting its numbers into the report**, plus
  an optional upstream PR. None of that could run in this session — see "Environment".

## Environment / gotchas

- The repo uses **`uv`** (not `venv`+`pip`). Setup is `uv sync`; run things with
  `.venv/bin/python ...`. Python is 3.14, `torch 2.12.0`.
- **This session had no CUDA** (`torch.cuda.is_available() == False`), so training was
  only validated for *correctness* on CPU, not run to convergence. Run the full training
  on your GPU box.
- The repo is **Jujutsu (jj) colocated with git**. I commit with `jj`. Note: `git status`
  is unreliable here (jj rewrites the index) — use `jj status` / `jj log`.
- `predict_landing` is called each descending step. It was fast enough (smoke ran in 6 s),
  so the optional descent-cache (plan Task 14) was **not** needed. Add it only if a full
  run turns out to be throughput-bound.

## Decisions & deviations I made (and why)

1. **`uv` instead of `venv`/pip** — the repo was already uv-managed; following the plan's
   `python -m venv venv` verbatim would have created a conflicting environment.
2. **`.gitignore` bug fix** — the stock Python `.gitignore` had an unanchored `env/` rule
   that silently *ignored the entire `breakout_rl/env/` source package* (observation,
   rewards, both Gym envs). Anchored the virtualenv patterns to the repo root (`/env/`).
   Without this, half the code would never have been committed.
3. **Fixed a flawed env test** (`test_breakout_env.py`) — the plan's parity assertion
   re-fed the same frame into the *stateful* `ObservationBuilder`, which zeroes velocity;
   it can never match a fresh builder. Rewrote it to assert the real intent (the obs the
   env returns is reproducible by an independent builder replaying the same wire frames).
4. **Corrected the decision-point detector (the important one).** The plan used a velocity
   sign-flip (`prev_vy ≤ 0 and vy > 0 and y > 140`) to mark a "decision point". In this
   gravity-free game the velocity only flips at collisions — all of which happen above
   `y = 140` — so **that condition never fired**. Effect: the high-level region was never
   re-chosen, so the M3 probe's three policies were identical and the SMDP options would
   span whole lives. Replaced with a **downward crossing of the decision line**
   (`prev_ball_y ≤ 140 < ball_y`, descending) in both `probe_aiming_authority.py` and
   `high_level_env.py`. After the fix, option lengths are k ≈ 60–80 and the probe
   discriminates. **If you change `server/logic.py` physics, re-pin and re-run.**
5. **`scripts/train_seeds.sh` uses `.venv`** directly (not `source venv/bin/activate`).
6. **Commits are batched by phase**, per your "don't spam commits" preference.

## Key findings

- **M3 gate (100 episodes):** center 5.0 bricks/life · random-region 15.8 · oracle-region
  14.6. Aiming any non-center region is **~3× / +191%** over center → the hierarchy is
  justified. But **oracle ≈ random** (the stochastic bounce caps myopic aiming), so the
  learned policy must win via board-aware, multi-step credit assignment, not greedy
  per-volley choice. Raw numbers: `checkpoints/m3_probe_results.txt`.
- **Upstream bug confirmed:** the ball can pass through a second simultaneously-overlapped
  brick without breaking it (see README §11). Fix is a candidate PR — *not yet opened*
  (it's outward-facing; tell me to proceed and I'll branch, fix, add a regression test,
  and push).

## What's left (your GPU + a couple of decisions)

```bash
# 1. Full reactive flat-DQN training (~hours on GPU)
.venv/bin/python -m breakout_rl.train.train_dqn --config breakout_rl/configs/flat_dqn.yaml

# 2. Hierarchical SMDP aim training
.venv/bin/python -m breakout_rl.train.train_aim  --config breakout_rl/configs/aim.yaml

# 3. Multi-seed runs + comparison table + curves (fills README §9)
bash scripts/train_seeds.sh
.venv/bin/python -m breakout_rl.eval.evaluate --episodes 30 --seeds 3
.venv/bin/python -m breakout_rl.eval.plot_curves --pattern "checkpoints/flat_dqn*/log.csv" --out docs/curves_flat_dqn.png

# 4. Deploy visual check (paddle tracks ball, score climbs)
.venv/bin/python -m server.server                 # terminal 1
.venv/bin/python -m breakout_rl.deploy.trained_agent --mode flat --checkpoint checkpoints/flat_dqn/online_final.pt
# open http://localhost:8765/  — repeat with --mode hier --checkpoint checkpoints/aim_smdp/online_final.pt
```

Then paste the curve PNG, the comparison table, and a deploy note into **README §9**.

Decisions for you: (a) open the upstream brick-collision PR? (b) keep both agents in the
report regardless of how the learned hierarchy compares to random-region (recommended —
the comparison itself is the interesting result)?
