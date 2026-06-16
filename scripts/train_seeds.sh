#!/usr/bin/env bash
# Train the flat DQN across 3 seeds for the multi-seed comparison (run on a GPU box).
# Uses the project's uv-managed virtualenv (.venv) directly — no activation needed.
set -euo pipefail
PY=.venv/bin/python
for s in 0 1 2; do
  "$PY" - <<PY
import yaml
c = yaml.safe_load(open("breakout_rl/configs/flat_dqn.yaml"))
c["seed"] = $s; c["run_id"] = f"flat_dqn_seed$s"
yaml.safe_dump(c, open("breakout_rl/configs/_flat_seed$s.yaml", "w"))
PY
  "$PY" -m breakout_rl.train.train_dqn --config breakout_rl/configs/_flat_seed$s.yaml
done
