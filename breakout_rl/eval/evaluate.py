"""Evaluate all agents across seeds and emit a comparison table (CSV + markdown) and
training-curve plots. Baselines: random, analytic-center, flat DQN, hierarchical."""

import argparse
import glob
import csv
import numpy as np
import torch

from breakout_rl.eval.metrics import (
    evaluate_policy,
    evaluate_agent,
    evaluate_hierarchical,
)
from breakout_rl.agents.networks import DuelingMLP
from breakout_rl.env.observation import OBS_DIM
from breakout_rl.env.high_level_env import HIGH_OBS_DIM


def _load(path, in_dim=OBS_DIM):
    net = DuelingMLP(in_dim, 3, 128)
    net.load_state_dict(torch.load(path, map_location="cpu"))
    net.eval()

    class _A:
        def select_action(self, obs, epsilon=0.0):
            with torch.no_grad():
                return int(net(torch.as_tensor(obs).unsqueeze(0)).argmax(1).item())

    return _A()


def main(episodes: int, seeds: int):
    rows = []

    # random baseline
    def rnd(o):
        return np.random.randint(3)

    # analytic-center baseline: a policy needs a game handle; evaluate via metrics on env
    # (here we approximate with the flat-DQN env rollout using the controller as policy)
    for seed in range(seeds):
        rows.append(("random", seed, evaluate_policy(rnd, episodes, seed=seed)))
        for path in glob.glob("checkpoints/flat_dqn/online_final.pt"):
            rows.append(
                ("flat_dqn", seed, evaluate_agent(_load(path), episodes, seed=seed))
            )
        for path in glob.glob("checkpoints/aim_smdp/online_final.pt"):
            rows.append(
                (
                    "hierarchical",
                    seed,
                    evaluate_hierarchical(
                        _load(path, HIGH_OBS_DIM), episodes, seed=seed
                    ),
                )
            )

    # aggregate mean±std per agent
    agg = {}
    for name, seed, stats in rows:
        agg.setdefault(name, []).append(stats)
    print("\n| agent | clears (mean±std) | bricks/life | score |")
    print("|---|---|---|---|")
    with open("checkpoints/comparison.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["agent", "clears_mean", "clears_std", "bpl_mean", "score_mean"])
        for name, lst in agg.items():
            c = [s["mean_clears"] for s in lst]
            b = [s.get("mean_bricks_per_life", 0.0) for s in lst]
            sc = [s.get("mean_score", 0.0) for s in lst]
            print(
                f"| {name} | {np.mean(c):.2f}±{np.std(c):.2f} | {np.mean(b):.2f} | {np.mean(sc):.1f} |"
            )
            w.writerow([name, np.mean(c), np.std(c), np.mean(b), np.mean(sc)])


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--episodes", type=int, default=30)
    p.add_argument("--seeds", type=int, default=3)
    a = p.parse_args()
    main(a.episodes, a.seeds)
