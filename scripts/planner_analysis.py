"""Analysis of the model-based MC planner (supporting evidence for README §9).

(1) Sample-budget ablation: more rollouts per region sharpen the expected-bricks-broken
    estimate, with diminishing returns.
(2) RNG-correlation check: the planner snapshots/restores the global ``random`` stream so
    the live episode stays reproducible. Done naively (no independent reseed), the volley
    that actually executes draws its bounce from the SAME state the first rollout started
    from -- the planner partly *peeks* at the realized outcome instead of estimating an
    expectation. MCPlanner reseeds the rollouts to an independent sub-stream to remove this;
    SharedRNGPlanner below reproduces the naive (pre-fix) behaviour to quantify the bias.

Run: PYTHONPATH="$PWD" .venv/bin/python scripts/planner_analysis.py
"""

import random as _random

from breakout_rl.agents.planner import MCPlanner
from breakout_rl.eval.metrics import evaluate_planner

EPISODES, SEED = 20, 0


class SharedRNGPlanner(MCPlanner):
    """Naive variant: rollouts share the global RNG stream with the executed volley (no
    independent reseed). Reproduces the pre-fix behaviour for the correlation comparison."""

    def plan(self, game):
        saved = _random.getstate()
        try:
            best_region, best_val = self.REGIONS[1], float("-inf")  # default CENTER
            for region in self.REGIONS:
                val = self._expected_broken(game, region)
                if val > best_val:
                    best_val, best_region = val, region
            return best_region
        finally:
            _random.setstate(saved)


print(f"MC planner analysis ({EPISODES} episodes, seed {SEED})\n")

print("(1) sample-budget ablation (decorrelated rollouts):")
print("    n_samples | clears | bricks/life")
for n in (1, 4, 16):
    s = evaluate_planner(MCPlanner(n_samples=n), EPISODES, seed=SEED)
    print(f"    {n:9d} | {s['mean_clears']:.2f}   | {s['mean_bricks_per_life']:.2f}")

print("\n(2) RNG-correlation check (shared vs decorrelated rollout stream):")
print("    n_samples | shared (naive) | decorrelated (honest)")
for n in (1, 16):
    sh = evaluate_planner(SharedRNGPlanner(n_samples=n), EPISODES, seed=SEED)
    de = evaluate_planner(MCPlanner(n_samples=n), EPISODES, seed=SEED)
    print(f"    {n:9d} | {sh['mean_clears']:.2f}           | {de['mean_clears']:.2f}")
