import random as _random

from server.logic import Breakout
from breakout_rl.physics.clone import clone_game
from breakout_rl.constants import REGION_LEFT, REGION_CENTER, REGION_RIGHT


class MCPlanner:
    """Model-based one-volley planner (a learning-free baseline for the high level).

    At each decision point it Monte-Carlo-evaluates every contact region: it clones the
    live game and rolls each region forward to the *next* decision point, letting the
    server's stochastic paddle bounce supply the randomness, then picks the region with
    the highest expected bricks broken. The rollout reuses ``HighLevelEnv``'s own
    ``_advance_to_decision_point`` so the planner's model of the dynamics is identical to
    training/eval by construction (same controller, same brick-counting, same option
    boundary) -- there is no separately-maintained simulator to drift.

    Horizon is a single volley (greedy on immediate expected bricks). This is deliberately
    myopic: it is the natural planning baseline to benchmark the long-horizon learned
    policy against, not a replacement for it.

    The server bounce draws from the global ``random`` stream, so ``plan`` snapshots and
    restores that stream: the rollouts consume randomness only transiently and the real
    episode's bounces are left bit-identical (keeping seeded evaluation reproducible)."""

    REGIONS = (REGION_LEFT, REGION_CENTER, REGION_RIGHT)

    def __init__(self, n_samples: int = 16, rollout_env=None) -> None:
        self.n_samples = n_samples
        if rollout_env is None:
            # local import: high_level_env imports controllers, not the planner -- importing
            # it at module top would be fine, but keeping it lazy avoids any future cycle.
            from breakout_rl.env.high_level_env import HighLevelEnv

            rollout_env = HighLevelEnv()
        self._env = rollout_env

    def _expected_broken(self, game: Breakout, region: int) -> float:
        total = 0
        for _ in range(self.n_samples):
            self._env.game = clone_game(game)
            _, _, _, _, broken = self._env._advance_to_decision_point(region)
            total += broken
        return total / self.n_samples

    def plan(self, game: Breakout) -> int:
        """Return the region with the highest expected bricks broken over the next volley."""
        saved = _random.getstate()  # keep the real episode's RNG stream untouched
        try:
            best_region, best_val = REGION_CENTER, float("-inf")
            for region in self.REGIONS:
                val = self._expected_broken(game, region)
                if val > best_val:
                    best_val, best_region = val, region
            return best_region
        finally:
            _random.setstate(saved)
