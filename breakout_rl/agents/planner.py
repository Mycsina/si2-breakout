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

    The server bounce draws from the global ``random`` stream. ``plan`` therefore (a)
    snapshots and restores that stream so the real episode's bounces are left bit-identical
    (seeded evaluation stays reproducible), and (b) reseeds the stream to an INDEPENDENT
    per-decision sub-stream *for the rollouts only*. Point (b) matters: without it, the
    volley that actually executes after ``plan`` returns draws its bounce from the very same
    RNG state the first rollout started from, so the planner would be partly *peeking* at the
    realized outcome rather than estimating a true expectation. Decorrelating the rollout
    stream removes that bias (measured: ~+0.15 clears at n=16, ~+0.45 at n=1 of inflation)."""

    REGIONS = (REGION_LEFT, REGION_CENTER, REGION_RIGHT)

    def __init__(
        self, n_samples: int = 16, rollout_env=None, rollout_seed: int = 987654321
    ) -> None:
        self.n_samples = n_samples
        self._rollout_seed = rollout_seed
        self._calls = 0
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
        saved = (
            _random.getstate()
        )  # the real episode's RNG stream; restored before return
        try:
            # rollouts draw from an independent per-decision stream (see class docstring),
            # decorrelated from the bounce the executed volley will draw from `saved`.
            self._calls += 1
            _random.seed(self._rollout_seed + self._calls)
            best_region, best_val = REGION_CENTER, float("-inf")
            for region in self.REGIONS:
                val = self._expected_broken(game, region)
                if val > best_val:
                    best_val, best_region = val, region
            return best_region
        finally:
            _random.setstate(saved)
