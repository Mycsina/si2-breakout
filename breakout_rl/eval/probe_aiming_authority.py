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
    prev_by = g.ball_y
    contacted = False
    for _ in range(max_steps):
        _apply(g, choose_action(g, region))
        g.update(DT)
        if g.lives < game.lives or g.game_over:
            break
        if g.ball_vy < 0:  # ball is ascending -> it has bounced off paddle/bricks
            contacted = True
        # next decision point: after a bounce, the ball crosses the decision line
        # downward again (entering the clean descent toward the paddle)
        if contacted and prev_by <= DECISION_LINE_Y < g.ball_y and g.ball_vy > 0:
            break
        prev_by = g.ball_y
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
        prev_by = g.ball_y
        steps = 0
        while not g.game_over and steps < max_steps:
            # decision point: ball crosses the decision line downward (clean descent)
            if prev_by <= DECISION_LINE_Y < g.ball_y and g.ball_vy > 0:
                if kind == "center":
                    region = REGION_CENTER
                elif kind == "random_region":
                    region = random.choice(REGIONS)
                elif kind == "oracle_region":
                    region = _best_region(g)
            prev_by = g.ball_y
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
