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
