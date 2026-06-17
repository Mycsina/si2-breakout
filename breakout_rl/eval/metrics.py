from typing import Callable, Dict
import numpy as np

from breakout_rl.env.breakout_env import BreakoutEnv


def evaluate_policy(
    policy: Callable[[np.ndarray], int],
    episodes: int,
    max_steps: int = 4000,
    paddle_width: float = 80.0,
    ball_speed: float = 300.0,
    seed: int = 0,
) -> Dict[str, float]:
    """Roll out a policy (obs -> action int) for several full games and aggregate
    clearing-focused metrics (see spec §7). Survival/score are expected to saturate;
    clears / bricks-per-life are the discriminating metrics."""
    env = BreakoutEnv(
        max_steps=max_steps, paddle_width=paddle_width, ball_speed=ball_speed
    )
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
                bricks_broken += prev_bricks - now
            if now == 0 or (
                prev_bricks <= 2 and now > prev_bricks
            ):  # board cleared+respawned
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
    return evaluate_policy(
        lambda o: agent.select_action(o, epsilon=0.0), episodes=episodes, **kw
    )


def _evaluate_high_level(
    region_fn: Callable,
    episodes: int,
    max_steps: int = 4000,
    max_option_steps: int = 3000,
    paddle_width: float = 80.0,
    ball_speed: float = 300.0,
    seed: int = 0,
):
    """Shared driver for any high-level region policy in the HighLevelEnv, aggregating
    clearing metrics across full games (each game = many options). ``region_fn(env, obs)``
    returns the region for the current decision point -- a learned agent ignores ``env``
    and reads ``obs``; the model-based planner ignores ``obs`` and reads ``env.game``.

    The hard-coded aim controller intercepts the ball on almost every volley, so a game
    essentially never ends on its own (lives are practically never lost) and each option
    ends at a decision point long before max_option_steps. A `while term or trunc` loop
    would therefore never terminate. We cap each episode at `max_steps` *primitive* steps
    (cumulative info["k"]) -- the same horizon used by the flat evaluate_policy -- so all
    policies are scored over an equal step budget."""
    from breakout_rl.env.high_level_env import HighLevelEnv

    env = HighLevelEnv(
        max_option_steps=max_option_steps,
        paddle_width=paddle_width,
        ball_speed=ball_speed,
    )
    clears, bricks_total, scores = [], [], []
    for ep in range(episodes):
        obs, _ = env.reset(seed=seed + ep)
        broken = 0
        prev = int(env.game.brick_array[:, 4].sum())
        prim_steps = 0
        while True:
            region = region_fn(env, obs)
            obs, R, term, trunc, info = env.step(region)
            prim_steps += info["k"]
            now = info["bricks_left"]
            if now < prev:
                broken += prev - now
            prev = now
            if term or trunc or prim_steps >= max_steps:
                break
        clears.append(broken // 16)
        bricks_total.append(broken / 3.0)
        scores.append(
            env.game.score
        )  # same game-score metric the flat evaluate_policy reports
    return {
        "mean_clears": float(np.mean(clears)),
        "mean_bricks_per_life": float(np.mean(bricks_total)),
        "mean_score": float(np.mean(scores)),
        "std_score": float(np.std(scores)),
    }


def evaluate_hierarchical(agent, episodes: int, **kw):
    """Evaluate a learned high-level region policy (greedy)."""
    return _evaluate_high_level(
        lambda env, obs: agent.select_action(obs, epsilon=0.0), episodes, **kw
    )


def evaluate_planner(planner, episodes: int, **kw):
    """Evaluate the model-based MC planner, which reads the live game state each step."""
    return _evaluate_high_level(lambda env, obs: planner.plan(env.game), episodes, **kw)
