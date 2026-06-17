import argparse
import csv
import random
from pathlib import Path

import numpy as np
import torch
import yaml
from torch.utils.tensorboard import SummaryWriter

from breakout_rl.env.breakout_env import BreakoutEnv
from breakout_rl.env.observation import OBS_DIM
from breakout_rl.env.rewards import RewardConfig
from breakout_rl.agents.dqn_agent import DQNAgent
from breakout_rl.agents.replay import PrioritizedReplay, Transition
from breakout_rl.train.curriculum import curriculum_params
from breakout_rl.eval.metrics import evaluate_agent


def linear(start, end, frac):
    frac = min(max(frac, 0.0), 1.0)
    return start + (end - start) * frac


def main(cfg_path: str) -> None:
    cfg = yaml.safe_load(open(cfg_path))
    seed = cfg["seed"]
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    run_dir = Path("checkpoints") / cfg["run_id"]
    run_dir.mkdir(parents=True, exist_ok=True)
    writer = SummaryWriter(str(run_dir / "tb"))
    csv_file = open(run_dir / "log.csv", "w", newline="")
    csv_writer = csv.writer(csv_file)
    csv_writer.writerow(
        ["step", "episode_return", "episode_score", "loss", "epsilon", "eval_score"]
    )

    reward_cfg = RewardConfig(**cfg["reward"])
    env = BreakoutEnv(
        max_steps=cfg["max_episode_steps"], gamma=cfg["gamma"], reward_cfg=reward_cfg
    )
    agent = DQNAgent(OBS_DIM, n_actions=3, hidden=cfg["hidden"], lr=cfg["lr"])
    buffer = PrioritizedReplay(cfg["buffer_capacity"])

    pw, bs = curriculum_params(0, cfg["curriculum_switch_step"])
    env.set_curriculum(pw, bs)
    obs, _ = env.reset(seed=seed)
    ep_return, ep_score = 0.0, 0
    last_loss = 0.0

    for step in range(1, cfg["total_steps"] + 1):
        pw, bs = curriculum_params(step, cfg["curriculum_switch_step"])
        env.set_curriculum(pw, bs)

        eps = linear(
            cfg["epsilon_start"], cfg["epsilon_end"], step / cfg["epsilon_decay_steps"]
        )
        action = agent.select_action(obs, eps)
        next_obs, reward, term, trunc, info = env.step(action)
        buffer.add(Transition(obs, action, reward, next_obs, term, cfg["gamma"]))
        obs = next_obs
        ep_return += reward
        ep_score = info["score"]

        if len(buffer) >= cfg["learn_start"] and step % cfg["train_every"] == 0:
            beta = linear(
                cfg["per_beta_start"], cfg["per_beta_end"], step / cfg["total_steps"]
            )
            last_loss = agent.update(buffer, cfg["batch_size"], beta)

        if step % cfg["target_sync_every"] == 0:
            agent.sync_target()

        if term or trunc:
            writer.add_scalar("train/episode_return", ep_return, step)
            writer.add_scalar("train/episode_score", ep_score, step)
            writer.add_scalar("train/epsilon", eps, step)
            csv_writer.writerow([step, ep_return, ep_score, last_loss, eps, ""])
            csv_file.flush()
            obs, _ = env.reset()
            ep_return, ep_score = 0.0, 0

        if step % cfg["eval_every"] == 0:
            stats = evaluate_agent(
                agent,
                episodes=cfg["eval_episodes"],
                paddle_width=80.0,
                ball_speed=300.0,
                seed=10_000 + step,
            )
            writer.add_scalar("eval/score", stats["mean_score"], step)
            writer.add_scalar("eval/clears", stats["mean_clears"], step)
            csv_writer.writerow([step, "", "", last_loss, eps, stats["mean_score"]])
            csv_file.flush()
            print(
                f"[{step}] eval score={stats['mean_score']:.1f} clears={stats['mean_clears']:.2f}"
            )

        if step % cfg["checkpoint_every"] == 0:
            torch.save(agent.online.state_dict(), run_dir / f"online_{step}.pt")

    torch.save(agent.online.state_dict(), run_dir / "online_final.pt")
    csv_file.close()
    writer.close()


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="breakout_rl/configs/flat_dqn.yaml")
    main(p.parse_args().config)
