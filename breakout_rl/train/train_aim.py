import argparse
import copy
import csv
import random
from pathlib import Path
import numpy as np
import torch
import yaml
from torch.utils.tensorboard import SummaryWriter

from breakout_rl.env.high_level_env import HighLevelEnv, HIGH_OBS_DIM
from breakout_rl.env.rewards import RewardConfig
from breakout_rl.agents.dqn_agent import DQNAgent
from breakout_rl.agents.qr_dqn_agent import QRDQNAgent
from breakout_rl.agents.replay import PrioritizedReplay, Transition
from breakout_rl.train.curriculum import curriculum_params
from breakout_rl.train.early_stop import EarlyStopper
from breakout_rl.eval.metrics import evaluate_hierarchical


def linear(a, b, frac):
    frac = min(max(frac, 0.0), 1.0)
    return a + (b - a) * frac


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
    cw = csv.writer(csv_file)
    cw.writerow(["option", "option_return", "score", "loss", "epsilon", "eval_clears"])

    env = HighLevelEnv(
        max_option_steps=cfg["max_option_steps"],
        gamma=cfg["gamma"],
        reward_cfg=RewardConfig(**cfg["reward"]),
    )
    if cfg.get("algo", "dqn") == "qr_dqn":
        agent = QRDQNAgent(
            HIGH_OBS_DIM,
            n_actions=3,
            n_quantiles=cfg.get("n_quantiles", 51),
            hidden=cfg["hidden"],
            lr=cfg["lr"],
        )
    else:
        agent = DQNAgent(HIGH_OBS_DIM, n_actions=3, hidden=cfg["hidden"], lr=cfg["lr"])
    buffer = PrioritizedReplay(cfg["buffer_capacity"])

    pw, bs = curriculum_params(0, cfg["curriculum_switch_option"])
    env.set_curriculum(pw, bs)
    obs, _ = env.reset(seed=seed)
    last_loss = 0.0
    ep_steps = 0

    stopper = EarlyStopper(
        patience=cfg.get("early_stop_patience", 10),
        min_delta=cfg.get("early_stop_min_delta", 0.02),
        warmup_options=cfg.get("early_stop_warmup_options", 40000),
    )
    best_state = None
    stopped_at = cfg["total_options"]

    for opt in range(1, cfg["total_options"] + 1):
        pw, bs = curriculum_params(opt, cfg["curriculum_switch_option"])
        env.set_curriculum(pw, bs)
        eps = linear(
            cfg["epsilon_start"], cfg["epsilon_end"], opt / cfg["epsilon_decay_options"]
        )
        region = agent.select_action(obs, eps)
        next_obs, R, term, trunc, info = env.step(region)
        ep_steps += info["k"]
        # SMDP: store gamma**k as the per-transition discount
        buffer.add(Transition(obs, region, R, next_obs, term, info["gamma_k"]))
        obs = next_obs

        if len(buffer) >= cfg["learn_start"] and opt % cfg["train_every"] == 0:
            beta = linear(
                cfg["per_beta_start"], cfg["per_beta_end"], opt / cfg["total_options"]
            )
            last_loss = agent.update(buffer, cfg["batch_size"], beta)
        if opt % cfg["target_sync_every"] == 0:
            agent.sync_target()
        # The aim controller almost never drops the ball, so games don't end on their own.
        # Cap each episode at max_episode_steps *primitive* steps so episodes are finite and,
        # crucially, the curriculum (applied only on reset) actually switches to stage 2 past
        # curriculum_switch_option instead of the env staying stuck on the stage-1 settings.
        if term or trunc or ep_steps >= cfg["max_episode_steps"]:
            cw.writerow([opt, R, info["score"], last_loss, eps, ""])
            csv_file.flush()
            obs, _ = env.reset()
            ep_steps = 0
        if opt % cfg["eval_every"] == 0:
            stats = evaluate_hierarchical(
                agent,
                episodes=cfg["eval_episodes"],
                paddle_width=80.0,
                ball_speed=300.0,
                seed=10_000 + opt,
            )
            writer.add_scalar("eval/clears", stats["mean_clears"], opt)
            writer.add_scalar(
                "eval/bricks_per_life", stats["mean_bricks_per_life"], opt
            )
            cw.writerow([opt, "", "", last_loss, eps, stats["mean_clears"]])
            csv_file.flush()
            # Convergence metric: clears (coarse) + bricks/life as a fine-grained
            # tie-breaker, so noisy equal-clear evals don't all look identical.
            metric = stats["mean_clears"] + 0.01 * stats["mean_bricks_per_life"]
            is_best, should_stop = stopper.update(metric, opt)
            if is_best:
                best_state = copy.deepcopy(agent.online.state_dict())
                torch.save(
                    best_state, run_dir / "online_best.pt"
                )  # gitignored crash net
            print(
                f"[{opt}] eval clears={stats['mean_clears']:.2f} "
                f"bpl={stats['mean_bricks_per_life']:.2f} "
                f"metric={metric:.3f} best={stopper.best:.3f} "
                f"stale={stopper.evals_since_best}/{stopper.patience}"
            )
            if should_stop:
                print(
                    f"[{opt}] EARLY STOP: metric plateaued "
                    f"(best={stopper.best:.3f}, no gain > {stopper.min_delta} "
                    f"for {stopper.patience} evals)"
                )
                stopped_at = opt
                break
        if opt % cfg["checkpoint_every"] == 0:
            torch.save(agent.online.state_dict(), run_dir / f"online_{opt}.pt")

    # Keep the BEST-eval policy as the final checkpoint (RL eval is noisy; the last
    # policy is not always the best). Fall back to current weights if no eval ran.
    final_state = best_state if best_state is not None else agent.online.state_dict()
    torch.save(final_state, run_dir / "online_final.pt")
    print(
        f"done: stopped at option {stopped_at}/{cfg['total_options']}, "
        f"best metric={stopper.best:.3f} -> online_final.pt"
    )
    csv_file.close()
    writer.close()


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="breakout_rl/configs/aim.yaml")
    main(p.parse_args().config)
