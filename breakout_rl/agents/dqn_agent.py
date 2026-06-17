import random
import numpy as np
import torch
import torch.nn as nn

from breakout_rl.agents.networks import DuelingMLP
from breakout_rl.agents.replay import PrioritizedReplay


class DQNAgent:
    def __init__(
        self,
        obs_dim: int,
        n_actions: int,
        hidden: int = 128,
        lr: float = 1e-3,
        device: str = "cuda",
    ) -> None:
        self.n_actions = n_actions
        self.device = torch.device(
            device if torch.cuda.is_available() or device == "cpu" else "cpu"
        )
        self.online = DuelingMLP(obs_dim, n_actions, hidden).to(self.device)
        self.target = DuelingMLP(obs_dim, n_actions, hidden).to(self.device)
        self.sync_target()
        self.opt = torch.optim.Adam(self.online.parameters(), lr=lr)

    def sync_target(self) -> None:
        self.target.load_state_dict(self.online.state_dict())

    @torch.no_grad()
    def select_action(self, obs: np.ndarray, epsilon: float) -> int:
        if random.random() < epsilon:
            return random.randrange(self.n_actions)
        x = torch.as_tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)
        return int(self.online(x).argmax(dim=1).item())

    def update(
        self,
        buffer: PrioritizedReplay,
        batch_size: int,
        beta: float,
        grad_clip: float = 10.0,
    ) -> float:
        batch, idxs, weights = buffer.sample(batch_size, beta)
        s = torch.as_tensor(np.stack([t.state for t in batch]), device=self.device)
        a = torch.as_tensor([t.action for t in batch], device=self.device).long()
        r = torch.as_tensor(
            [t.reward for t in batch], dtype=torch.float32, device=self.device
        )
        ns = torch.as_tensor(
            np.stack([t.next_state for t in batch]), device=self.device
        )
        done = torch.as_tensor(
            [t.done for t in batch], dtype=torch.float32, device=self.device
        )
        gamma = torch.as_tensor(
            [t.gamma for t in batch], dtype=torch.float32, device=self.device
        )
        w = torch.as_tensor(weights, device=self.device)

        q = self.online(s).gather(1, a.unsqueeze(1)).squeeze(1)
        with torch.no_grad():
            next_a = self.online(ns).argmax(dim=1, keepdim=True)  # Double DQN
            next_q = self.target(ns).gather(1, next_a).squeeze(1)
            target = r + gamma * next_q * (1.0 - done)
        td = q - target
        loss = (w * td.pow(2)).mean()

        self.opt.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.online.parameters(), grad_clip)
        self.opt.step()

        buffer.update_priorities(idxs, td.detach().cpu().numpy())
        return float(loss.item())
