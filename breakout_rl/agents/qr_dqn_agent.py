import random
import numpy as np
import torch
import torch.nn as nn

from breakout_rl.agents.networks import QRDuelingMLP
from breakout_rl.agents.replay import PrioritizedReplay


def quantile_huber_loss(
    theta: torch.Tensor, target: torch.Tensor, kappa: float = 1.0
) -> torch.Tensor:
    """Per-sample QR-DQN loss.

    ``theta``  (B, Nq): the predicted quantiles for the taken action, at fractions
    tau_i = (i + 0.5)/Nq.  ``target`` (B, Nq): the (detached) target quantile samples.
    Returns a (B,) tensor: for each sample, sum over predicted quantiles i of the
    tau-weighted Huber loss averaged over target samples j (the standard QR-DQN form)."""
    nq = theta.shape[1]
    # pairwise temporal-difference: u_ij = target_j - theta_i  -> (B, Nq_i, Nq_j)
    u = target.unsqueeze(1) - theta.unsqueeze(2)
    abs_u = u.abs()
    huber = torch.where(abs_u <= kappa, 0.5 * u.pow(2), kappa * (abs_u - 0.5 * kappa))
    tau = (
        (torch.arange(nq, device=theta.device, dtype=torch.float32) + 0.5) / nq
    ).view(1, nq, 1)  # (1, Nq_i, 1)
    weight = (tau - (u.detach() < 0).float()).abs()
    loss = (weight * huber).mean(dim=2).sum(dim=1)  # mean over j, sum over i -> (B,)
    return loss


class QRDQNAgent:
    """Distributional (quantile-regression) DQN for the high-level region policy.

    Same interface as DQNAgent (select_action / update / sync_target / .online) so it is a
    drop-in for the SMDP trainer. Models the return *distribution* induced by the random
    paddle bounce as ``n_quantiles`` quantiles per action; acts greedily on the mean
    quantile. Keeps Double-DQN next-action selection and the per-transition SMDP discount
    (gamma**k) in the distributional Bellman target, and PER (priority = sample loss)."""

    def __init__(
        self,
        obs_dim: int,
        n_actions: int,
        n_quantiles: int = 51,
        hidden: int = 128,
        lr: float = 1e-3,
        device: str = "cuda",
    ) -> None:
        self.n_actions = n_actions
        self.n_quantiles = n_quantiles
        self.device = torch.device(
            device if torch.cuda.is_available() or device == "cpu" else "cpu"
        )
        self.online = QRDuelingMLP(obs_dim, n_actions, n_quantiles, hidden).to(
            self.device
        )
        self.target = QRDuelingMLP(obs_dim, n_actions, n_quantiles, hidden).to(
            self.device
        )
        self.sync_target()
        self.opt = torch.optim.Adam(self.online.parameters(), lr=lr)

    def sync_target(self) -> None:
        self.target.load_state_dict(self.online.state_dict())

    @torch.no_grad()
    def select_action(self, obs: np.ndarray, epsilon: float) -> int:
        if random.random() < epsilon:
            return random.randrange(self.n_actions)
        x = torch.as_tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)
        q = self.online(x).mean(dim=2)  # (1, A) mean over quantiles
        return int(q.argmax(dim=1).item())

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

        nq = self.n_quantiles
        a_idx = a.view(-1, 1, 1).expand(-1, 1, nq)
        theta = self.online(s).gather(1, a_idx).squeeze(1)  # (B, Nq)
        with torch.no_grad():
            next_a = self.online(ns).mean(dim=2).argmax(dim=1)  # Double DQN
            next_idx = next_a.view(-1, 1, 1).expand(-1, 1, nq)
            theta_next = self.target(ns).gather(1, next_idx).squeeze(1)  # (B, Nq)
            target = r.unsqueeze(1) + (gamma * (1.0 - done)).unsqueeze(1) * theta_next
        per_sample = quantile_huber_loss(theta, target)  # (B,)
        loss = (w * per_sample).mean()

        self.opt.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.online.parameters(), grad_clip)
        self.opt.step()

        buffer.update_priorities(idxs, per_sample.detach().cpu().numpy())
        return float(loss.item())
