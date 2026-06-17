import torch
import torch.nn as nn


class DuelingMLP(nn.Module):
    def __init__(self, in_dim: int, n_actions: int, hidden: int = 128) -> None:
        super().__init__()
        self.torso = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
        )
        self.value = nn.Sequential(
            nn.Linear(hidden, hidden), nn.ReLU(), nn.Linear(hidden, 1)
        )
        self.adv = nn.Sequential(
            nn.Linear(hidden, hidden), nn.ReLU(), nn.Linear(hidden, n_actions)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.torso(x)
        v = self.value(h)  # (B, 1)
        a = self.adv(h)  # (B, n_actions)
        return v + (a - a.mean(dim=1, keepdim=True))


class QRDuelingMLP(nn.Module):
    """Dueling network for QR-DQN: instead of one Q value per action it outputs
    ``n_quantiles`` of the return distribution per action, shape (B, n_actions, n_quantiles).
    The dueling decomposition is applied per quantile. The scalar Q used for action
    selection is the mean over quantiles (``forward(x).mean(dim=2)``)."""

    def __init__(
        self, in_dim: int, n_actions: int, n_quantiles: int, hidden: int = 128
    ) -> None:
        super().__init__()
        self.n_actions = n_actions
        self.n_quantiles = n_quantiles
        self.torso = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
        )
        self.value = nn.Sequential(
            nn.Linear(hidden, hidden), nn.ReLU(), nn.Linear(hidden, n_quantiles)
        )
        self.adv = nn.Sequential(
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, n_actions * n_quantiles),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.torso(x)
        v = self.value(h).view(-1, 1, self.n_quantiles)  # (B, 1, Nq)
        a = self.adv(h).view(-1, self.n_actions, self.n_quantiles)  # (B, A, Nq)
        return v + (a - a.mean(dim=1, keepdim=True))  # (B, A, Nq)
