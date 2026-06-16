from dataclasses import dataclass
from typing import List, Tuple
import numpy as np


@dataclass
class Transition:
    state: np.ndarray
    action: int
    reward: float
    next_state: np.ndarray
    done: bool
    gamma: float  # per-sample discount (gamma for flat, gamma**k for SMDP options)


class PrioritizedReplay:
    """Proportional prioritized experience replay (simple array-backed implementation).
    Priority p_i = (|td_error| + eps) ** alpha; sampling prob = p_i / sum(p)."""

    def __init__(self, capacity: int, alpha: float = 0.6, eps: float = 1e-5) -> None:
        self.capacity = capacity
        self.alpha = alpha
        self.eps = eps
        self.data: List[Transition] = []
        self.priorities = np.zeros(capacity, dtype=np.float64)
        self.pos = 0

    def __len__(self) -> int:
        return len(self.data)

    def add(self, t: Transition) -> None:
        max_p = self.priorities.max() if self.data else 1.0
        if len(self.data) < self.capacity:
            self.data.append(t)
        else:
            self.data[self.pos] = t
        self.priorities[self.pos] = max_p  # new samples get max priority
        self.pos = (self.pos + 1) % self.capacity

    def sample(self, batch_size: int, beta: float) -> Tuple[List[Transition], np.ndarray, np.ndarray]:
        n = len(self.data)
        prios = self.priorities[:n] ** self.alpha
        probs = prios / prios.sum()
        idxs = np.random.choice(n, batch_size, p=probs)
        batch = [self.data[i] for i in idxs]
        weights = (n * probs[idxs]) ** (-beta)
        weights = weights / weights.max()
        return batch, idxs, weights.astype(np.float32)

    def update_priorities(self, idxs: np.ndarray, td_errors: np.ndarray) -> None:
        for i, e in zip(idxs, td_errors):
            self.priorities[int(i)] = abs(float(e)) + self.eps
