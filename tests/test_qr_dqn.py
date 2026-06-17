import numpy as np
import torch

from breakout_rl.agents.networks import QRDuelingMLP
from breakout_rl.agents.qr_dqn_agent import QRDQNAgent, quantile_huber_loss
from breakout_rl.agents.replay import PrioritizedReplay, Transition


def test_qr_network_outputs_quantiles_per_action():
    net = QRDuelingMLP(in_dim=5, n_actions=3, n_quantiles=8)
    out = net(torch.zeros(4, 5))
    assert out.shape == (4, 3, 8)  # (B, A, Nq)
    assert out.mean(dim=2).shape == (4, 3)  # scalar Q per action for greedy selection


def test_quantile_huber_loss_is_zero_for_a_matched_degenerate_distribution():
    # the QR loss is pairwise (every predicted quantile vs every target sample), so it
    # vanishes only when both distributions collapse to the same single value.
    theta = torch.full((6, 16), 2.5)
    target = torch.full((6, 16), 2.5)
    loss = quantile_huber_loss(theta, target)
    assert loss.shape == (6,)
    assert torch.allclose(loss, torch.zeros(6), atol=1e-6)


def test_quantile_huber_loss_is_positive_for_a_mismatch():
    theta = torch.zeros(3, 8)
    target = torch.ones(3, 8)  # target strictly above prediction
    loss = quantile_huber_loss(theta, target)
    assert loss.shape == (3,)
    assert (loss > 0).all()


def test_qr_agent_acts_and_learns():
    agent = QRDQNAgent(obs_dim=5, n_actions=3, n_quantiles=8, device="cpu")
    a = agent.select_action(np.zeros(5, dtype=np.float32), epsilon=0.0)
    assert a in (0, 1, 2)

    buf = PrioritizedReplay(capacity=100)
    for _ in range(16):
        buf.add(
            Transition(
                state=np.random.randn(5).astype(np.float32),
                action=np.random.randint(3),
                reward=float(np.random.randn()),
                next_state=np.random.randn(5).astype(np.float32),
                done=False,
                gamma=0.99,
            )
        )
    loss = agent.update(buf, batch_size=8, beta=0.4)
    assert isinstance(loss, float) and np.isfinite(loss)
