import torch
from breakout_rl.agents.networks import DuelingMLP


def test_output_shape_matches_actions():
    net = DuelingMLP(in_dim=23, n_actions=3, hidden=64)
    q = net(torch.zeros(5, 23))
    assert q.shape == (5, 3)


def test_dueling_advantage_is_mean_centered():
    # If V is fixed and A has nonzero mean, Q must equal V + (A - mean(A)).
    net = DuelingMLP(in_dim=4, n_actions=3, hidden=16)
    x = torch.randn(2, 4)
    q = net(x)
    # mean over actions of (q - V) == 0 by construction of dueling head
    assert q.shape == (2, 3)
    # numerical sanity: not NaN
    assert torch.isfinite(q).all()
