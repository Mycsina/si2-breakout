import numpy as np
from breakout_rl.agents.dqn_agent import DQNAgent
from breakout_rl.agents.replay import PrioritizedReplay, Transition


def test_select_action_in_range_and_greedy_is_deterministic():
    agent = DQNAgent(obs_dim=23, n_actions=3, device="cpu")
    obs = np.zeros(23, np.float32)
    a = agent.select_action(obs, epsilon=0.0)
    assert a in (0, 1, 2)
    assert agent.select_action(obs, epsilon=0.0) == a  # greedy deterministic


def test_update_runs_and_returns_finite_loss():
    agent = DQNAgent(obs_dim=4, n_actions=3, device="cpu")
    buf = PrioritizedReplay(capacity=128)
    for i in range(128):
        buf.add(Transition(
            state=np.random.randn(4).astype(np.float32), action=i % 3,
            reward=1.0, next_state=np.random.randn(4).astype(np.float32),
            done=False, gamma=0.99,
        ))
    loss = agent.update(buf, batch_size=32, beta=0.4)
    assert np.isfinite(loss)


def test_target_sync_copies_weights():
    agent = DQNAgent(obs_dim=4, n_actions=3, device="cpu")
    # perturb online net, sync, assert equal
    for p in agent.online.parameters():
        p.data.add_(1.0)
    agent.sync_target()
    for po, pt in zip(agent.online.parameters(), agent.target.parameters()):
        assert (po.data == pt.data).all()
