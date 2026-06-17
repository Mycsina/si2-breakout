import numpy as np
from breakout_rl.agents.replay import PrioritizedReplay, Transition


def _t(i):
    return Transition(
        state=np.zeros(4, np.float32),
        action=0,
        reward=float(i),
        next_state=np.zeros(4, np.float32),
        done=False,
        gamma=0.99,
    )


def test_len_and_capacity():
    buf = PrioritizedReplay(capacity=3)
    for i in range(5):
        buf.add(_t(i))
    assert len(buf) == 3  # overwrote oldest


def test_sample_shapes_and_weight_range():
    buf = PrioritizedReplay(capacity=64)
    for i in range(64):
        buf.add(_t(i))
    batch, idxs, weights = buf.sample(8, beta=0.4)
    assert len(batch) == 8 and len(idxs) == 8
    assert weights.shape == (8,)
    assert np.all(weights > 0) and np.all(weights <= 1.0 + 1e-6)


def test_priority_update_biases_sampling():
    buf = PrioritizedReplay(capacity=64)
    for i in range(64):
        buf.add(_t(i))
    # make index 0 hugely important; it should be sampled frequently
    buf.update_priorities(np.array([0]), np.array([1000.0]))
    counts = 0
    for _ in range(50):
        _, idxs, _ = buf.sample(4, beta=0.4)
        counts += int(0 in idxs.tolist())
    assert counts > 0
