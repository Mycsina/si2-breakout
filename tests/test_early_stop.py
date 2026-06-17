from breakout_rl.train.early_stop import EarlyStopper


def test_improvement_marks_best_and_resets_patience():
    s = EarlyStopper(patience=3, min_delta=0.02, warmup_options=0)
    is_best, stop = s.update(1.0, opt=100)
    assert is_best and not stop and s.best == 1.0
    # a clear improvement is a new best and resets the counter
    s.update(1.0, opt=200)  # no improvement
    is_best, stop = s.update(1.5, opt=300)
    assert is_best and not stop and s.evals_since_best == 0


def test_stops_after_patience_without_improvement():
    s = EarlyStopper(patience=3, min_delta=0.02, warmup_options=0)
    s.update(2.0, opt=100)  # best
    assert s.update(2.0, opt=200) == (False, False)  # 1 stale
    assert s.update(1.9, opt=300) == (False, False)  # 2 stale
    assert s.update(2.01, opt=400) == (
        False,
        True,
    )  # 3 stale (< min_delta over best) -> stop


def test_warmup_blocks_early_stop():
    s = EarlyStopper(patience=2, min_delta=0.02, warmup_options=1000)
    s.update(1.0, opt=100)  # best
    s.update(1.0, opt=200)  # 1 stale
    _, stop = s.update(1.0, opt=300)  # 2 stale but before warmup
    assert not stop
    # same staleness, now past warmup -> stop
    _, stop = s.update(1.0, opt=1000)
    assert stop


def test_sub_min_delta_gain_does_not_reset():
    s = EarlyStopper(patience=2, min_delta=0.05, warmup_options=0)
    s.update(3.0, opt=100)  # best = 3.0
    # +0.04 is below min_delta -> not a new best, counts as stale
    is_best, _ = s.update(3.04, opt=200)
    assert not is_best and s.evals_since_best == 1
