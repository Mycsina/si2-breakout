"""Plateau-based early stopping for the SMDP trainer.

RL eval is noisy and the *final* policy is not always the best, so we track the best eval
metric seen so far, keep the checkpoint that produced it, and stop once the metric has not
improved by more than ``min_delta`` for ``patience`` consecutive evaluations. A ``warmup``
floor prevents stopping during the early high-exploration / pre-curriculum-switch phase.
"""


class EarlyStopper:
    def __init__(self, patience: int, min_delta: float, warmup_options: int):
        self.patience = patience
        self.min_delta = min_delta
        self.warmup_options = warmup_options
        self.best = float("-inf")
        self.evals_since_best = 0

    def update(self, metric: float, opt: int) -> tuple[bool, bool]:
        """Feed one evaluation. Returns ``(is_new_best, should_stop)``.

        ``is_new_best`` marks an improvement of more than ``min_delta`` over the running
        best (the caller should snapshot the checkpoint when this is True). ``should_stop``
        is True once ``patience`` evaluations have passed with no such improvement, but only
        after ``opt`` reaches ``warmup_options``.
        """
        if metric > self.best + self.min_delta:
            self.best = metric
            self.evals_since_best = 0
            return True, False
        self.evals_since_best += 1
        should_stop = (
            opt >= self.warmup_options and self.evals_since_best >= self.patience
        )
        return False, should_stop
