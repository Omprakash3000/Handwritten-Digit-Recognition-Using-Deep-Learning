"""Early stopping based on validation metric."""

from __future__ import annotations

import copy

from torch import nn


class EarlyStopping:
    def __init__(self, patience: int = 5, mode: str = "max") -> None:
        self.patience = patience
        self.mode = mode
        self.counter = 0
        self.best_score: float | None = None
        self.best_state: dict | None = None
        self.should_stop = False

    def step(self, score: float, model: nn.Module) -> bool:
        improved = (
            self.best_score is None
            or (self.mode == "max" and score > self.best_score)
            or (self.mode == "min" and score < self.best_score)
        )
        if improved:
            self.best_score = score
            self.best_state = copy.deepcopy(model.state_dict())
            self.counter = 0
            return True
        self.counter += 1
        if self.counter >= self.patience:
            self.should_stop = True
        return False
