"""
Walk-forward split helpers.

Index bounds use Python slice semantics: start is inclusive, end is exclusive.
"""

from dataclasses import dataclass
from typing import List


@dataclass(frozen=True)
class WalkForwardFold:
    fold_number: int
    train_start: int
    train_end: int
    test_start: int
    test_end: int

    @property
    def embargo_start(self) -> int:
        return self.train_end

    @property
    def embargo_end(self) -> int:
        return self.test_start

    @property
    def train_slice(self) -> slice:
        return slice(self.train_start, self.train_end)

    @property
    def test_slice(self) -> slice:
        return slice(self.test_start, self.test_end)


def _validate_split_args(
    data_length: int,
    train_window: int,
    test_window: int,
    step: int,
    embargo: int,
) -> None:
    if data_length < 0:
        raise ValueError("data_length must be non-negative")
    if train_window <= 0:
        raise ValueError("train_window must be positive")
    if test_window <= 0:
        raise ValueError("test_window must be positive")
    if step <= 0:
        raise ValueError("step must be positive")
    if embargo < 0:
        raise ValueError("embargo must be non-negative")


def rolling_splits(
    data_length: int,
    train_window: int,
    test_window: int,
    step: int,
    embargo: int = 0,
) -> List[WalkForwardFold]:
    """Create rolling train/test folds with an optional isolation gap."""
    _validate_split_args(data_length, train_window, test_window, step, embargo)

    folds: List[WalkForwardFold] = []
    train_start = 0
    fold_number = 0

    while True:
        train_end = train_start + train_window
        test_start = train_end + embargo
        test_end = test_start + test_window
        if test_end > data_length:
            break

        folds.append(
            WalkForwardFold(
                fold_number=fold_number,
                train_start=train_start,
                train_end=train_end,
                test_start=test_start,
                test_end=test_end,
            )
        )
        fold_number += 1
        train_start += step

    return folds


def expanding_splits(
    data_length: int,
    initial_train_window: int,
    test_window: int,
    step: int,
    embargo: int = 0,
) -> List[WalkForwardFold]:
    """Create expanding train/test folds with an optional isolation gap."""
    _validate_split_args(data_length, initial_train_window, test_window, step, embargo)

    folds: List[WalkForwardFold] = []
    train_end = initial_train_window
    fold_number = 0

    while True:
        test_start = train_end + embargo
        test_end = test_start + test_window
        if test_end > data_length:
            break

        folds.append(
            WalkForwardFold(
                fold_number=fold_number,
                train_start=0,
                train_end=train_end,
                test_start=test_start,
                test_end=test_end,
            )
        )
        fold_number += 1
        train_end += step

    return folds
