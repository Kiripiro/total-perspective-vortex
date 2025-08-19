from __future__ import annotations
from typing import Dict
import numpy as np
from sklearn.model_selection import StratifiedShuffleSplit


def stratified_holdout(
    X: np.ndarray,
    y: np.ndarray,
    test_size: float = 0.2,
    random_state: int = 42,
) -> Dict[str, np.ndarray]:
    """Perform a single stratified split into trainval/test.

    Returns a dict with X_trainval, y_trainval, X_test, y_test.
    """
    sss = StratifiedShuffleSplit(
        n_splits=1, test_size=test_size, random_state=random_state
    )
    (train_idx, test_idx), = sss.split(X, y)
    return {
        "X_trainval": X[train_idx],
        "y_trainval": y[train_idx],
        "X_test": X[test_idx],
        "y_test": y[test_idx],
    }

__all__ = ["stratified_holdout"]
