import numpy as np


def compute_mse(X: np.ndarray, X_approx: np.ndarray) -> float:
    return np.mean(np.sum((X - X_approx) ** 2, axis=-1))
