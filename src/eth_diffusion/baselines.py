import numpy as np


def ema(values: np.ndarray, span: int) -> np.ndarray:
    """Return the causal exponential moving average."""
    values = np.asarray(values, dtype=np.float64)
    if values.ndim != 1 or len(values) == 0:
        raise ValueError("values must be a non-empty one-dimensional array")
    if span < 1:
        raise ValueError("span must be positive")
    alpha = 2.0 / (span + 1.0)
    result = np.empty_like(values)
    result[0] = values[0]
    for i in range(1, len(values)):
        result[i] = alpha * values[i] + (1.0 - alpha) * result[i - 1]
    return result
