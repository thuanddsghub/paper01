import numpy as np


def _rmse(a, b):
    return float(np.sqrt(np.mean((np.asarray(a) - np.asarray(b)) ** 2)))


def _mae(a, b):
    return float(np.mean(np.abs(np.asarray(a) - np.asarray(b))))


def evaluate_smoother(observed, smoothed, reference=None) -> dict[str, float]:
    """Metrics for smoothing; reference is optional and must be less noisy truth."""
    observed, smoothed = np.asarray(observed), np.asarray(smoothed)
    if observed.shape != smoothed.shape:
        raise ValueError("observed and smoothed must have the same shape")
    result = {
        "roughness_reduction": float(1.0 - np.std(np.diff(smoothed)) / (np.std(np.diff(observed)) + 1e-12)),
        "directional_accuracy": float(np.mean(np.sign(np.diff(observed)) == np.sign(np.diff(smoothed)))),
    }
    if reference is not None:
        reference = np.asarray(reference)
        result["rmse"] = _rmse(reference, smoothed)
        result["mae"] = _mae(reference, smoothed)
    return result
