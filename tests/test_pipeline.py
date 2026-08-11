import numpy as np

from eth_diffusion.baselines import ema
from eth_diffusion.metrics import evaluate_smoother


def test_ema_starts_at_first_observation():
    result = ema(np.array([1.0, 3.0, 3.0]), span=2)
    assert result[0] == 1.0
    assert result[-1] > result[0]


def test_metrics_are_finite():
    observed = np.array([1.0, 2.0, 1.0, 2.0])
    result = evaluate_smoother(observed, ema(observed, 2))
    assert set(result) == {"roughness_reduction", "directional_accuracy"}
    assert all(np.isfinite(list(result.values())))
