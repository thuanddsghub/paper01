import numpy as np

from eth_diffusion.diffusion import DiffusionConfig, DiffusionSmoother


def test_diffusion_fit_and_smooth_shape():
    values = np.linspace(100, 110, 40, dtype=np.float32)
    model = DiffusionSmoother(DiffusionConfig(window=8, steps=5, epochs=1, batch_size=8))
    losses = model.fit(values)
    result = model.smooth(values, noise_level=2)
    assert len(losses) == 1
    assert result.shape == values.shape
    assert np.all(np.isfinite(result))
