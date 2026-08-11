# Diffusion smoothing for ETH/USD

Reproducible IEEE-style experiment: smooth the hourly `Close` series of ETH/USD
with a one-dimensional diffusion denoiser and compare it with EMA baselines.

## Experimental protocol

- Input: ETH/USD hourly Close only; no future values are used by the causal EMA.
- Split: chronological 70% train, 15% validation, 15% test.
- Baselines: EMA spans 6, 12, 24 and 48 hours.
- Metrics: RMSE/MAE against a low-noise reference when available, directional
  accuracy, and roughness reduction. Report mean and standard deviation over
  repeated test windows, never a random split.

## Data format

CSV must contain `timestamp,close`, with one hourly observation per row.

## Quick start

```bash
pip install -r requirements.txt
PYTHONPATH=src python -m eth_diffusion.download \
  --output data/eth_usd_1h.csv --period 60d
PYTHONPATH=src python -m eth_diffusion.run --csv data/eth_usd_1h.csv
```

Yahoo Finance exposes a limited amount of intraday history; the downloader
defaults to 60 days for the 1-hour interval. This limitation should be stated
in the paper and, for a longer study, data should be collected periodically.

The model is implemented in `src/eth_diffusion/diffusion.py`. Fit it only on
the training split, then call `smooth` on validation/test data. Keep the seed,
window, number of diffusion steps, noise level and epochs in the experiment
metadata so the result is reproducible.
