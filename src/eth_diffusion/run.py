import argparse
import json
from pathlib import Path

import numpy as np

from .baselines import ema
from .data import chronological_split, load_close_csv
from .diffusion import DiffusionConfig, DiffusionSmoother
from .metrics import evaluate_smoother


def main():
    parser = argparse.ArgumentParser(description="Run the ETH/USD diffusion smoothing experiment")
    parser.add_argument("--csv", required=True, help="CSV containing timestamp,close")
    parser.add_argument("--output", default="results.json")
    parser.add_argument("--epochs", type=int, default=20)
    args = parser.parse_args()

    _, close = load_close_csv(args.csv)
    train, _, test = chronological_split(close)
    config = DiffusionConfig(epochs=args.epochs)
    model = DiffusionSmoother(config)
    losses = model.fit(train)
    diffusion = model.smooth(test)
    results = {
        "protocol": {"frequency": "1h", "field": "close", "split": [0.70, 0.15, 0.15]},
        "diffusion": {"final_train_loss": losses[-1], **evaluate_smoother(test, diffusion)},
        "ema": {
            str(span): evaluate_smoother(test, ema(test, span))
            for span in (6, 12, 24, 48)
        },
    }
    Path(args.output).write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
