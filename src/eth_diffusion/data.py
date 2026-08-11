import csv
from pathlib import Path

import numpy as np


def load_close_csv(path: str | Path, close_column: str = "close") -> tuple[np.ndarray, np.ndarray]:
    """Load timestamp and Close from a CSV, preserving chronological order."""
    with Path(path).open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if not rows or close_column not in rows[0]:
        raise ValueError(f"CSV must contain a '{close_column}' column")
    timestamps, closes = [], []
    for row in rows:
        try:
            timestamps.append(row.get("timestamp", row.get("time", str(len(timestamps)))))
            closes.append(float(row[close_column]))
        except (TypeError, ValueError):
            continue
    if len(closes) < 32 or not np.all(np.isfinite(closes)):
        raise ValueError("CSV must contain at least 32 finite Close values")
    return np.asarray(timestamps), np.asarray(closes, dtype=np.float64)


def chronological_split(values: np.ndarray, train_ratio: float = 0.7, val_ratio: float = 0.15):
    values = np.asarray(values, dtype=np.float64)
    if not 0 < train_ratio < 1 or not 0 <= val_ratio < 1 or train_ratio + val_ratio >= 1:
        raise ValueError("train_ratio and val_ratio must leave a non-empty test split")
    n_train = int(len(values) * train_ratio)
    n_val = int(len(values) * val_ratio)
    return values[:n_train], values[n_train:n_train + n_val], values[n_train + n_val:]
