from dataclasses import dataclass

import numpy as np
import torch
from torch import nn


class NoisePredictor(nn.Module):
    def __init__(self, window: int, steps: int):
        super().__init__()
        self.time = nn.Embedding(steps, 32)
        self.net = nn.Sequential(
            nn.Linear(window + 32, 128), nn.SiLU(),
            nn.Linear(128, 128), nn.SiLU(), nn.Linear(128, window)
        )

    def forward(self, x, t):
        return self.net(torch.cat([x, self.time(t)], dim=1))


@dataclass
class DiffusionConfig:
    window: int = 48
    steps: int = 100
    epochs: int = 20
    batch_size: int = 64
    learning_rate: float = 1e-3
    seed: int = 42


class DiffusionSmoother:
    """A compact DDPM denoiser for one-dimensional hourly price windows."""

    def __init__(self, config: DiffusionConfig | None = None, device: str = "cpu"):
        self.config = config or DiffusionConfig()
        torch.manual_seed(self.config.seed)
        self.device = torch.device(device)
        betas = torch.linspace(1e-4, 0.02, self.config.steps, device=self.device)
        self.betas = betas
        self.alphas = 1.0 - betas
        self.alpha_bars = torch.cumprod(self.alphas, dim=0)
        self.model = NoisePredictor(self.config.window, self.config.steps).to(self.device)
        self.mean = 0.0
        self.std = 1.0

    def _windows(self, values: np.ndarray) -> np.ndarray:
        values = np.asarray(values, dtype=np.float32)
        if len(values) < self.config.window:
            raise ValueError("not enough observations for the configured window")
        return np.stack([
            values[i:i + self.config.window]
            for i in range(len(values) - self.config.window + 1)
        ])

    def fit(self, values: np.ndarray) -> list[float]:
        """Train only on the chronological training split and return epoch losses."""
        values = np.asarray(values, dtype=np.float32)
        self.mean, self.std = float(values.mean()), float(values.std() + 1e-8)
        windows = torch.from_numpy((self._windows(values) - self.mean) / self.std)
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.config.learning_rate)
        generator = torch.Generator().manual_seed(self.config.seed)
        losses = []
        self.model.train()
        for _ in range(self.config.epochs):
            order = torch.randperm(len(windows), generator=generator)
            epoch_loss = 0.0
            count = 0
            for start in range(0, len(order), self.config.batch_size):
                x0 = windows[order[start:start + self.config.batch_size]].to(self.device)
                t = torch.randint(self.config.steps, (len(x0),), device=self.device)
                noise = torch.randn_like(x0)
                abar = self.alpha_bars[t].unsqueeze(1)
                xt = abar.sqrt() * x0 + (1.0 - abar).sqrt() * noise
                loss = torch.mean((self.model(xt, t) - noise) ** 2)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                epoch_loss += float(loss.detach()) * len(x0)
                count += len(x0)
            losses.append(epoch_loss / count)
        return losses

    @torch.no_grad()
    def smooth(self, values: np.ndarray, noise_level: int = 25) -> np.ndarray:
        """Denoise overlapping windows and average their center predictions."""
        if not 1 <= noise_level < self.config.steps:
            raise ValueError("noise_level must be between 1 and steps - 1")
        windows = self._windows(np.asarray(values, dtype=np.float32))
        x0 = torch.from_numpy((windows - self.mean) / self.std).to(self.device)
        t0 = torch.full((len(x0),), noise_level, dtype=torch.long, device=self.device)
        x = self.alpha_bars[t0].sqrt().unsqueeze(1) * x0
        x += (1.0 - self.alpha_bars[t0]).sqrt().unsqueeze(1) * torch.randn_like(x0)
        self.model.eval()
        for step in range(noise_level, -1, -1):
            t = torch.full((len(x),), step, dtype=torch.long, device=self.device)
            eps = self.model(x, t)
            alpha = self.alphas[step]
            abar = self.alpha_bars[step]
            x = (x - (1 - alpha) / (1 - abar).sqrt() * eps) / alpha.sqrt()
            if step > 0:
                x += self.betas[step].sqrt() * torch.randn_like(x)
        denoised = x.cpu().numpy() * self.std + self.mean
        result = np.zeros(len(values), dtype=np.float64)
        counts = np.zeros(len(values), dtype=np.float64)
        for i, window in enumerate(denoised):
            result[i:i + self.config.window] += window
            counts[i:i + self.config.window] += 1
        return result / counts
