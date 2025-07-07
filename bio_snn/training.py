"""Utilities for training networks on datasets with logging and checkpoints."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Tuple

import matplotlib.pyplot as plt
import numpy as np

from .energy_based import EnergyNetwork
from .datasets import load_digits_dataset


@dataclass
class TrainingConfig:
    sizes: Iterable[int]
    lr: float = 0.01
    epochs: int = 10
    batch_size: int = 32
    checkpoint_dir: str | Path = "checkpoints"
    log_interval: int = 1
    random_state: int | None = None


class EnergyTrainer:
    """Trainer for EnergyNetwork on the digits dataset."""

    def __init__(self, config: TrainingConfig) -> None:
        self.config = config
        self.net = EnergyNetwork(list(config.sizes))
        (self.X_train,
         self.X_test,
         self.y_train,
         self.y_test) = load_digits_dataset(random_state=config.random_state)
        self.checkpoint_dir = Path(config.checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.history: list[float] = []

    def _batch_iter(self) -> Iterable[Tuple[np.ndarray, np.ndarray]]:
        idx = np.random.permutation(len(self.X_train))
        for start in range(0, len(idx), self.config.batch_size):
            batch_idx = idx[start : start + self.config.batch_size]
            x = self.X_train[batch_idx]
            y = self.y_train[batch_idx]
            yield x, y

    def _save_checkpoint(self, epoch: int) -> None:
        path = self.checkpoint_dir / f"epoch_{epoch}.npz"
        np.savez(path, *self.net.weights)

    def train(self) -> None:
        for epoch in range(1, self.config.epochs + 1):
            epoch_loss = 0.0
            for x_batch, y_batch in self._batch_iter():
                for x, target in zip(x_batch, y_batch):
                    # one-hot encode target
                    t = np.zeros(self.config.sizes[-1])
                    t[target] = 1.0
                    loss = self.net.train_step(x, t, lr=self.config.lr)
                    epoch_loss += loss
            avg_loss = epoch_loss / len(self.X_train)
            self.history.append(avg_loss)
            if epoch % self.config.log_interval == 0:
                print(f"Epoch {epoch:03d}: loss={avg_loss:.4f}")
            self._save_checkpoint(epoch)
        self._plot_history()

    def _plot_history(self) -> None:
        plt.figure()
        plt.plot(self.history, label="train loss")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title("Training Loss")
        plt.legend()
        plt.tight_layout()
        plot_path = self.checkpoint_dir / "training_loss.png"
        plt.savefig(plot_path)
        plt.close()
