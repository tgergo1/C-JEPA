"""Energy-based network supporting multiple layer types."""

from __future__ import annotations

import numpy as np

from .layers import BaseLayer


class ModularEnergyNetwork:
    """Flexible energy-based network using pluggable layers."""

    def __init__(self, layers: list[BaseLayer], reg: float = 1e-4) -> None:
        self.layers = layers
        self.reg = reg

    def forward(self, x: np.ndarray) -> np.ndarray:
        for layer in self.layers[:-1]:
            x = np.tanh(layer.forward(x))
        return self.layers[-1].forward(x)

    def energy(self, x: np.ndarray, target: np.ndarray) -> float:
        out = self.forward(x)
        mse = 0.5 * np.sum((out - target) ** 2)
        reg_term = 0.5 * self.reg * sum(
            np.sum(getattr(layer, "weight") ** 2)
            for layer in self.layers
            if hasattr(layer, "weight")
        )
        return mse + reg_term

    def train_step(self, x: np.ndarray, target: np.ndarray, lr: float = 0.01) -> float:
        activations = [x]
        for layer in self.layers[:-1]:
            activations.append(np.tanh(layer.forward(activations[-1])))
        out = self.layers[-1].forward(activations[-1])
        error = out - target
        grad = error
        grad = self.layers[-1].backward(grad)
        for i in range(len(self.layers) - 2, -1, -1):
            grad = grad * (1 - activations[i + 1] ** 2)
            grad = self.layers[i].backward(grad)
        for layer in self.layers:
            layer.update(lr)
        return np.linalg.norm(error)

    def reset_state(self) -> None:
        for layer in self.layers:
            layer.reset_state()
