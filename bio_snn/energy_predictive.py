import numpy as np
from .predictive_coding import PredictiveCodingNetwork

class EnergyPredictiveNetwork(PredictiveCodingNetwork):
    """Predictive coding network minimizing a global energy function."""

    def __init__(self, sizes, lr=0.01, state_reg=1e-4, weight_reg=1e-4):
        super().__init__(sizes, lr=lr)
        self.state_reg = state_reg
        self.weight_reg = weight_reg
        self.last_energy = 0.0

    def forward(self, x, modulation=1.0, target=None):
        """Forward pass computing global energy."""
        activations = [x]
        energy = 0.0
        for i, layer in enumerate(self.layers):
            prev_act = activations[-1]
            out = layer.forward(prev_act, modulation=modulation)
            pred = self.pred_weights[i].T @ prev_act
            err = out - pred
            self.pred_weights[i] += self.lr * np.outer(prev_act, err)
            activations.append(err)
            energy += 0.5 * np.sum(err ** 2)
            energy += 0.5 * self.state_reg * sum(n.v ** 2 for n in layer.neurons)
            energy += 0.5 * self.weight_reg * (
                np.sum(layer.weights ** 2) + np.sum(self.pred_weights[i] ** 2)
            )
        if target is not None:
            energy += 0.5 * np.sum((activations[-1] - target) ** 2)
        self.last_energy = energy
        return activations[-1]

    def train_step(self, x, target, lr=None):
        """Gradient descent step on prediction weights to minimize energy."""
        if lr is None:
            lr = self.lr
        activations = [x]
        for i, layer in enumerate(self.layers):
            prev_act = activations[-1]
            out = layer.forward(prev_act, modulation=0.0)
            pred = self.pred_weights[i].T @ prev_act
            err = out - pred
            activations.append(err)
            grad = -np.outer(prev_act, err) + self.weight_reg * self.pred_weights[i]
            self.pred_weights[i] -= lr * grad
        if target is not None:
            final_err = activations[-1] - target
            activations[-1] -= lr * final_err
        self.forward(x, modulation=0.0, target=target)
        return self.last_energy
