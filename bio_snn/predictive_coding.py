import numpy as np
from .network import Network

class PredictiveCodingNetwork(Network):
    """Hierarchical network minimizing local prediction errors."""

    def __init__(self, sizes):
        super().__init__(sizes)
        self.pred_weights = [np.zeros((sizes[i], sizes[i+1])) for i in range(len(sizes)-1)]

    def forward(self, x):
        activations = [x]
        for i, layer in enumerate(self.layers):
            pred = self.pred_weights[i].T @ activations[-1]
            err = activations[-1] - pred
            out = layer.forward(err)
            activations.append(out)
        return activations[-1]
