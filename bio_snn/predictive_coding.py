import numpy as np
from .network import Network
from .neuron import SpikingNeuron

class PredictiveCodingNetwork(Network):
    """Hierarchical network minimizing local prediction errors."""

    def __init__(self, sizes, lr=0.01, neuron_cls=SpikingNeuron):
        super().__init__(sizes, neuron_cls=neuron_cls)
        self.pred_weights = [np.zeros((sizes[i], sizes[i+1])) for i in range(len(sizes)-1)]
        self.lr = lr

    def forward(self, x, modulation=1.0):
        """Propagate an input using local prediction errors and neuromodulation."""
        activations = [x]
        for i, layer in enumerate(self.layers):
            prev_act = activations[-1]

            # bottom-up activation for the next layer
            out = layer.forward(prev_act, modulation=modulation)

            # predict the next layer's activity from the current layer
            pred = self.pred_weights[i].T @ prev_act

            # prediction error for the next layer
            err = out - pred

            # simple local learning rule for prediction weights
            self.pred_weights[i] += self.lr * np.outer(prev_act, err)

            # propagate the error upward
            activations.append(err)

        return activations[-1]

    def reset_state(self) -> None:
        """Reset network and prediction weights."""
        super().reset_state()
        for w in self.pred_weights:
            w.fill(0)
