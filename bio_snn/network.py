import numpy as np
from .neuron import SpikingNeuron
from .plasticity import STDP

class Layer:
    def __init__(self, n_in, n_neurons):
        self.weights = np.random.randn(n_neurons, n_in) * 0.1
        self.neurons = [SpikingNeuron() for _ in range(n_neurons)]
        self.stdp = STDP()
        self.pre_traces = np.zeros((n_neurons, n_in))
        self.post_traces = np.zeros(n_neurons)

    def forward(self, x):
        outputs = []
        for i, neuron in enumerate(self.neurons):
            inp = self.weights[i] * x
            spike = neuron.forward(inp)
            outputs.append(spike)
            # update plasticity
            for j, pre in enumerate(x):
                self.weights[i, j], self.pre_traces[i, j], self.post_traces[i] = (
                    self.stdp.update(self.weights[i, j], pre, spike, self.pre_traces[i, j], self.post_traces[i])
                )
        return np.array(outputs)

class Network:
    """Simple feedforward SNN with STDP and predictive coding."""
    def __init__(self, sizes):
        self.layers = []
        for n_in, n_out in zip(sizes[:-1], sizes[1:]):
            self.layers.append(Layer(n_in, n_out))

    def forward(self, x):
        for layer in self.layers:
            x = layer.forward(x)
        return x
