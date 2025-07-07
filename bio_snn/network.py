import numpy as np
from .neuron import SpikingNeuron
from .plasticity import STDP, SynapticScaling

class Layer:
    def __init__(self, n_in, n_neurons):
        self.weights = np.random.randn(n_neurons, n_in) * 0.1
        self.neurons = [SpikingNeuron() for _ in range(n_neurons)]
        self.stdp = STDP()
        self.scaling = SynapticScaling()
        self.pre_traces = np.zeros((n_neurons, n_in))
        self.post_traces = np.zeros(n_neurons)
        self.avg_rates = np.zeros(n_neurons)
        self.tau_rate = 100.0

    def forward(self, x, modulation=1.0):
        outputs = []
        for i, neuron in enumerate(self.neurons):
            inp = self.weights[i] * x
            spike = neuron.forward(inp)
            outputs.append(spike)

            # update plasticity with neuromodulation
            for j, pre in enumerate(x):
                self.weights[i, j], self.pre_traces[i, j], self.post_traces[i] = (
                    self.stdp.update(
                        self.weights[i, j],
                        pre,
                        spike,
                        self.pre_traces[i, j],
                        self.post_traces[i],
                        modulation=modulation,
                    )
                )

            # update firing rate estimate
            self.avg_rates[i] += (spike - self.avg_rates[i]) / self.tau_rate
            # apply synaptic scaling
            self.weights[i] = self.scaling.update(self.weights[i], self.avg_rates[i])

        return np.array(outputs)

class Network:
    """Simple feedforward SNN with STDP and predictive coding."""

    def __init__(self, sizes):
        self.layers = []
        for n_in, n_out in zip(sizes[:-1], sizes[1:]):
            self.layers.append(Layer(n_in, n_out))

    def forward(self, x, modulation=1.0):
        for layer in self.layers:
            x = layer.forward(x, modulation=modulation)
        return x
