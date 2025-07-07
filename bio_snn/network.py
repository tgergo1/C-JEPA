"""Simple feedforward SNN architecture."""

from dataclasses import dataclass, field
from typing import Type
import numpy as np
from .neuron import SpikingNeuron
from .plasticity import STDP, SynapticScaling

@dataclass
class Layer:
    """Single SNN layer with local plasticity."""

    n_in: int
    n_neurons: int
    neuron_cls: Type[SpikingNeuron] = SpikingNeuron
    stdp: STDP = field(default_factory=STDP)
    scaling: SynapticScaling = field(default_factory=SynapticScaling)
    tau_rate: float = 100.0

    weights: np.ndarray = field(init=False)
    neurons: list[SpikingNeuron] = field(init=False)
    pre_traces: np.ndarray = field(init=False)
    post_traces: np.ndarray = field(init=False)
    avg_rates: np.ndarray = field(init=False)

    def __post_init__(self) -> None:
        self.weights = np.random.randn(self.n_neurons, self.n_in) * 0.1
        self.neurons = [self.neuron_cls() for _ in range(self.n_neurons)]
        self.pre_traces = np.zeros((self.n_neurons, self.n_in))
        self.post_traces = np.zeros(self.n_neurons)
        self.avg_rates = np.zeros(self.n_neurons)

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

    def reset_state(self) -> None:
        """Reset neuron states and traces for a fresh simulation."""
        for neuron in self.neurons:
            neuron.reset()
        self.pre_traces.fill(0)
        self.post_traces.fill(0)
        self.avg_rates.fill(0)

class Network:
    """Simple feedforward SNN with STDP and predictive coding."""

    def __init__(self, sizes, neuron_cls: Type[SpikingNeuron] = SpikingNeuron):
        self.layers = []
        for n_in, n_out in zip(sizes[:-1], sizes[1:]):
            self.layers.append(Layer(n_in, n_out, neuron_cls=neuron_cls))

    def forward(self, x, modulation=1.0):
        for layer in self.layers:
            x = layer.forward(x, modulation=modulation)
        return x

    def reset_state(self) -> None:
        """Reset all layers for a new episode."""
        for layer in self.layers:
            layer.reset_state()
