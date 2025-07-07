"""Plasticity mechanisms used in the spiking network."""

from dataclasses import dataclass
import numpy as np


@dataclass
class STDP:
    """Spike-Timing-Dependent Plasticity with optional neuromodulation."""

    lr: float = 0.01
    tau_pre: float = 20.0
    tau_post: float = 20.0

    def update(self, w, pre_spike, post_spike, pre_trace, post_trace, modulation=1.0):
        """Update a synapse given pre/post spikes and traces."""
        pre_trace = pre_trace * np.exp(-1.0 / self.tau_pre) + pre_spike
        post_trace = post_trace * np.exp(-1.0 / self.tau_post) + post_spike
        dw = modulation * self.lr * (pre_spike * post_trace - post_spike * pre_trace)
        w += dw
        return w, pre_trace, post_trace


@dataclass
class SynapticScaling:
    """Homeostatic synaptic scaling to maintain target firing rates."""

    target: float = 0.1
    lr: float = 0.001

    def update(self, weights, rate):
        scale = 1.0 + self.lr * (self.target - rate)
        return weights * scale
