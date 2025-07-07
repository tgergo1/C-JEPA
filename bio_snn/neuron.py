"""Spiking neuron models used by the network."""

from dataclasses import dataclass, field
import numpy as np


@dataclass
class SpikingNeuron:
    """A simple spiking neuron with adaptive threshold and two compartments."""

    n_dendrites: int = 2
    tau_mem: float = 10.0
    tau_thresh: float = 20.0
    v_reset: float = 0.0
    v_thresh: float = 1.0

    v: float = field(init=False)
    threshold: float = field(init=False)
    last_spike: float = field(default=-np.inf, init=False)
    baseline_thresh: float = field(init=False)

    def __post_init__(self) -> None:
        self.v = self.v_reset
        self.threshold = self.v_thresh
        self.baseline_thresh = self.v_thresh

    def forward(self, inputs, dt=1.0):
        """Integrate inputs and return spike (0/1)."""
        dv = (np.sum(inputs) - self.v) / self.tau_mem
        self.v += dv * dt
        if self.v >= self.threshold:
            self.v = self.v_reset
            self.threshold += 0.5  # adaptive threshold
            self.last_spike = 0.0
            return 1
        # decay threshold toward baseline
        self.threshold += (1.0 - self.threshold) / self.tau_thresh
        if self.last_spike != -np.inf:
            self.last_spike += dt
        return 0

    def reset(self) -> None:
        """Reset membrane potential and adaptive threshold."""
        self.v = self.v_reset
        self.threshold = self.baseline_thresh
        self.last_spike = -np.inf
