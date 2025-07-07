import numpy as np

class SpikingNeuron:
    """A simple spiking neuron with adaptive threshold and two compartments."""

    def __init__(self, n_dendrites=2, tau_mem=10.0, tau_thresh=20.0, v_reset=0.0, v_thresh=1.0):
        self.n_dendrites = n_dendrites
        self.v = v_reset
        self.threshold = v_thresh
        self.v_reset = v_reset
        self.tau_mem = tau_mem
        self.tau_thresh = tau_thresh
        self.last_spike = -np.inf

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
