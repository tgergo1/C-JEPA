import numpy as np

class STDP:
    """Spike-Timing-Dependent Plasticity with optional neuromodulation."""

    def __init__(self, lr=0.01, tau_pre=20.0, tau_post=20.0):
        self.lr = lr
        self.tau_pre = tau_pre
        self.tau_post = tau_post

    def update(self, w, pre_spike, post_spike, pre_trace, post_trace, modulation=1.0):
        """Update a synapse given pre/post spikes and traces."""
        pre_trace = pre_trace * np.exp(-1.0 / self.tau_pre) + pre_spike
        post_trace = post_trace * np.exp(-1.0 / self.tau_post) + post_spike
        dw = modulation * self.lr * (pre_spike * post_trace - post_spike * pre_trace)
        w += dw
        return w, pre_trace, post_trace


class SynapticScaling:
    """Homeostatic synaptic scaling to maintain target firing rates."""

    def __init__(self, target=0.1, lr=0.001):
        self.target = target
        self.lr = lr

    def update(self, weights, rate):
        scale = 1.0 + self.lr * (self.target - rate)
        return weights * scale
