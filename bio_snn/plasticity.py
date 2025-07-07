import numpy as np

class STDP:
    """Spike-Timing-Dependent Plasticity."""
    def __init__(self, lr=0.01, tau_pre=20.0, tau_post=20.0):
        self.lr = lr
        self.tau_pre = tau_pre
        self.tau_post = tau_post

    def update(self, w, pre_spike, post_spike, pre_trace, post_trace):
        pre_trace = pre_trace * np.exp(-1.0 / self.tau_pre) + pre_spike
        post_trace = post_trace * np.exp(-1.0 / self.tau_post) + post_spike
        dw = self.lr * (pre_spike * post_trace - post_spike * pre_trace)
        w += dw
        return w, pre_trace, post_trace
