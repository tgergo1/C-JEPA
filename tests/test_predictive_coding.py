import numpy as np
from bio_snn.predictive_coding import PredictiveCodingNetwork


def test_predictive_forward_runs():
    net = PredictiveCodingNetwork([2, 3, 1])
    x = np.array([0.5, -0.5])
    out = net.forward(x)
    assert out.shape == (1,)


def test_network_reset_clears_traces():
    net = PredictiveCodingNetwork([2, 3, 1])
    # manually set traces to mimic activity
    for layer in net.layers:
        layer.post_traces.fill(1.0)
    assert any(layer.post_traces.any() for layer in net.layers)
    net.reset_state()
    assert all(not layer.post_traces.any() for layer in net.layers)
