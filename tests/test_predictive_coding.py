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


def test_prediction_weights_update_and_reset():
    net = PredictiveCodingNetwork([2, 3, 1])
    # boost layer weights so neurons fire
    for layer in net.layers:
        layer.weights.fill(1.0)
    x = np.array([1.0, 1.0])
    assert all(not w.any() for w in net.pred_weights)
    for _ in range(10):
        net.forward(x)
    # after multiple steps prediction weights should be updated
    assert any(w.any() for w in net.pred_weights)
    net.reset_state()
    # reset clears prediction weights
    assert all(not w.any() for w in net.pred_weights)
