import numpy as np
from bio_snn.predictive_coding import PredictiveCodingNetwork


def test_predictive_forward_runs():
    net = PredictiveCodingNetwork([2, 3, 1])
    x = np.array([0.5, -0.5])
    out = net.forward(x)
    assert out.shape == (1,)
