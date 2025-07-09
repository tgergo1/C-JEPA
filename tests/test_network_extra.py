import numpy as np
from bio_snn.network import Network


def test_network_forward_updates_weights():
    net = Network([1, 1])
    before = net.layers[0].weights.copy()
    out = net.forward(np.array([1.0]))
    assert out.shape == (1,)
    after = net.layers[0].weights
    assert not np.allclose(before, after)


def test_network_reset_state_clears_layers():
    net = Network([1, 1])
    net.forward(np.array([2.0]))
    net.reset_state()
    layer = net.layers[0]
    assert not layer.pre_traces.any()
    assert not layer.post_traces.any()
    assert not layer.avg_rates.any()
