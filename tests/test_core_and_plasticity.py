import numpy as np
from bio_snn.plasticity import STDP, SynapticScaling
from bio_snn.network import Layer
from bio_snn.core import BaseEnergyNetwork


def test_stdp_weight_update_with_post_spike():
    stdp = STDP(lr=0.1, tau_pre=20.0, tau_post=20.0)
    w, pre_t, post_t = stdp.update(0.5, pre_spike=1, post_spike=0,
                                   pre_trace=0.0, post_trace=0.0)
    assert np.isclose(w, 0.5)
    w2, pre_t, post_t = stdp.update(w, pre_spike=0, post_spike=1,
                                    pre_trace=pre_t, post_trace=post_t)
    assert w2 < w


def test_synaptic_scaling_adjusts_weights():
    scaling = SynapticScaling(target=0.1, lr=0.5)
    w = np.array([1.0, 2.0])
    w_new = scaling.update(w, rate=0.0)
    expected = w * (1 + 0.5 * (0.1 - 0.0))
    assert np.allclose(w_new, expected)


def test_layer_reset_state_clears_activity():
    layer = Layer(1, 1)
    layer.weights.fill(5.0)
    layer.forward(np.array([2.0]))
    assert layer.pre_traces.any() and layer.post_traces.any()
    assert layer.avg_rates.any()
    layer.reset_state()
    assert not layer.pre_traces.any()
    assert not layer.post_traces.any()
    assert not layer.avg_rates.any()
    n = layer.neurons[0]
    assert n.v == n.v_reset
    assert n.threshold == n.baseline_thresh
    assert n.last_spike == -float("inf")


def test_core_backprop_and_apply_grads():
    net = BaseEnergyNetwork([2, 3, 1], reg=0.0)
    x = np.array([0.1, -0.2])
    out, activations = net.forward_activations(x)
    assert np.allclose(out, net.forward(x))
    grad_out = np.array([0.5])
    grads, grad_x = net.backprop(activations, grad_out)
    old = [w.copy() for w in net.weights]
    net.apply_grads(grads, lr=0.1)
    for o, n, g in zip(old, net.weights, grads):
        assert np.allclose(n, o - 0.1 * g)
    assert len(grads) == len(net.weights)
    for g, w in zip(grads, old):
        assert g.shape == w.shape
