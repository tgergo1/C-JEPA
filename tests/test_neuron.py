from bio_snn.neuron import SpikingNeuron


def test_neuron_spikes():
    n = SpikingNeuron()
    spikes = []
    for _ in range(50):
        spikes.append(n.forward([2.0]))
    assert any(spikes)


def test_reset_restores_baseline():
    n = SpikingNeuron()
    for _ in range(10):
        n.forward([2.0])
    assert n.v != n.v_reset or n.threshold != n.baseline_thresh
    n.reset()
    assert n.v == n.v_reset
    assert n.threshold == n.baseline_thresh
