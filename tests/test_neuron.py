from bio_snn.neuron import SpikingNeuron


def test_neuron_spikes():
    n = SpikingNeuron()
    spikes = []
    for _ in range(50):
        spikes.append(n.forward([2.0]))
    assert any(spikes)
