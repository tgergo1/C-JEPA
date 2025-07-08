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


def test_adaptive_threshold_increases_then_decays():
    n = SpikingNeuron()
    # repeatedly provide strong input to trigger a spike
    spike = 0
    for _ in range(5):
        spike = n.forward([5.0])
        if spike:
            break
    assert spike == 1
    increased = n.threshold
    assert increased > n.baseline_thresh

    # run without input to allow decay toward baseline
    for _ in range(100):
        n.forward([0.0])
    assert n.threshold < increased
    assert abs(n.threshold - n.baseline_thresh) < 0.05


def test_last_spike_tracking():
    n = SpikingNeuron()
    assert n.last_spike == -float('inf')
    # ensure a spike occurs
    for _ in range(5):
        if n.forward([5.0]):
            break
    assert n.last_spike == 0.0
    # after a few steps without spikes the timer should increase
    for _ in range(3):
        n.forward([0.0])
    assert n.last_spike > 0
