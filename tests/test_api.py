import numpy as np
from bio_snn.api import build_snn, LearningSchedule, train_energy


def test_build_predictive_network():
    net = build_snn([2, 3, 1], network_type="predictive")
    x = np.array([0.5, -0.5])
    out = net.forward(x)
    assert out.shape == (1,)


def test_energy_schedule_reduces_energy():
    net = build_snn([2, 4, 1], network_type="energy")
    x = np.array([1.0, -1.0])
    target = np.array([0.5])
    e1 = net.energy(x, target)
    schedule = LearningSchedule(lr=0.05, epochs=20)
    train_energy(net, x, target, schedule)
    e2 = net.energy(x, target)
    assert e2 < e1
