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


def test_custom_energy_fn_used():
    def energy(out, target, weights, reg):
        return np.sum((out - target) ** 2)

    def grad(out, target):
        return 2 * (out - target)

    net = build_snn([2, 2, 1], network_type="energy", energy_fn=energy, grad_fn=grad)
    x = np.array([0.1, 0.2])
    target = np.array([0.0])
    e1 = net.energy(x, target)
    train_energy(net, x, target, LearningSchedule(lr=0.1, epochs=5))
    e2 = net.energy(x, target)
    assert e2 < e1
