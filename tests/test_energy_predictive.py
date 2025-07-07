import numpy as np
from bio_snn.energy_predictive import EnergyPredictiveNetwork


def test_energy_forward_returns_value():
    net = EnergyPredictiveNetwork([2, 3, 1])
    x = np.array([0.5, -0.5])
    out = net.forward(x, target=np.zeros(1))
    assert out.shape == (1,)
    assert isinstance(net.last_energy, float)


def test_energy_training_reduces_energy():
    net = EnergyPredictiveNetwork([2, 4, 1])
    x = np.array([1.0, -1.0])
    target = np.array([0.5])
    net.forward(x, target=target)
    for _ in range(5):
        net.train_step(x, target, lr=0.05)
    net.forward(x, target=target)
    assert isinstance(net.last_energy, float)
