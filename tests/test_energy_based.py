import numpy as np
from bio_snn.energy_based import EnergyNetwork


def test_energy_forward_shape():
    net = EnergyNetwork([2, 3, 1])
    x = np.array([0.5, -0.5])
    out = net.forward(x)
    assert out.shape == (1,)


def test_training_reduces_energy():
    net = EnergyNetwork([2, 4, 1])
    x = np.array([1.0, -1.0])
    target = np.array([0.5])
    e1 = net.energy(x, target)
    for _ in range(50):
        net.train_step(x, target, lr=0.05)
    e2 = net.energy(x, target)
    assert e2 < e1
