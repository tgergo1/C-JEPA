import numpy as np
import torch
from bio_snn.torch_energy import TorchEnergyNetwork


def test_torch_energy_forward_shape():
    net = TorchEnergyNetwork([2, 3, 1])
    x = torch.tensor([0.5, -0.5])
    out = net.forward(x)
    assert out.shape == (1,)


def test_torch_training_reduces_energy():
    net = TorchEnergyNetwork([2, 4, 1], lr=0.05)
    x = torch.tensor([1.0, -1.0])
    target = torch.tensor([0.5])
    e1 = net.energy(x, target).item()
    for _ in range(50):
        net.train_step(x, target)
    e2 = net.energy(x, target).item()
    assert e2 < e1
