import numpy as np
from bio_snn.modular_network import ModularEnergyNetwork
from bio_snn.layers import DenseLayer, ConvLayer, FlattenLayer, RecurrentLayer


def test_conv_forward_shape():
    conv = ConvLayer(1, 2, kernel_size=2)
    flat = FlattenLayer()
    dense = DenseLayer(18, 1)  # output 2*(3*3) = 18 for 4x4 input
    net = ModularEnergyNetwork([conv, flat, dense])
    x = np.random.randn(1, 4, 4)
    out = net.forward(x)
    assert out.shape == (1,)


def test_recurrent_state_update():
    rnn = RecurrentLayer(1, 3)
    dense = DenseLayer(3, 1)
    net = ModularEnergyNetwork([rnn, dense])
    x1 = np.array([0.5])
    out1 = net.forward(x1)
    x2 = np.array([-0.2])
    out2 = net.forward(x2)
    assert out1.shape == out2.shape == (1,)
    assert not np.allclose(out1, out2)


def test_modular_training_reduces_energy():
    layers = [DenseLayer(2, 3), DenseLayer(3, 1)]
    net = ModularEnergyNetwork(layers)
    x = np.array([1.0, -1.0])
    target = np.array([0.5])
    e1 = net.energy(x, target)
    for _ in range(50):
        net.train_step(x, target, lr=0.05)
    e2 = net.energy(x, target)
    assert e2 < e1
