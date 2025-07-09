import numpy as np
from bio_snn.api import SNNModel
from bio_snn.layers import DenseLayer


def test_highlevel_training_reduces_energy():
    model = SNNModel()
    model.add_layer(DenseLayer, 2, 3)
    model.add_layer(DenseLayer, 3, 1)
    model.compile(lr=0.05)
    x = np.array([1.0, -1.0])
    target = np.array([0.5])
    e1 = model.energy(x, target)
    for _ in range(50):
        model.train_step(x, target)
    e2 = model.energy(x, target)
    assert e2 < e1


def test_learning_schedule_called():
    calls = []
    def schedule(step: int) -> float:
        calls.append(step)
        return 0.01

    model = SNNModel()
    model.add_layer(DenseLayer, 1, 1)
    model.compile(lr=0.01, lr_schedule=schedule)
    model.train_step(np.array([0.0]), np.array([0.0]))
    assert calls == [0]
