import numpy as np
import pytest
from bio_snn.api import SNNModel
from bio_snn.layers import DenseLayer


def test_predict_before_compile_raises():
    model = SNNModel()
    with pytest.raises(RuntimeError):
        model.predict(np.array([0.0]))


def test_layers_property_after_compile():
    model = SNNModel()
    model.add_layer(DenseLayer, 2, 3)
    model.add_layer(DenseLayer, 3, 1)
    model.compile(lr=0.01)
    layers = model.layers
    assert len(layers) == 2
    assert all(hasattr(l, "forward") for l in layers)
