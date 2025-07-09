import numpy as np
from bio_snn.layers import DenseLayer, FlattenLayer, ConvLayer, RecurrentLayer


def test_dense_layer_forward_backward_update():
    layer = DenseLayer(2, 1)
    layer.weight.fill(0.5)
    x = np.array([1.0, -1.0])
    out = layer.forward(x)
    assert out.shape == (1,)
    assert np.allclose(out, [0.0])

    grad_x = layer.backward(np.array([1.0]))
    assert np.allclose(layer.grad_w, np.array([[1.0, -1.0]]))
    old = layer.weight.copy()
    layer.update(lr=0.1)
    assert np.allclose(layer.weight, old - 0.1 * layer.grad_w)
    assert grad_x.shape == x.shape


def test_flatten_layer_round_trip():
    layer = FlattenLayer()
    x = np.array([[1, 2], [3, 4]])
    flat = layer.forward(x)
    assert flat.shape == (4,)
    grad = layer.backward(np.arange(4))
    assert grad.shape == (2, 2)


def test_conv_layer_forward_backward_shapes():
    conv = ConvLayer(1, 1, kernel_size=2)
    conv.weight.fill(1.0)
    x = np.array([[[1, 2, 3], [4, 5, 6], [7, 8, 9]]], dtype=float)
    out = conv.forward(x)
    assert out.shape == (1, 2, 2)
    expected = np.array([[[12, 16], [24, 28]]], dtype=float)
    assert np.allclose(out, expected)

    grad_x = conv.backward(np.ones((1, 2, 2)))
    assert conv.grad_w.shape == conv.weight.shape
    assert grad_x.shape == x.shape


def test_recurrent_layer_state_update_and_reset():
    rnn = RecurrentLayer(1, 2)
    rnn.w_in.fill(0.5)
    rnn.w_rec.fill(0.0)
    x = np.array([1.0])
    h1 = rnn.forward(x)
    assert h1.shape == (2,)
    grad_x = rnn.backward(np.ones(2))
    old_w = rnn.w_in.copy()
    rnn.update(0.1)
    assert not np.allclose(old_w, rnn.w_in)
    assert grad_x.shape == x.shape
    rnn.reset_state()
    assert np.allclose(rnn.h, np.zeros_like(rnn.h))
