import numpy as np
from bio_snn.jepa import JointEmbeddingNetwork


def test_jepa_forward_shapes():
    net = JointEmbeddingNetwork([2, 4, 2])
    ctx = np.array([0.5, -0.5])
    tgt = np.array([-0.2, 0.1])
    pred, target = net.forward(ctx, tgt)
    assert pred.shape == target.shape == (2,)


def test_jepa_training_reduces_loss():
    net = JointEmbeddingNetwork([2, 4, 2])
    ctx = np.array([1.0, -1.0])
    tgt = np.array([0.5, 0.5])
    l1 = net.loss(ctx, tgt)
    for _ in range(20):
        net.train_step(ctx, tgt, lr=0.05)
    l2 = net.loss(ctx, tgt)
    assert l2 < l1
