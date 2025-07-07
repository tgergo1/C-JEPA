"""Modular neural network layers for energy-based models."""

from __future__ import annotations

import numpy as np


class BaseLayer:
    """Abstract base layer."""

    def forward(self, x: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def backward(self, grad: np.ndarray) -> np.ndarray:
        """Return gradient w.r.t. input and accumulate parameter grads."""
        raise NotImplementedError

    def update(self, lr: float) -> None:
        pass

    def reset_state(self) -> None:
        pass


class DenseLayer(BaseLayer):
    """Simple fully connected layer."""

    def __init__(self, n_in: int, n_out: int) -> None:
        self.weight = np.random.randn(n_out, n_in) * 0.1
        self.grad_w = np.zeros_like(self.weight)
        self.x: np.ndarray | None = None

    def forward(self, x: np.ndarray) -> np.ndarray:
        self.x = x
        return self.weight @ x

    def backward(self, grad: np.ndarray) -> np.ndarray:
        self.grad_w = np.outer(grad, self.x)
        return self.weight.T @ grad

    def update(self, lr: float) -> None:
        self.weight -= lr * self.grad_w


class ConvLayer(BaseLayer):
    """Naive 2D convolution for single or multi-channel inputs."""

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int) -> None:
        self.weight = np.random.randn(out_channels, in_channels, kernel_size, kernel_size) * 0.1
        self.grad_w = np.zeros_like(self.weight)
        self.kernel_size = kernel_size
        self.x: np.ndarray | None = None

    def forward(self, x: np.ndarray) -> np.ndarray:
        """x shape: (C, H, W)."""
        self.x = x
        C, H, W = x.shape
        F = self.weight.shape[0]
        k = self.kernel_size
        H_out = H - k + 1
        W_out = W - k + 1
        out = np.zeros((F, H_out, W_out))
        for f in range(F):
            for c in range(C):
                w = self.weight[f, c]
                for i in range(H_out):
                    for j in range(W_out):
                        region = x[c, i : i + k, j : j + k]
                        out[f, i, j] += np.sum(region * w)
        return out

    def backward(self, grad: np.ndarray) -> np.ndarray:
        x = self.x
        assert x is not None
        C, H, W = x.shape
        F = self.weight.shape[0]
        k = self.kernel_size
        H_out = H - k + 1
        W_out = W - k + 1
        self.grad_w.fill(0)
        grad_x = np.zeros_like(x)
        for f in range(F):
            for c in range(C):
                w = self.weight[f, c]
                for i in range(H_out):
                    for j in range(W_out):
                        region = x[c, i : i + k, j : j + k]
                        self.grad_w[f, c] += grad[f, i, j] * region
                        grad_x[c, i : i + k, j : j + k] += grad[f, i, j] * w
        return grad_x

    def update(self, lr: float) -> None:
        self.weight -= lr * self.grad_w


class FlattenLayer(BaseLayer):
    """Flatten arbitrary input to a 1D vector."""

    def __init__(self) -> None:
        self.shape: tuple[int, ...] | None = None

    def forward(self, x: np.ndarray) -> np.ndarray:
        self.shape = x.shape
        return x.ravel()

    def backward(self, grad: np.ndarray) -> np.ndarray:
        assert self.shape is not None
        return grad.reshape(self.shape)


class RecurrentLayer(BaseLayer):
    """Simple recurrent layer with tanh activation."""

    def __init__(self, n_in: int, n_hidden: int) -> None:
        self.w_in = np.random.randn(n_hidden, n_in) * 0.1
        self.w_rec = np.random.randn(n_hidden, n_hidden) * 0.1
        self.h = np.zeros(n_hidden)
        self.grad_w_in = np.zeros_like(self.w_in)
        self.grad_w_rec = np.zeros_like(self.w_rec)
        self.x: np.ndarray | None = None
        self.prev_h: np.ndarray | None = None

    def forward(self, x: np.ndarray) -> np.ndarray:
        self.x = x
        self.prev_h = self.h.copy()
        self.h = np.tanh(self.w_in @ x + self.w_rec @ self.h)
        return self.h

    def backward(self, grad: np.ndarray) -> np.ndarray:
        dtanh = (1 - self.h ** 2) * grad
        assert self.x is not None and self.prev_h is not None
        self.grad_w_in = np.outer(dtanh, self.x)
        self.grad_w_rec = np.outer(dtanh, self.prev_h)
        grad_x = self.w_in.T @ dtanh
        self.h = self.prev_h  # restore previous state for truncated BPTT
        return grad_x

    def update(self, lr: float) -> None:
        self.w_in -= lr * self.grad_w_in
        self.w_rec -= lr * self.grad_w_rec

    def reset_state(self) -> None:
        self.h.fill(0)
