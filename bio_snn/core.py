import numpy as np

class BaseEnergyNetwork:
    """Core numpy implementation shared by higher-level networks."""

    def __init__(self, sizes, reg=1e-4):
        self.reg = reg
        self.weights = [
            np.random.randn(n_out, n_in) * 0.1
            for n_in, n_out in zip(sizes[:-1], sizes[1:])
        ]

    # --- generic forward/backprop utilities ---
    def forward(self, x):
        for w in self.weights[:-1]:
            x = np.tanh(w @ x)
        return self.weights[-1] @ x

    def energy(self, x, target):
        out = self.forward(x)
        mse = 0.5 * np.sum((out - target) ** 2)
        reg_term = 0.5 * self.reg * sum(np.sum(w ** 2) for w in self.weights)
        return mse + reg_term

    def train_step(self, x, target, lr=0.01):
        activations = [x]
        for w in self.weights[:-1]:
            activations.append(np.tanh(w @ activations[-1]))
        out = self.weights[-1] @ activations[-1]
        error = out - target

        grad_out = error
        grads = []
        for i in reversed(range(len(self.weights))):
            a_prev = activations[i]
            grad_w = np.outer(grad_out, a_prev) + self.reg * self.weights[i]
            grads.insert(0, grad_w)
            if i > 0:
                grad_a = self.weights[i].T @ grad_out
                grad_out = grad_a * (1 - activations[i] ** 2)

        for w, g in zip(self.weights, grads):
            w -= lr * g

        return np.linalg.norm(error)

    def forward_activations(self, x):
        activations = [x]
        for w in self.weights[:-1]:
            activations.append(np.tanh(w @ activations[-1]))
        out = self.weights[-1] @ activations[-1]
        activations.append(out)
        return out, activations

    def backprop(self, activations, grad_out):
        grads = []
        for i in reversed(range(len(self.weights))):
            a_prev = activations[i]
            grad_w = np.outer(grad_out, a_prev) + self.reg * self.weights[i]
            grads.insert(0, grad_w)
            if i > 0:
                grad_a = self.weights[i].T @ grad_out
                grad_out = grad_a * (1 - activations[i] ** 2)
        return grads, grad_out

    def apply_grads(self, grads, lr):
        for w, g in zip(self.weights, grads):
            w -= lr * g
