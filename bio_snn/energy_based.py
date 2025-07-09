import numpy as np

try:
    import torch
    TORCH_AVAILABLE = True
except Exception:  # pragma: no cover - torch is optional
    TORCH_AVAILABLE = False

class EnergyNetwork:
    """Energy-based network with optional accelerated backend."""

    def __init__(self, sizes, reg=1e-4, backend="numpy", device=None, multi_gpu=False):
        self.backend = backend
        self.reg = reg

        if backend == "torch":
            if not TORCH_AVAILABLE:
                raise ImportError("Torch backend requested but torch is not installed")
            self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
            self.layers = torch.nn.ModuleList()
            for n_in, n_out in zip(sizes[:-1], sizes[1:]):
                layer = torch.nn.Linear(n_in, n_out, bias=False)
                torch.nn.init.normal_(layer.weight, mean=0.0, std=0.1)
                self.layers.append(layer)
            model = torch.nn.Sequential(*self.layers).to(self.device)
            if multi_gpu and torch.cuda.device_count() > 1:
                model = torch.nn.DataParallel(model)
            self.model = model
        else:
            self.weights = [
                np.random.randn(n_out, n_in) * 0.1
                for n_in, n_out in zip(sizes[:-1], sizes[1:])
            ]

    def forward(self, x):
        """Forward pass with tanh activations."""
        if self.backend == "torch":
            if not torch.is_tensor(x):
                x = torch.tensor(x, dtype=torch.float32, device=self.device)
            else:
                x = x.to(self.device)
            if x.dim() == 1:
                x = x.unsqueeze(0)
            modules = self.model.module if isinstance(self.model, torch.nn.DataParallel) else self.model
            for i, layer in enumerate(modules):
                x = layer(x)
                if i < len(modules) - 1:
                    x = torch.tanh(x)
            if x.shape[0] == 1:
                x = x.squeeze(0)
            return x
        else:
            for w in self.weights[:-1]:
                x = np.tanh(w @ x)
            return self.weights[-1] @ x

    def energy(self, x, target):
        """Compute global energy for input and target."""
        if self.backend == "torch":
            if not torch.is_tensor(x):
                x_t = torch.tensor(x, dtype=torch.float32, device=self.device)
            else:
                x_t = x.to(self.device)
            if not torch.is_tensor(target):
                target_t = torch.tensor(target, dtype=torch.float32, device=self.device)
            else:
                target_t = target.to(self.device)
            out = self.forward(x_t)
            mse = 0.5 * torch.sum((out - target_t) ** 2)
            modules = self.model.module if isinstance(self.model, torch.nn.DataParallel) else self.model
            reg_term = 0.5 * self.reg * sum(torch.sum(layer.weight ** 2) for layer in modules)
            return (mse + reg_term).item()
        else:
            out = self.forward(x)
            mse = 0.5 * np.sum((out - target) ** 2)
            reg_term = 0.5 * self.reg * sum(np.sum(w ** 2) for w in self.weights)
            return mse + reg_term

    def train_step(self, x, target, lr=0.01):
        """Perform one gradient descent step to minimize energy."""
        if self.backend == "torch":
            self.model.train()
            if not torch.is_tensor(x):
                x_t = torch.tensor(x, dtype=torch.float32, device=self.device)
            else:
                x_t = x.to(self.device)
            if not torch.is_tensor(target):
                target_t = torch.tensor(target, dtype=torch.float32, device=self.device)
            else:
                target_t = target.to(self.device)
            if x_t.dim() == 1:
                x_t = x_t.unsqueeze(0)
            if target_t.dim() == 1:
                target_t = target_t.unsqueeze(0)

            modules = self.model.module if isinstance(self.model, torch.nn.DataParallel) else self.model
            for layer in modules:
                if layer.weight.grad is not None:
                    layer.weight.grad.zero_()

            out = self.forward(x_t)
            error = out - target_t
            mse = 0.5 * torch.sum(error ** 2)
            reg_term = 0.5 * self.reg * sum(torch.sum(layer.weight ** 2) for layer in modules)
            loss = mse + reg_term
            loss.backward()
            with torch.no_grad():
                for layer in modules:
                    layer.weight -= lr * layer.weight.grad
            return torch.linalg.norm(error).item()
        else:
            activations = [x]
            for w in self.weights[:-1]:
                activations.append(np.tanh(w @ activations[-1]))
            out = self.weights[-1] @ activations[-1]
            error = out - target

            # Backpropagate gradients
            grad_out = error
            grads = []
            for i in reversed(range(len(self.weights))):
                a_prev = activations[i]
                grad_w = np.outer(grad_out, a_prev) + self.reg * self.weights[i]
                grads.insert(0, grad_w)
                if i > 0:
                    grad_a = self.weights[i].T @ grad_out
                    grad_out = grad_a * (1 - activations[i] ** 2)

            # Update weights
            for w, g in zip(self.weights, grads):
                w -= lr * g

            return np.linalg.norm(error)

    # --- additional helpers for JEPA style training ---
    def forward_activations(self, x):
        """Return output and layer activations for numpy backend."""
        if self.backend == "torch":
            raise NotImplementedError("forward_activations not supported for torc"
                                    "h backend")
        activations = [x]
        for w in self.weights[:-1]:
            activations.append(np.tanh(w @ activations[-1]))
        out = self.weights[-1] @ activations[-1]
        activations.append(out)
        return out, activations

    def backprop(self, activations, grad_out):
        """Return gradients and gradient w.r.t. input for numpy backend."""
        if self.backend == "torch":
            raise NotImplementedError("backprop not supported for torch backend")
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
        """Update weights with provided gradients."""
        if self.backend == "torch":
            raise NotImplementedError("apply_grads not supported for torch backen"
                                    "d")
        for w, g in zip(self.weights, grads):
            w -= lr * g

