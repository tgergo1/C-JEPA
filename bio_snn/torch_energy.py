import torch
import torch.nn as nn
import torch.nn.functional as F

class TorchEnergyNetwork(nn.Module):
    """Energy-based network using PyTorch automatic differentiation."""

    def __init__(self, sizes, reg=1e-4, lr=0.01):
        super().__init__()
        self.layers = nn.ModuleList([
            nn.Linear(n_in, n_out) for n_in, n_out in zip(sizes[:-1], sizes[1:])
        ])
        self.reg = reg
        self.optimizer = torch.optim.SGD(self.parameters(), lr=lr)

    def forward(self, x):
        if not isinstance(x, torch.Tensor):
            x = torch.tensor(x, dtype=torch.float32)
        for layer in self.layers[:-1]:
            x = torch.tanh(layer(x))
        return self.layers[-1](x)

    def energy(self, x, target):
        out = self.forward(x)
        if not isinstance(target, torch.Tensor):
            target = torch.tensor(target, dtype=torch.float32)
        mse = 0.5 * torch.sum((out - target) ** 2)
        reg_term = 0.5 * self.reg * sum(torch.sum(p ** 2) for p in self.parameters())
        return mse + reg_term

    def train_step(self, x, target):
        self.optimizer.zero_grad()
        loss = self.energy(x, target)
        loss.backward()
        self.optimizer.step()
        return loss.item()
