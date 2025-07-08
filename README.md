# Biologically Inspired Spiking Neural Network

This repository contains an initial prototype of a spiking neural network that
implements several biologically motivated mechanisms:

- **Spiking neurons** with adaptive thresholds and simple dendritic
  compartments.
- **Spike-Timing-Dependent Plasticity (STDP)** learning rule.
- A small **predictive coding** network that attempts to minimize local
  prediction errors between layers.
- A simple **energy-based** network for gradient descent experiments.

The implementation is intentionally lightweight and serves as a starting point
for experimenting with more elaborate models that incorporate dendritic
computation, neuromodulation and hierarchical organization as described in the
project proposal.

Recent refactoring introduced reproducible initialization and state reset
functions making the code easier to use for controlled experiments.

## Basic Usage

Install dependencies:

```bash
pip install numpy matplotlib
```

Run a short simulation with the predictive coding network:

```python
from bio_snn.predictive_coding import PredictiveCodingNetwork
import numpy as np

net = PredictiveCodingNetwork([2, 3, 1])

x = np.array([1.0, 0.0])
for _ in range(100):
    y = net.forward(x)
print("Output:", y)
```

This will create a tiny network with one hidden layer and run it for a few
steps.

An additional ``EnergyNetwork`` class demonstrates energy-based training with
backpropagation. Example:

```python
from bio_snn.energy_based import EnergyNetwork
import numpy as np

net = EnergyNetwork([2, 4, 1])
x = np.array([1.0, -1.0])
target = np.array([0.5])
for _ in range(50):
    net.train_step(x, target)
print("Energy:", net.energy(x, target))
```

## High-Level API

A small helper module simplifies building networks and running energy-based
training schedules.  It also lets you swap in custom neuron classes or energy
functions:

```python
from bio_snn.api import build_snn, LearningSchedule, train_energy
import numpy as np

net = build_snn([2, 4, 1], network_type="energy")
schedule = LearningSchedule(lr=0.05, epochs=20)
train_energy(net, np.array([1.0, -1.0]), np.array([0.5]), schedule)
print("Energy:", net.energy(np.array([1.0, -1.0]), np.array([0.5])))
```

To experiment with a different objective you can pass custom functions:

```python
def my_energy(out, target, weights, reg):
    return 0.5 * np.sum((out - target) ** 2) + 0.1 * sum(np.sum(w**2) for w in weights)

def my_grad(out, target):
    return out - target

net = build_snn(
    [2, 4, 1], network_type="energy", energy_fn=my_energy, grad_fn=my_grad
)
```

## Command Line Interface

A simple CLI is provided to quickly run a simulation without writing any code.
Example usage:

```bash
python -m bio_snn.interface --sizes 2,3,1 --input 1,0 --steps 100 --seed 0
```

This will construct a small network with the given layer sizes, feed the input
vector for 100 steps and print the final output.  Providing a ``--seed`` ensures
reproducible results, while ``--modulation`` can be used to adjust the strength
of plasticity online.
