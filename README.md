# Biologically Inspired Spiking Neural Network

This repository contains an initial prototype of a spiking neural network that
implements several biologically motivated mechanisms:

- **Spiking neurons** with adaptive thresholds and simple dendritic
  compartments.
- **Spike-Timing-Dependent Plasticity (STDP)** learning rule.
- A small **predictive coding** network that attempts to minimize local
  prediction errors between layers.

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

Run a short simulation:

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
steps. Extending the model with more detailed neuron dynamics and testing it on
continual learning tasks or robustness benchmarks are natural next steps.

## Command Line Interface

A simple CLI is provided to quickly run a simulation without writing any code. Example usage:

```bash
python -m bio_snn.interface --sizes 2,3,1 --input 1,0 --steps 100 --seed 0
```

This will construct a small network with the given layer sizes, feed the input vector for 100 steps and print the final output.
Providing a ``--seed`` ensures reproducible results, while ``--modulation`` can
be used to adjust the strength of plasticity online.
