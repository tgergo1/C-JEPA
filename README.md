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
pip install numpy torch matplotlib scikit-learn
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

To enable GPU acceleration, instantiate the network with ``backend="torch"``. If a
CUDA-capable device is available, computations will run on the GPU and can be
parallelized across multiple GPUs using ``multi_gpu=True``:

```python
net = EnergyNetwork([2, 4, 1], backend="torch", multi_gpu=True)

For larger scale experiments you can train the network on the
``sklearn`` digits dataset using the provided ``EnergyTrainer`` utility:

```python
from bio_snn.training import EnergyTrainer, TrainingConfig

cfg = TrainingConfig(sizes=[64, 32, 10], lr=0.05, epochs=5)
trainer = EnergyTrainer(cfg)
trainer.train()
```

This will load the dataset, run several training epochs while logging loss
values, save checkpoints in ``./checkpoints`` and produce a ``training_loss.png``
plot for quick visualization.

### Preparing the dataset

For quick experiments you can pre-generate the digits dataset splits using the
``scripts/gather_digits_dataset.py`` helper:

```bash
python scripts/gather_digits_dataset.py --out-dir data --test-size 0.2 --random-state 0
```

This will create ``data/train.npz`` and ``data/test.npz`` files which can be
loaded later with ``numpy.load`` for rapid prototyping.

### PyTorch implementation

For experiments requiring automatic differentiation or more advanced optimizers,
the repository also provides ``TorchEnergyNetwork`` which mirrors the numpy
version but is implemented with PyTorch. Using it is nearly identical:

```python
from bio_snn.torch_energy import TorchEnergyNetwork
import torch

net = TorchEnergyNetwork([2, 4, 1])
x = torch.tensor([1.0, -1.0])
target = torch.tensor([0.5])
for _ in range(50):
    net.train_step(x, target)
print("Energy:", net.energy(x, target).item())
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

## Tutorial Notebooks

Several Jupyter notebooks in the ``notebooks`` directory demonstrate how to
train the ``EnergyNetwork`` on common tasks:

- ``energy_image_reconstruction.ipynb`` shows a tiny autoencoder that
  reconstructs 8x8 images.
- ``energy_sequence_prediction.ipynb`` trains the network to predict the next
  value in a numerical sequence.
- ``energy_generative_modeling.ipynb`` illustrates how the network can learn a
  simple one-dimensional distribution for generative modeling experiments.

## JEPA-style Training

The package includes a lightweight `JointEmbeddingNetwork` for simple joint embedding predictive experiments. It shares an encoder for context and target inputs and learns to predict the target embedding from the context.

```python
from bio_snn.jepa import JointEmbeddingNetwork
import numpy as np

net = JointEmbeddingNetwork([2, 4, 2])
ctx = np.array([1.0, 0.0])
tgt = np.array([0.5, -0.5])
for _ in range(50):
    net.train_step(ctx, tgt, lr=0.05)
loss = net.loss(ctx, tgt)
print("Loss:", loss)
```

This trains the network so that the predicted embedding matches the target embedding.
