# C-JEPA (Cognitive JEPA)

C-JEPA bundles a collection of biologically inspired spiking neural network
modules together with a minimal Joint Embedding Predictive Architecture.  The
code base was designed for experimentation and remains lightweight so it can be
extended easily.

## Features

- **Spiking neurons** with adaptive thresholds and simple dendritic compartments
  implemented in pure NumPy.
- **Spike-Timing-Dependent Plasticity (STDP)** learning rule.
- **Predictive coding** networks that minimise local prediction errors.
- **Energy-based** networks that support gradient based optimisation.
- **EnergyPredictiveNetwork** which combines predictive coding with a global
  energy objective.
- **JointEmbeddingNetwork** for JEPA-style training.
- **TorchEnergyNetwork** optional PyTorch backend with multi-GPU support.
- **Dataset utilities** for generating and loading digit data.
- **CLI tools** and **training helpers** for running experiments.
- **Tutorial notebooks** demonstrating reconstruction, sequence prediction and
  generative modelling.

## Installation

```bash
pip install numpy torch matplotlib scikit-learn
```

## Quick start

The predictive coding network can be run with just a few lines of code:

```python
from bio_snn.predictive_coding import PredictiveCodingNetwork
import numpy as np

net = PredictiveCodingNetwork([2, 3, 1])

x = np.array([1.0, 0.0])
for _ in range(100):
    y = net.forward(x)
print("Output:", y)
```

### Energy-based example

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

GPU acceleration is enabled by instantiating the network with
`backend="torch"`. Multiple GPUs can be used with `multi_gpu=True`.

### Training on digits

```python
from bio_snn.training import EnergyTrainer, TrainingConfig

cfg = TrainingConfig(sizes=[64, 32, 10], lr=0.05, epochs=5)
trainer = EnergyTrainer(cfg)
trainer.train()
```

Prepare the dataset beforehand using the helper script:

```bash
python scripts/gather_digits_dataset.py --out-dir data --test-size 0.2 --random-state 0
```

### Torch implementation

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

### Command line interface

A simple CLI is available to run simulations without writing any code:

```bash
python -m bio_snn.interface --sizes 2,3,1 --input 1,0 --steps 100 --seed 0
```

### Tutorial notebooks

- `energy_image_reconstruction.ipynb` – tiny autoencoder example.
- `energy_sequence_prediction.ipynb` – sequence prediction demo.
- `energy_generative_modeling.ipynb` – simple generative model.

### JEPA-style training

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

This trains the network so that the predicted embedding matches the target
embedding.
