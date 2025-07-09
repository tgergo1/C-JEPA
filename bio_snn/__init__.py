"""Simple biologically inspired SNN components."""

from .neuron import SpikingNeuron
from .network import Network
from .predictive_coding import PredictiveCodingNetwork
from .energy_based import EnergyNetwork
from .core import BaseEnergyNetwork
try:
    from .torch_energy import TorchEnergyNetwork
except ImportError:  # pragma: no cover - torch is optional
    TorchEnergyNetwork = None
from .api import SNNModel
from .modular_network import ModularEnergyNetwork
from .layers import DenseLayer, ConvLayer, FlattenLayer, RecurrentLayer
from .energy_predictive import EnergyPredictiveNetwork
from .jepa import JointEmbeddingNetwork
from .interface import run_simulation
from .datasets import load_digits_dataset, gather_digits_dataset
from .training import EnergyTrainer, TrainingConfig

__all__ = [
    "SpikingNeuron",
    "Network",
    "PredictiveCodingNetwork",
    "EnergyNetwork",
    "BaseEnergyNetwork",
    "TorchEnergyNetwork",
    "ModularEnergyNetwork",
    "SNNModel",
    "DenseLayer",
    "ConvLayer",
    "FlattenLayer",
    "RecurrentLayer",
    "EnergyPredictiveNetwork",
    "run_simulation",
    "load_digits_dataset",
    "gather_digits_dataset",
    "EnergyTrainer",
    "TrainingConfig",
    "JointEmbeddingNetwork",
]
