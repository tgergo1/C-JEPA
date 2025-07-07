"""Simple biologically inspired SNN components."""

from .neuron import SpikingNeuron
from .network import Network
from .predictive_coding import PredictiveCodingNetwork
from .energy_based import EnergyNetwork
from .interface import run_simulation
from .datasets import load_digits_dataset
from .training import EnergyTrainer, TrainingConfig

__all__ = [
    "SpikingNeuron",
    "Network",
    "PredictiveCodingNetwork",
    "EnergyNetwork",
    "run_simulation",
    "load_digits_dataset",
    "EnergyTrainer",
    "TrainingConfig",
]
