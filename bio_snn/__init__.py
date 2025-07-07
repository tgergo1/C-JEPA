"""Simple biologically inspired SNN components."""

from .neuron import SpikingNeuron
from .network import Network
from .predictive_coding import PredictiveCodingNetwork
from .energy_based import EnergyNetwork
from .modular_network import ModularEnergyNetwork
from .layers import DenseLayer, ConvLayer, FlattenLayer, RecurrentLayer
from .energy_predictive import EnergyPredictiveNetwork
from .interface import run_simulation

__all__ = [
    "SpikingNeuron",
    "Network",
    "PredictiveCodingNetwork",
    "EnergyNetwork",
    "ModularEnergyNetwork",
    "DenseLayer",
    "ConvLayer",
    "FlattenLayer",
    "RecurrentLayer",
    "EnergyPredictiveNetwork",
    "run_simulation",
]
