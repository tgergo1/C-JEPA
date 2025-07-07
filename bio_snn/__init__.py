"""Simple biologically inspired SNN components."""

from .neuron import SpikingNeuron
from .network import Network
from .predictive_coding import PredictiveCodingNetwork
from .interface import run_simulation

__all__ = [
    "SpikingNeuron",
    "Network",
    "PredictiveCodingNetwork",
    "run_simulation",
]
