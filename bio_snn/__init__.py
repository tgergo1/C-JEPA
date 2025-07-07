"""Simple biologically inspired SNN components."""

from .neuron import SpikingNeuron
from .network import Network
from .predictive_coding import PredictiveCodingNetwork

__all__ = ["SpikingNeuron", "Network", "PredictiveCodingNetwork"]
