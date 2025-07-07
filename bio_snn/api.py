from dataclasses import dataclass
from typing import Sequence, Type, Callable, Optional
import numpy as np

from .neuron import SpikingNeuron
from .network import Network
from .predictive_coding import PredictiveCodingNetwork
from .energy_based import EnergyNetwork


@dataclass
class LearningSchedule:
    """Simple learning schedule for training loops."""

    lr: float = 0.01
    epochs: int = 1


def build_snn(
    sizes: Sequence[int],
    *,
    network_type: str = "predictive",
    neuron_cls: Type[SpikingNeuron] = SpikingNeuron,
    energy_fn: Optional[Callable] = None,
    reg: float = 1e-4,
):
    """Create an SNN instance based on the requested type."""
    if network_type == "predictive":
        return PredictiveCodingNetwork(sizes, neuron_cls=neuron_cls)
    if network_type == "feedforward":
        return Network(sizes, neuron_cls=neuron_cls)
    if network_type == "energy":
        return EnergyNetwork(sizes, reg=reg, energy_fn=energy_fn)
    raise ValueError(f"Unknown network_type: {network_type}")


def train_energy(
    net: EnergyNetwork,
    x: np.ndarray,
    target: np.ndarray,
    schedule: LearningSchedule,
):
    """Train an EnergyNetwork according to the schedule."""
    for _ in range(schedule.epochs):
        net.train_step(x, target, lr=schedule.lr)
