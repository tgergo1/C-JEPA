"""High level API for building and training spiking networks."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, List, Optional

import numpy as np

from .layers import BaseLayer
from .modular_network import ModularEnergyNetwork


@dataclass
class LayerSpec:
    """Specification for constructing a layer."""

    factory: Callable[..., BaseLayer]
    args: tuple
    kwargs: dict

    def build(self) -> BaseLayer:
        return self.factory(*self.args, **self.kwargs)


class SNNModel:
    """Sequential model built from layer specs.

    Examples
    --------
    >>> from bio_snn.layers import DenseLayer
    >>> model = SNNModel()
    >>> model.add_layer(DenseLayer, 2, 3)
    >>> model.add_layer(DenseLayer, 3, 1)
    >>> model.compile(lr=0.05)
    >>> out = model.predict(np.array([0.5, -0.5]))
    """

    def __init__(self) -> None:
        self.layer_specs: List[LayerSpec] = []
        self.net: Optional[ModularEnergyNetwork] = None
        self.energy_fn: Optional[Callable[[np.ndarray, np.ndarray], float]] = None
        self.lr: float = 0.01
        self.lr_schedule: Optional[Callable[[int], float]] = None
        self.step: int = 0

    def add_layer(self, factory: Callable[..., BaseLayer], *args, **kwargs) -> None:
        """Append a layer specification to the stack."""
        self.layer_specs.append(LayerSpec(factory, args, kwargs))

    def compile(
        self,
        energy_fn: Optional[Callable[[np.ndarray, np.ndarray], float]] = None,
        lr: float = 0.01,
        lr_schedule: Optional[Callable[[int], float]] = None,
    ) -> None:
        """Finalize model construction and choose training options."""
        layers = [spec.build() for spec in self.layer_specs]
        self.net = ModularEnergyNetwork(layers)
        self.energy_fn = energy_fn if energy_fn is not None else self.net.energy
        self.lr = lr
        self.lr_schedule = lr_schedule
        self.step = 0

    def predict(self, x: np.ndarray) -> np.ndarray:
        if self.net is None:
            raise RuntimeError("Model has not been compiled")
        return self.net.forward(x)

    def energy(self, x: np.ndarray, target: np.ndarray) -> float:
        if self.net is None or self.energy_fn is None:
            raise RuntimeError("Model has not been compiled")
        return self.energy_fn(x, target)

    def train_step(self, x: np.ndarray, target: np.ndarray) -> float:
        if self.net is None:
            raise RuntimeError("Model has not been compiled")
        lr = self.lr_schedule(self.step) if self.lr_schedule else self.lr
        self.step += 1
        return self.net.train_step(x, target, lr=lr)

    # expose low level layers for customization
    @property
    def layers(self) -> List[BaseLayer]:
        if self.net is None:
            raise RuntimeError("Model has not been compiled")
        return self.net.layers
