"""Dataset loaders and preprocessing utilities for standard ML datasets."""

from .digits import load_digits_dataset, gather_digits_dataset

__all__ = ["load_digits_dataset", "gather_digits_dataset"]
