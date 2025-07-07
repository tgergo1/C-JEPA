"""Utilities for loading and preprocessing the sklearn digits dataset."""

from typing import Tuple
import numpy as np
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


def load_digits_dataset(test_size: float = 0.25, random_state: int | None = None
                        ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Load digits dataset and return normalized train/test splits.

    Parameters
    ----------
    test_size: float, default 0.25
        Fraction of the dataset to reserve for testing.
    random_state: int or None, optional
        Random seed for reproducibility of the train/test split.

    Returns
    -------
    X_train: array, shape (n_train_samples, n_features)
    X_test: array, shape (n_test_samples, n_features)
    y_train: array, shape (n_train_samples,)
    y_test: array, shape (n_test_samples,)
    """
    # load full dataset
    data = load_digits()
    X = data.data.astype(float)
    y = data.target.astype(int)

    # split before scaling to avoid information leakage
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    # standardize using statistics from the training split only
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    return X_train, X_test, y_train, y_test
