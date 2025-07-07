import numpy as np
from bio_snn.datasets import load_digits_dataset


def test_load_digits_dataset_shapes():
    X_train, X_test, y_train, y_test = load_digits_dataset(test_size=0.2, random_state=0)
    assert X_train.ndim == 2
    assert X_train.shape[0] == y_train.shape[0]
    assert X_test.shape[0] == y_test.shape[0]
    # dataset should be standardized
    assert np.allclose(X_train.mean(), 0, atol=1e-1)
