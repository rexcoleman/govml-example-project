"""
Shared fixtures for govML example project tests.
"""
import numpy as np
import pytest
from sklearn.datasets import make_classification


@pytest.fixture
def synthetic_iris():
    """Generate a synthetic iris-like dataset for unit tests.
    Avoids depending on sklearn's built-in iris for test isolation.
    """
    X, y = make_classification(
        n_samples=150, n_features=4, n_informative=3,
        n_redundant=1, n_classes=3, n_clusters_per_class=1,
        random_state=42,
    )
    return X, y


@pytest.fixture
def seed():
    return 42
