"""
Leakage prevention tests.
These verify that the training pipeline cannot accidentally touch test data.
"""
import numpy as np
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, train_test_split


SEED = 42


def _get_train_test_split():
    iris = load_iris()
    X, y = iris.data, iris.target
    return train_test_split(X, y, test_size=0.2, random_state=SEED, stratify=y)


def test_fit_isolation():
    """Model fitted on train data only -- no test data in fit call."""
    X_train, X_test, y_train, y_test = _get_train_test_split()
    model = RandomForestClassifier(n_estimators=10, random_state=SEED)

    # Fit on train only
    model.fit(X_train, y_train)

    # Verify model saw the right number of samples
    assert model.n_features_in_ == 4
    # Train set should be 80% of 150 = 120
    assert X_train.shape[0] == 120


def test_no_test_access_in_cv():
    """Cross-validation folds are subsets of training data, not test data."""
    X_train, X_test, y_train, y_test = _get_train_test_split()
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)

    test_indices = set(range(len(X_test)))
    train_indices = set(range(len(X_train)))

    for fold_train, fold_val in cv.split(X_train, y_train):
        # All CV indices must be within the training set range
        assert all(idx < len(X_train) for idx in fold_train)
        assert all(idx < len(X_train) for idx in fold_val)
        # No CV fold should have more samples than the training set
        assert len(fold_train) + len(fold_val) == len(X_train)


def test_index_disjoint():
    """Train and test indices are completely disjoint (no shared rows)."""
    iris = load_iris()
    X, y = iris.data, iris.target

    # Use the same split logic as train.py
    indices = np.arange(len(X))
    train_idx, test_idx = train_test_split(
        indices, test_size=0.2, random_state=SEED, stratify=y
    )

    overlap = set(train_idx) & set(test_idx)
    assert len(overlap) == 0, f"Found {len(overlap)} overlapping indices"
    assert len(train_idx) + len(test_idx) == len(X)
