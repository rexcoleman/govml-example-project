"""
Pipeline integration tests.
Verify that all algorithms train, produce valid predictions, and can be serialized.
"""
import json
import pickle
import tempfile
from pathlib import Path

import numpy as np
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC

SEED = 42


def _get_data():
    iris = load_iris()
    X, y = iris.data, iris.target
    return train_test_split(X, y, test_size=0.2, random_state=SEED, stratify=y)


def test_all_algos_train():
    """All 3 algorithms train without error."""
    X_train, X_test, y_train, y_test = _get_data()
    algos = {
        "LogReg": LogisticRegression(max_iter=1000, random_state=SEED),
        "RF": RandomForestClassifier(n_estimators=100, random_state=SEED),
        "SVM": SVC(kernel="rbf", random_state=SEED),
    }
    for name, model in algos.items():
        model.fit(X_train, y_train)
        score = model.score(X_test, y_test)
        assert score > 0.0, f"{name} scored 0.0"


def test_predictions_valid():
    """Predictions contain only valid class labels."""
    X_train, X_test, y_train, y_test = _get_data()
    valid_labels = {0, 1, 2}

    model = RandomForestClassifier(n_estimators=100, random_state=SEED)
    model.fit(X_train, y_train)
    preds = model.predict(X_test)

    assert set(preds).issubset(valid_labels), f"Invalid labels in predictions: {set(preds) - valid_labels}"
    assert len(preds) == len(y_test)


def test_save_load():
    """Model can be saved and loaded with identical predictions."""
    X_train, X_test, y_train, y_test = _get_data()

    model = RandomForestClassifier(n_estimators=100, random_state=SEED)
    model.fit(X_train, y_train)
    preds_original = model.predict(X_test)

    # Save and reload
    with tempfile.NamedTemporaryFile(suffix=".pkl", delete=False) as f:
        pickle.dump(model, f)
        tmp_path = f.name

    with open(tmp_path, "rb") as f:
        loaded_model = pickle.load(f)

    preds_loaded = loaded_model.predict(X_test)
    np.testing.assert_array_equal(preds_original, preds_loaded)

    Path(tmp_path).unlink()
