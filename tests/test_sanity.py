"""
Sanity tests.
Verify that trained models beat trivial baselines and are stable across seeds.
"""
import numpy as np
from sklearn.datasets import load_iris
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.svm import SVC

SEED = 42
SEEDS = [42, 123, 456, 789, 1024]


def _get_train_data():
    iris = load_iris()
    X, y = iris.data, iris.target
    X_train, _, y_train, _ = train_test_split(
        X, y, test_size=0.2, random_state=SEED, stratify=y
    )
    return X_train, y_train


def test_beats_dummy():
    """All 3 algorithms beat the stratified dummy baseline."""
    X_train, y_train = _get_train_data()
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)

    dummy = DummyClassifier(strategy="stratified", random_state=SEED)
    dummy_score = np.mean(cross_val_score(dummy, X_train, y_train, cv=cv))

    algos = [
        LogisticRegression(max_iter=1000, random_state=SEED),
        RandomForestClassifier(n_estimators=100, random_state=SEED),
        SVC(kernel="rbf", random_state=SEED),
    ]

    for model in algos:
        score = np.mean(cross_val_score(model, X_train, y_train, cv=cv))
        assert score > dummy_score, (
            f"{type(model).__name__} ({score:.3f}) did not beat "
            f"dummy ({dummy_score:.3f})"
        )


def test_shuffled_below_real():
    """A model trained on shuffled labels scores worse than on real labels."""
    X_train, y_train = _get_train_data()
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)

    model_real = RandomForestClassifier(n_estimators=100, random_state=SEED)
    real_score = np.mean(cross_val_score(model_real, X_train, y_train, cv=cv))

    rng = np.random.RandomState(SEED)
    y_shuffled = y_train.copy()
    rng.shuffle(y_shuffled)

    model_shuffled = RandomForestClassifier(n_estimators=100, random_state=SEED)
    shuffled_score = np.mean(cross_val_score(model_shuffled, X_train, y_shuffled, cv=cv))

    assert shuffled_score < real_score, (
        f"Shuffled ({shuffled_score:.3f}) >= real ({real_score:.3f}) -- "
        f"model may not be learning real patterns"
    )


def test_all_seeds_above_threshold():
    """All 5 seeds produce accuracy > 0.80 for every algorithm."""
    iris = load_iris()
    X, y = iris.data, iris.target
    threshold = 0.80

    for seed in SEEDS:
        X_train, _, y_train, _ = train_test_split(
            X, y, test_size=0.2, random_state=seed, stratify=y
        )
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)

        algos = {
            "LogReg": LogisticRegression(max_iter=1000, random_state=seed),
            "RF": RandomForestClassifier(n_estimators=100, random_state=seed),
            "SVM": SVC(kernel="rbf", random_state=seed),
        }

        for name, model in algos.items():
            score = np.mean(cross_val_score(model, X_train, y_train, cv=cv))
            assert score > threshold, (
                f"Seed {seed}, {name}: {score:.3f} < {threshold}"
            )
