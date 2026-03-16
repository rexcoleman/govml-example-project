"""
Run sanity baselines: DummyClassifier (stratified) and shuffled-label experiment.
Saves results to outputs/sanity_baselines.json.
"""
import json
from pathlib import Path

import numpy as np
from sklearn.datasets import load_iris
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split

PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUTPUTS_DIR = PROJECT_ROOT / "outputs"
SEED = 42
CV_FOLDS = 5


def main():
    iris = load_iris()
    X, y = iris.data, iris.target

    X_train, _, y_train, _ = train_test_split(
        X, y, test_size=0.2, random_state=SEED, stratify=y
    )

    cv = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=SEED)

    # Baseline 1: Stratified dummy
    dummy = DummyClassifier(strategy="stratified", random_state=SEED)
    dummy_scores = cross_val_score(dummy, X_train, y_train, cv=cv, scoring="accuracy")
    print(f"Dummy (stratified): {np.mean(dummy_scores):.4f} +/- {np.std(dummy_scores):.4f}")

    # Baseline 2: Shuffled labels (destroys signal)
    rng = np.random.RandomState(SEED)
    y_shuffled = y_train.copy()
    rng.shuffle(y_shuffled)

    rf = RandomForestClassifier(n_estimators=100, random_state=SEED)
    shuffled_scores = cross_val_score(rf, X_train, y_shuffled, cv=cv, scoring="accuracy")
    print(f"RF on shuffled labels: {np.mean(shuffled_scores):.4f} +/- {np.std(shuffled_scores):.4f}")

    # Save
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    output = {
        "dummy_stratified": {
            "cv_scores": dummy_scores.tolist(),
            "mean": float(np.mean(dummy_scores)),
            "std": float(np.std(dummy_scores)),
        },
        "shuffled_labels_rf": {
            "cv_scores": shuffled_scores.tolist(),
            "mean": float(np.mean(shuffled_scores)),
            "std": float(np.std(shuffled_scores)),
        },
    }
    out_path = OUTPUTS_DIR / "sanity_baselines.json"
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nBaselines saved to {out_path}")


if __name__ == "__main__":
    main()
