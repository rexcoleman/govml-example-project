"""
Generate learning curves: performance vs training set size for all 3 algorithms.
Saves results to outputs/learning_curves.json.
"""
import json
from pathlib import Path

import numpy as np
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, learning_curve, train_test_split
from sklearn.svm import SVC

PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUTPUTS_DIR = PROJECT_ROOT / "outputs"
SEED = 42
CV_FOLDS = 5
TRAIN_FRACTIONS = [0.2, 0.4, 0.6, 0.8, 1.0]


def main():
    iris = load_iris()
    X, y = iris.data, iris.target

    X_train, _, y_train, _ = train_test_split(
        X, y, test_size=0.2, random_state=SEED, stratify=y
    )

    cv = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=SEED)

    algorithms = {
        "LogisticRegression": LogisticRegression(
            C=1.0, max_iter=1000, solver="lbfgs", random_state=SEED
        ),
        "RandomForest": RandomForestClassifier(
            n_estimators=100, random_state=SEED
        ),
        "SVM_RBF": SVC(C=1.0, kernel="rbf", gamma="scale", random_state=SEED),
    }

    results = {}
    for name, model in algorithms.items():
        train_sizes_abs, train_scores, val_scores = learning_curve(
            model, X_train, y_train,
            train_sizes=TRAIN_FRACTIONS,
            cv=cv, scoring="accuracy", random_state=SEED,
        )
        results[name] = {
            "train_sizes": train_sizes_abs.tolist(),
            "train_mean": np.mean(train_scores, axis=1).tolist(),
            "train_std": np.std(train_scores, axis=1).tolist(),
            "val_mean": np.mean(val_scores, axis=1).tolist(),
            "val_std": np.std(val_scores, axis=1).tolist(),
        }
        print(f"{name}: val scores at each fraction = "
              f"{[f'{v:.3f}' for v in np.mean(val_scores, axis=1)]}")

    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUTPUTS_DIR / "learning_curves.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nLearning curves saved to {out_path}")


if __name__ == "__main__":
    main()
