"""
Train 3 algorithms x 5 seeds with stratified 5-fold CV.
Saves results to outputs/results.json.
"""
import json
import os
import sys
from pathlib import Path

import numpy as np
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.svm import SVC

PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUTPUTS_DIR = PROJECT_ROOT / "outputs"
SEEDS = [42, 123, 456, 789, 1024]
CV_FOLDS = 5
TEST_SIZE = 0.2


def get_algorithms(seed):
    """Return algorithm dict for a given seed."""
    return {
        "LogisticRegression": LogisticRegression(
            C=1.0, max_iter=1000, solver="lbfgs", random_state=seed
        ),
        "RandomForest": RandomForestClassifier(
            n_estimators=100, max_depth=None, random_state=seed
        ),
        "SVM_RBF": SVC(
            C=1.0, kernel="rbf", gamma="scale", random_state=seed
        ),
    }


def main():
    # Load data
    iris = load_iris()
    X, y = iris.data, iris.target

    results = {}

    for seed in SEEDS:
        # Stratified train/test split -- test set is NEVER used here
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=TEST_SIZE, random_state=seed, stratify=y
        )

        algos = get_algorithms(seed)
        results[str(seed)] = {}

        for name, model in algos.items():
            cv = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=seed)
            scores = cross_val_score(model, X_train, y_train, cv=cv, scoring="accuracy")

            results[str(seed)][name] = {
                "cv_scores": scores.tolist(),
                "mean": float(np.mean(scores)),
                "std": float(np.std(scores)),
            }
            print(f"Seed {seed:>4d} | {name:<22s} | "
                  f"Mean={np.mean(scores):.4f} +/- {np.std(scores):.4f}")

    # Summary across seeds
    print("\n--- Summary (mean across 5 seeds) ---")
    summary = {}
    algo_names = list(get_algorithms(0).keys())
    for name in algo_names:
        means = [results[str(s)][name]["mean"] for s in SEEDS]
        summary[name] = {
            "seed_means": means,
            "grand_mean": float(np.mean(means)),
            "grand_std": float(np.std(means)),
        }
        print(f"{name:<22s} | Grand Mean={np.mean(means):.4f} +/- {np.std(means):.4f}")

    # Save
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    output = {"per_seed": results, "summary": summary, "seeds": SEEDS}
    out_path = OUTPUTS_DIR / "results.json"
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
